import faiss
import json
from thefuzz import process
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os 
import torch

from gnn_encoder import GATRanker, rank_movies_by_attention
from graph_utils import (
    extract_subgraph_from_global,
    convert_nx_to_pyg,
)

os.environ['HF_HOME'] = 'D:/huggingface_cache'
TEXT_EMB_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def find_best_match(query, choices, score_cutoff=85):
    """
    Finds the best fuzzy match for a query among a list of choices.
    Choices may be a list of names or (name, id) tuples.
    """
    try:
        actual_choices = [choice[0] if isinstance(choice, tuple) else choice for choice in choices]
        best_match, score = process.extractOne(query, actual_choices)
        if score >= score_cutoff:
            return best_match
    except:
        pass
    return None


def find_best_name(name, entity_asset):
    """
    Finds the best matching name for a given entity using embeddings.
    """
    all_names_in_db = list(entity_asset["mapping"].values())
    best_match_name = find_best_match_with_embedding(
        name_to_find=name,
        entity_text_faiss_index=entity_asset["text_index"],
        entity_names_in_db=all_names_in_db
    )
    return best_match_name


def load_recommendation_assets(base_path="./dataset/faiss"):
    """
    Load all recommendation assets (FAISS indices and mappings).
    """
    print("Loading all recommendation assets...")
    try:
        assets = {}
        entities = ["movie", "actor", "director", "genre"]
        
        for entity in entities:
            gnn_emb_path = f"{base_path}/{entity}_embeddings.faiss"
            text_emb_path = f"{base_path}/{entity}_text_embeddings.faiss"
            mapping_path = f"{base_path}/{entity}_mapping.json"
            
            assets[entity] = {
                "gnn_index": faiss.read_index(gnn_emb_path),
                "text_index": faiss.read_index(text_emb_path),
                "mapping": json.load(open(mapping_path, 'r', encoding='utf-8'))
            }
            print(f"Loaded {entity} assets.")
            
        print("All recommendation assets loaded successfully.")
        return assets
    except Exception as e:
        print(f"Failed to load recommendation assets: {e}")
        return None


import re
def clean_cypher_query(query: str) -> str:
    """
    Extracts Cypher query text from a string and cleans extra whitespace/newlines.
    """
    match = re.search(r'```(?:cypher)?\s*([^`]+)```', query, re.DOTALL)
    if match:
        query = match.group(1).strip()
    return re.sub(r'\s+', ' ', query).strip()


def find_best_match_with_embedding(name_to_find, entity_text_faiss_index, entity_names_in_db):
    """
    Finds the best match for a name using semantic embeddings and FAISS search.
    """
    if not name_to_find or not entity_text_faiss_index:
        return None
    
    query_embedding = TEXT_EMB_MODEL.encode([name_to_find])
    D, I = entity_text_faiss_index.search(np.array(query_embedding, dtype=np.float32), 1)
    
    DISTANCE_THRESHOLD = 1.2 
    if I.size > 0 and D[0][0] < DISTANCE_THRESHOLD:
        matched_index = I[0][0]
        best_match_name = entity_names_in_db[matched_index]
        return best_match_name
    return None


def format_candidates_for_prompt(rerank_dict: dict) -> str:
    """
    Convert rerank_dict into a string format suitable for an LLM prompt.
    """
    output_str = ""
    for category, movies in rerank_dict.items():
        output_str += f"\n## Recommended based on {category.capitalize()}:\n"
        for movie in movies:
            title = movie.get('title', 'N/A')
            importance = movie.get('importance', 0)
            overview = movie.get('overview', 'No overview available.').strip()
            output_str += f"- Title: {title}\n  GNN Score: {importance:.4f}\n  Overview: {overview}\n"
    return output_str


def retrieve_movies_by_preference(preferences, assets, graph, chains):
    """
    Retrieve movies from Neo4j based on user preferences.
    Uses Cypher generation chain and optional genre mapping.
    """
    cypher_chain = chains['cypher_gen']
    genre_mapper_chain = chains['genre_mapper']

    def process_actor(name, asset):
        return find_best_name(name, asset)

    def process_director(name, asset):
        return find_best_name(name, asset)
    
    def process_movie(name, asset):
        return find_best_name(name, asset)

    def process_genre(name, asset):
        mapped_genre = genre_mapper_chain.invoke({
            "user_genre": name,
            "available_genres": ", ".join(asset["mapping"].values())
        })['text'].strip()
        if mapped_genre != 'None':
            return mapped_genre
        return None

    handlers = {
        "actors": process_actor, "directors": process_director,
        "genres": process_genre, "movies": process_movie,
    }

    for entity_type, handler in handlers.items():
        if preferences.get(entity_type):
            asset_key = entity_type[:-1] if entity_type.endswith('s') else entity_type
            entity_asset = assets.get(asset_key)
            if entity_asset:
                processed_list = [
                    p for name in preferences[entity_type] if (p := handler(name, entity_asset))
                ]
                preferences[entity_type] = processed_list
    
    retrieved_dict = {}

    def preferences_to_question(entity_type, values):
        vals_str = ", ".join(values)
        entity_name = entity_type[:-1] if entity_type.endswith('s') else entity_type
        return f"Find movies related to {entity_name}(s): {vals_str}"

    for entity_type, values in preferences.items():
        if not values:
            continue
        question = preferences_to_question(entity_type, values)
        print(f"Generating query for [{entity_type}]")
        print(f"Question: {question}")
        
        cypher_query = cypher_chain.invoke({
            "schema": graph.schema, 
            "question": question
        })['text']
        print(f"Generated Cypher: {cypher_query}")
        
        clean_cypher = clean_cypher_query(cypher_query)
        movies = graph.query(clean_cypher)
        retrieved_dict[entity_type] = movies

    return retrieved_dict


def fetch_movie_quality_scores_from_dict(movie_dict):
    """
    Extract average ratings and rating counts directly from movie_overview_dict.
    Returns {movie_id: (avg_rating, rating_count)} so that val[0], val[1] indexing works.
    """
    quality_map = {}
    for key, val in movie_dict.items():
        try:
            movie_id = int(key.replace("movie_", ""))
            # val = (title, overview, avgRating, ratingCount)
            avg_rating = float(val[2]) if val[2] not in (None, "No avg rating") else 0.0
            rating_count = int(val[3]) if val[3] not in (None, "No rating counts") else 0
            quality_map[movie_id] = (avg_rating, rating_count)
        except Exception:
            continue
    return quality_map

def fetch_movie_quality_scores_from_nodes(graph, nodes):
    """
    Fetch avg_rating and rating_count directly from Movie nodes.
    `nodes` contains identifiers like "movie_123", so we extract the int IDs.
    """
    # 1. nodes 리스트에서 "movie_123" → 123 으로 변환
    movie_ids = [
        int(n.replace("movie_", ""))
        for n in nodes if isinstance(n, str) and n.startswith("movie_")
    ]
    if not movie_ids:
        return {}

    # 2. Cypher 쿼리 (Movie 노드 속성만 사용)
    cypher_query = f"""
    MATCH (u:User)-[r:RATED]->(m:Movie)
    WHERE m.movieId IN [{",".join(map(str, movie_ids))}]
    RETURN m.movieId AS movie_id,
           avg(r.rating) AS avg_rating,
           count(r.rating) AS rating_count
    """

    # 3. 실행 및 매핑
    try:
        rows = graph.query(clean_cypher_query(cypher_query))
        quality_map = {
            int(row["movie_id"]): (
                float(row.get("avg_rating", 0.0) or 0.0),
                int(row.get("rating_count", 0) or 0)
            )
            for row in rows
        }
        return quality_map
    except Exception as e:
        print(f"Rating query failed: {e}")
        return {}

def fetch_movie_overviews(graph, movie_ids):
    """
    Fetch movie titles and overviews for given movie_ids.
    """
    if not movie_ids:
        return {}

    cypher_query = f"""
    MATCH (m:Movie)
    WHERE m.movieId IN [{",".join(map(str, movie_ids))}]
    RETURN m.movieId AS id, m.title AS title, m.overview AS overview
    """
    try:
        rows = graph.query(clean_cypher_query(cypher_query))
        return {
            f"movie_{row['id']}": (
                row.get("title", "Unknown"),
                row.get("overview", "No overview available")
            )
            for row in rows
        }
    except Exception as e:
        print(f"Overview query failed: {e}")
        return {}


def build_pyg_from_subgraph(subgraph_nx, assets):
    """
    Convert a NetworkX subgraph to PyTorch Geometric Data.
    """
    data, nodes = convert_nx_to_pyg(subgraph_nx, assets)
    if data.x is None or data.x.size(0) == 0:
        return None, None
    return data, nodes


def run_gat_model(data):
    """
    Run the GATRanker model on PyG Data.
    """
    model = GATRanker(in_channels=data.x.shape[1], hidden_channels=data.x.shape[1])
    model.eval()
    with torch.no_grad():
        _, attention_scores = model(data.x, data.edge_index)
    return attention_scores


def enrich_movies_with_overview(top_movies, movie_overview_dict, overview_map):
    """
    Enrich ranked movies with overview and title information.
    """
    final_movies = []
    for item in top_movies:
        movie_id = item["movid_id"]
        if movie_id in movie_overview_dict:
            title, overview = movie_overview_dict[movie_id]
        elif movie_id in overview_map:
            title, overview = overview_map[movie_id]
        else:
            title, overview = "Unknown", "No overview available"
        item["title"] = title
        item["overview"] = overview
        final_movies.append(item)
    return final_movies


def semantic_filter_movies(movie_list, query, semantic_top_k=10):
    """
    SentenceTransformer를 이용해 retrieved_dict에서 query와 가장 유사한 영화만 필터링
    
    Args:
        retrieved_dict (dict): {movie_id: {"title": ..., "overview": ...}} 형태
        query (str): 사용자 검색 질의
        semantic_top_k (int): 상위 몇 개를 추릴지
    
    Returns:
        dict: 상위 semantic_top_k 영화만 담은 dict
    """
    # 후보 영화 overview 수집
    overviews = [m["m.overview"] for m in movie_list]
    movie_keys = [m["m.movieId"] for m in movie_list]

    if not overviews:
        return movie_list

    # 임베딩 계산
    doc_emb = TEXT_EMB_MODEL.encode(overviews, convert_to_tensor=True, show_progress_bar=False)
    query_emb = TEXT_EMB_MODEL.encode(query, convert_to_tensor=True)

    # cosine similarity 계산
    scores = util.pytorch_cos_sim(query_emb, doc_emb)[0].cpu().numpy()

    # 상위 semantic_top_k 인덱스 추출
    top_idx = scores.argsort()[::-1][:semantic_top_k]

    # 필터링된 dict 생성
    filtered_list = [movie_list[i] for i in top_idx]

    return filtered_list


def find_movies_with_faiss(preferences, assets, graph, chains, global_graph_nx, query,
                           top_k=5, alpha=0.7, beta=0.3):
    """
    Retrieve candidate movies and rerank them using GAT attention + user ratings.
    """
    retrieved_dict = retrieve_movies_by_preference(preferences, assets, graph, chains)

    if not retrieved_dict:
        return retrieved_dict

    rerank_dict = {}
    for key, movie_list in retrieved_dict.items():
        if not movie_list:
            rerank_dict[key] = []
            continue

        movie_list = semantic_filter_movies(movie_list, query)

        movie_dict = {
            f"movie_{m.get('m.movieId')}": (
                m.get('m.title', 'Unknown'),
                m.get('m.overview', 'No overview available')
                #m.get('m.avgRating', 'No avg rating')
                #m.get('m.ratingCount', 'No rating counts')
            )
            for m in movie_list if 'm.movieId' in m
        }
        if not movie_dict:
            rerank_dict[key] = []
            continue

        movie_ids = list(movie_dict.keys())
        subgraph_nx = extract_subgraph_from_global(global_graph_nx, movie_ids, assets)
        if subgraph_nx.number_of_nodes() == 0:
            rerank_dict[key] = []
            continue

        data, nodes = build_pyg_from_subgraph(subgraph_nx, assets)
        if data is None:
            rerank_dict[key] = []
            continue

        try:
            attention_scores = run_gat_model(data)
        except Exception as e:
            print(f"GATRanker failed for key={key}: {e}")
            rerank_dict[key] = []
            continue

        #quality_map = fetch_movie_quality_scores_from_dict(movie_dict)
        quality_map = fetch_movie_quality_scores_from_nodes(graph, nodes)

        sorted_movies = rank_movies_by_attention(
            attention_scores, data, nodes, subgraph_nx, quality_map,
            alpha=alpha, beta=beta
        )
        top_movies = sorted_movies[:top_k]

        rerank_ids = [int(item["movid_id"].replace("movie_", "")) for item in top_movies]
        overview_map = fetch_movie_overviews(graph, rerank_ids)

        final_movies = enrich_movies_with_overview(top_movies, movie_dict, overview_map)
        rerank_dict[key] = final_movies

    return rerank_dict
