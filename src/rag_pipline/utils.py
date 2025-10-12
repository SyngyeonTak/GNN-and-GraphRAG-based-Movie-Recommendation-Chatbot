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
    get_similar_movies_from_seeds,
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

            # 기본 구조
            entity_assets = {
                "gnn_index": faiss.read_index(gnn_emb_path),
                "text_index": faiss.read_index(text_emb_path),
                "mapping": json.load(open(mapping_path, 'r', encoding='utf-8'))
            }

            # movie일 경우 overview 임베딩도 추가
            if entity == "movie":
                overview_emb_path = f"{base_path}/{entity}_overview_embeddings.faiss"
                if os.path.exists(overview_emb_path):
                    entity_assets["overview_index"] = faiss.read_index(overview_emb_path)
                    print("  ↳ Loaded movie overview embeddings.")
                else:
                    print("  ⚠️ movie_overview_embeddings.faiss not found — skipping.")

            assets[entity] = entity_assets
            print(f"Loaded {entity} assets.")
            
        print("✅ All recommendation assets loaded successfully.")
        return assets

    except Exception as e:
        print(f"❌ Failed to load recommendation assets: {e}")
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


def format_candidates_for_prompt(rerank_list):
    """
    Convert rerank_dict into a string format suitable for an LLM prompt.
    """
    output_str = ""
    for movie in rerank_list:
        title = movie.get('title', 'N/A')
        importance = movie.get('importance', 0)
        overview = movie.get('overview', 'No overview available.').strip()
        output_str += f"- Title: {title}\n  GNN Score: {importance:.4f}\n  Overview: {overview}\n"

    return output_str


def combine_preferences_to_question(preferences):
    """
    Combine multiple entity preferences into one natural language question.

    Special rule:
    - If only 'movies' (or 'movie') key is provided, simply return that movie itself.
      (Do NOT generate 'related to' or 'similar' style questions)
    """

    # 소문자 키 일관 처리
    preferences = {k.lower(): v for k, v in preferences.items()}

    # 1️⃣ 단일 영화만 있는 경우 → 그대로 반환
    if preferences.get("movies") and not any(
        preferences.get(k) for k in ["actors", "directors", "genres", "countries", "years"]
    ):
        movie_title = preferences["movies"][0]
        return f"Find the movie titled {movie_title}."

    # 2️⃣ 일반 조합 처리
    parts = []

    for entity_type, values in preferences.items():
        if not values:
            continue

        vals_str = ", ".join(values)
        entity = entity_type.lower()

        if entity in ["actor", "actors"]:
            parts.append(f"starring {vals_str}")
        elif entity in ["director", "directors"]:
            parts.append(f"directed by {vals_str}")
        elif entity in ["genre", "genres"]:
            parts.append(f"belonging to the genres {vals_str}")
        elif entity in ["country", "countries"]:
            parts.append(f"produced in {vals_str}")
        elif entity in ["year", "years"]:
            parts.append(f"released in {vals_str}")
        elif entity in ["movie", "movies"]:
            # 영화가 다른 엔티티와 같이 존재할 경우만 related 처리
            parts.append(f"related to the movie {vals_str}")
        else:
            parts.append(f"related to {entity_type.lower()}: {vals_str}")

    if not parts:
        return "Find movies."

    # 마지막 항목만 and로 연결
    if len(parts) > 1:
        question_body = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    else:
        question_body = parts[0]

    return f"Find movies {question_body}."

def retrieve_movies_by_preference(preferences, assets, graph, chains):
    """
    Retrieve movies from Neo4j based on user preferences.
    Uses Cypher generation chain and optional genre mapping.
    """
    cypher_chain = chains['cypher_gen']
    cypher_comb_chain = chains['cypher_combined_gen']
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
    
    retrieved_list = []

    comb_question = combine_preferences_to_question(preferences)

    cypher_comb_query = cypher_comb_chain.invoke({
            "schema": graph.schema, 
            "question": comb_question
        })['text']
        
    print(f"Generated cypher_comb_query: {cypher_comb_query}")
    clean_comb_cypher = clean_cypher_query(cypher_comb_query)
    movies_comb = graph.query(clean_comb_cypher)

    retrieved_list = movies_comb

    return retrieved_list


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
        movie_id = item["movie_id"]
        if movie_id in movie_overview_dict:
            movie_dict = movie_overview_dict[movie_id]
            title = movie_dict['title']
            overview = movie_dict['overview']
        elif movie_id in overview_map:
            title, overview = overview_map[movie_id]
        else:
            title, overview = "Unknown", "No overview available"
        item["title"] = title
        item["overview"] = overview
        final_movies.append(item)
    return final_movies


def semantic_filter_movies(movie_list, query, assets, semantic_top_k=10):
    # movie asset 로드
    overview_index = assets["movie"]["overview_index"]

    # movie_list 내 ID 추출
    movie_ids = [m["m.movieId"] for m in movie_list]

    # query 임베딩 계산
    query_emb = TEXT_EMB_MODEL.encode(query, convert_to_numpy=True).astype('float32')
    
    # query_emb을 이용해 전체 인덱스에서 검색 (cosine similarity 기반)
    #overview_index = faiss.IndexIDMap2(overview_index)
    faiss.normalize_L2(query_emb.reshape(1, -1))
    _, I = overview_index.search(query_emb.reshape(1, -1), overview_index.ntotal)  # 여유 있게 검색

    # I 순서대로 movie_ids 필터링
    matched_ids = [index + 1 for index in I[0] if (index + 1) in movie_ids]

    # movie_ids의 순서를 matched_ids 순서로 정렬
    sorted_list = sorted(movie_list, key=lambda m: matched_ids.index(m["m.movieId"]) 
                         if m["m.movieId"] in matched_ids else float('inf'))

    # 상위 semantic_top_k만 선택
    filtered_list = sorted_list[:semantic_top_k]

    return filtered_list

def find_movies_with_faiss(preferences, assets, graph, chains, global_graph_nx, query,
                           top_k=5, alpha=0.7, beta=0.3):
    """
    Retrieve candidate movies and rerank them using GAT attention + user ratings.
    Automatically excludes seed movie mentioned in user query (e.g., "movies like Interstellar").
    """
    # Step 1. Retrieve candidates by preferences
    retrieved_list = retrieve_movies_by_preference(preferences, assets, graph, chains)
    if not retrieved_list:
        return []

    # Step 2. Semantic filtering
    movie_list = semantic_filter_movies(retrieved_list, query, assets)

    # Step 3. Build dictionary for titles and overviews
    movie_dict = {
        f"movie_{m.get('m.movieId')}": {
            "movie_id": f"movie_{m.get('m.movieId')}",
            "title": m.get('m.title', 'Unknown'),
            "overview": m.get('m.overview', 'No overview available')
        }
        for m in movie_list if 'm.movieId' in m
    }

    movie_ids = list(movie_dict.keys())

    # Step 4. FAISS-based similarity expansion
    if len(movie_ids) < 5:
        sim_movie_ids = get_similar_movies_from_seeds(movie_ids, assets)
        movie_ids = movie_ids + sim_movie_ids
    
    rec_movie_ids = list(set(movie_ids))

    # Step 5. Build GNN subgraph and compute attention
    subgraph_nx = extract_subgraph_from_global(global_graph_nx, rec_movie_ids)
    data, nodes = build_pyg_from_subgraph(subgraph_nx, assets)
    attention_scores = run_gat_model(data)
    quality_map = fetch_movie_quality_scores_from_nodes(graph, nodes)

    # Step 6. Rank movies by attention and quality
    sorted_movies = rank_movies_by_attention(
        attention_scores, data, nodes, subgraph_nx, quality_map,
        alpha=alpha, beta=beta
    )

    # Step 7. Filter to only relevant movie_ids
    rec_movies = [
        m for m in sorted_movies
        if m.get('movie_id') in rec_movie_ids
    ][:top_k]

    # Step 8. Fetch overviews and enrich results
    rerank_ids = [int(m["movie_id"].replace("movie_", "")) for m in rec_movies]
    overview_map = fetch_movie_overviews(graph, rerank_ids)
    final_movies = enrich_movies_with_overview(rec_movies, movie_dict, overview_map)

    return final_movies