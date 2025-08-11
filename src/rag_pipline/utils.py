import faiss
import json
from thefuzz import process
import numpy as np
from sentence_transformers import SentenceTransformer
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
    try:
        # choices가 (이름, ID) 형태의 튜플 리스트일 수 있으므로 이름만 추출
        actual_choices = [choice[0] if isinstance(choice, tuple) else choice for choice in choices]
        best_match, score = process.extractOne(query, actual_choices)

        if score >= score_cutoff:
            return best_match
    except:
        pass
    return None

def load_recommendation_assets(base_path="./dataset/faiss"):
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
            print(f"✅ Loaded {entity} assets.")
            
        print("✅ All recommendation assets loaded successfully.")
        return assets
    except Exception as e:
        print(f"❌ Failed to load recommendation assets: {e}")
        return None

def clean_cypher_query(query: str) -> str:
    if "```cypher" in query:
        query = query.split("```cypher", 1)[1]
    if "```" in query:
        query = query.split("```", 1)[0]
    return query.replace('\n', ' ').strip()

def find_best_match_with_embedding(name_to_find, entity_text_faiss_index, entity_names_in_db):
    if not name_to_find or not entity_text_faiss_index:
        return None
    
    # 1. 찾으려는 이름을 텍스트 임베딩으로 변환
    query_embedding = TEXT_EMB_MODEL.encode([name_to_find])
    
    # 2. FAISS 인덱스에서 가장 유사한 임베딩 검색 (상위 1개)
    # D: 거리, I: 인덱스
    D, I = entity_text_faiss_index.search(np.array(query_embedding, dtype=np.float32), 1)
    
    # 특정 임계값(threshold)을 기준으로 너무 거리가 멀면 매칭 실패로 간주할 수 있습니다.
    # 이 값은 실험을 통해 조정하는 것이 좋습니다.
    DISTANCE_THRESHOLD = 1.2 
    
    if I.size > 0 and D[0][0] < DISTANCE_THRESHOLD:
        matched_index = I[0][0]
        best_match_name = entity_names_in_db[matched_index]
        return best_match_name
    
    return None

def format_candidates_for_prompt(rerank_dict: dict) -> str:
    """
    rerank_dict를 LLM 프롬프트에 넣기 좋은 문자열로 변환합니다.
    """
    output_str = ""
    for category, movies in rerank_dict.items():
        output_str += f"\n## Recommended based on {category.capitalize()}:\n"
        for movie in movies: # 각 카테고리별 상위 3개만 프롬프트에 포함하여 간결성 유지
            title = movie.get('title', 'N/A')
            importance = movie.get('importance', 0)
            overview = movie.get('overview', 'No overview available.').strip()
            output_str += f"- Title: {title}\n  GNN Score: {importance:.4f}\n  Overview: {overview}\n"
    return output_str

def retrieve_movies_by_preference(preferences, assets, graph, chains):
    cypher_chain = chains['cypher_gen']
    genre_mapper_chain = chains['genre_mapper']

    def find_best_name(name, entity_asset):
        all_names_in_db = list(entity_asset["mapping"].values())
        best_match_name = find_best_match_with_embedding(
            name_to_find=name,
            entity_text_faiss_index=entity_asset["text_index"],
            entity_names_in_db=all_names_in_db
        )
        return best_match_name

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
    
    # --- 2. 각 선호도별로 개별 쿼리 생성 및 실행 ---

    retrieved_dict = {}

    # 개별 질문 생성을 위한 간단한 함수
    def preferences_to_question(entity_type, values):
        vals_str = ", ".join(values)
        entity_name = entity_type[:-1] if entity_type.endswith('s') else entity_type
        return f"Find movies related to {entity_name}(s): {vals_str}"

    for entity_type, values in preferences.items():
        # 해당 선호도에 값이 없으면 건너뛰기
        if not values:
            continue
        
        # 1. 개별 질문 생성
        question = preferences_to_question(entity_type, values)
        print(f"--- Generating query for [{entity_type}] ---")
        print(f"Question: {question}")
        
        # 2. 개별 Cypher 쿼리 생성
        cypher_query = cypher_chain.invoke({
            "schema": graph.schema, 
            "question": question
        })['text']
        print(f"Generated Cypher: {cypher_query}")
        
        # 3. 개별 쿼리 실행
        clean_cypher = clean_cypher_query(cypher_query) # clean_cypher_query 함수가 있다고 가정
        movies = graph.query(clean_cypher)
        
        # 4. 결과를 최종 딕셔너리에 저장
        retrieved_dict[entity_type] = movies

    # 5. 모든 작업이 끝난 후, 결과 딕셔너리 반환
    return retrieved_dict

def find_movies_with_faiss(preferences, assets, graph, chains, global_graph_nx, top_k=5):
    """
    '취향 벡터'를 직접 생성하여 Faiss로 영화를 검색하고 추천합니다.
    """
    # 1. 취향 JSON으로 '취향 벡터' 직접 생성

    retrieved_dict = retrieve_movies_by_preference(preferences, assets, graph, chains)

    if retrieved_dict is None:
        return [{"title": "Sorry, I couldn't find good examples for your taste. Please try another preference."}]
    
    rerank_dict = {}
    for key, movie_list in retrieved_dict.items():
        movie_titles = [f'{movie_dict['m.title']}' for movie_dict in movie_list]
        movie_overview_dict = {f'{movie_dict['m.title']}' : f'{movie_dict['m.overview']}' for movie_dict in movie_list}
        subgraph_nx = extract_subgraph_from_global(global_graph_nx, movie_titles)
        print(subgraph_nx) # 서브그래프 데이터 확인
        
        data, nodes = convert_nx_to_pyg(subgraph_nx, assets)

        model = GATRanker(in_channels=data.x.shape[1], hidden_channels=data.x.shape[1])
        model.eval()
        with torch.no_grad():
            _, attention_scores = model(data.x, data.edge_index)

        sorted_movies = rank_movies_by_attention(attention_scores, data, nodes, subgraph_nx)
        top_movies = sorted_movies[:10]

        for idx in range(len(top_movies) - 1, -1, -1):
            movie_title = top_movies[idx]['title']
            
            try:
                # 1. 딕셔너리에서 overview를 가져오려고 시도합니다.
                movie_overview = movie_overview_dict[movie_title]
                # 2. 성공하면, overview를 top_movies에 추가합니다.
                top_movies[idx]['overview'] = movie_overview
                
            except KeyError:
                # 3. movie_title이 딕셔너리에 없어 KeyError가 발생하면,
                #    해당 영화를 top_movies 리스트에서 삭제합니다.
                print(f"Warning: Overview for '{movie_title}' not found. Removing from list.")
                del top_movies[idx]

            rerank_dict[key] = top_movies
        
    return rerank_dict