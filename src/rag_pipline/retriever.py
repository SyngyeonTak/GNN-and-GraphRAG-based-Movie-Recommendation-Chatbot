from utils import find_best_match, load_recommendation_assets, clean_cypher_query
import re
import json
import numpy as np

# ----------------------------------------------------------------------
# 3. '취향 벡터'를 직접 생성하는 핵심 함수 (신규)
# ----------------------------------------------------------------------
def create_taste_vector(preferences, assets, llm, movie_suggester_chain, genre_mapper_chain):
    """
    (UPGRADED) 취향 JSON과 LLM 추천 영화를 모두 사용하여 '하이브리드 취향 벡터'를 생성합니다.
    - 장르 매칭 실패 시 LLM으로 의미상 가장 유사한 장르를 찾습니다. (Q1 해결)
    - LLM이 추천한 대표 영화들의 벡터를 취향 벡터에 추가합니다. (Q2 해결)
    """
    found_vectors = []
    
    # --- 1. 직접적인 취향 정보 (장르, 배우, 감독) 벡터화 ---
    for entity_type in ["actors", "directors", "genres"]:
        if entity_type not in preferences or not preferences[entity_type]:
            continue
            
        asset_key = entity_type[:-1]
        entity_asset = assets.get(asset_key)
        if not entity_asset: continue


        all_keys_in_db = []
        all_names_in_db = []

        for key, name in entity_asset["mapping"].items():
            all_keys_in_db.append(key)
            all_names_in_db.append(name)

        for name in preferences[entity_type]:
            # 1-A. 먼저 철자 기반 퍼지 매칭 시도
            
            best_match_name = find_best_match(name, all_names_in_db)
            
            # 1-B. (Q1 해결) 장르 매칭이 실패했고, 타입이 'genres'인 경우, LLM으로 의미 매칭 시도
            if not best_match_name and entity_type == "genres":
                print(f"⚠️ No direct match for genre '{name}'. Trying semantic mapping...")
                mapped_genre = genre_mapper_chain.invoke({
                    "user_genre": name,
                    "available_genres": ", ".join(all_names_in_db)
                })['text'].strip()

                if mapped_genre != 'None':
                    best_match_name = mapped_genre
                    print(f"✅ Semantically mapped '{name}' to '{best_match_name}'")

            if best_match_name:
                #entity_id = entity_asset["mapping"][best_match_name]
                entity_id = all_names_in_db.index(best_match_name) 
                vector = entity_asset["index"].reconstruct(entity_id)
                found_vectors.append(vector)
                print(f"👍 Found vector for '{name}' via matched entity '{best_match_name}'")

    # --- 2. (Q2 해결) LLM 추천 영화 벡터 추가 ---
    preference_str = ", ".join([item for sublist in preferences.values() for item in sublist])
    if preference_str:
        print(f"\nEnhancing taste vector with LLM-suggested movies for preference: '{preference_str}'")
        suggested_movies_str = movie_suggester_chain.invoke({"preference": preference_str})['text']
        suggested_movie_titles = [title.strip() for title in suggested_movies_str.split(',')]
        
        all_movie_titles_in_db = list(assets["movie"]["mapping"].values())
        for movie_title in suggested_movie_titles:
            best_match_movie = find_best_match(movie_title, all_movie_titles_in_db)
            if best_match_movie:
                #movie_id = assets["movie"]["mapping"][best_match_movie]
                movie_id = all_movie_titles_in_db.index(best_match_movie)
                vector = assets["movie"]["index"].reconstruct(movie_id)
                found_vectors.append(vector)
                print(f"👍 Found vector for LLM-suggested movie '{movie_title}' by matching to '{best_match_movie}'")

    # --- 3. 최종 취향 벡터 생성 ---
    if not found_vectors:
        print("⚠️ No valid vectors found for the given preferences.")
        return None
        
    session_taste_vector = np.mean(found_vectors, axis=0)
    print(f"\n✅ Created a hybrid taste vector from {len(found_vectors)} sources.")
    return session_taste_vector

# ----------------------------------------------------------------------
# 4. Faiss 검색 함수 로직 변경 (수정)
# ----------------------------------------------------------------------
def find_movies_with_faiss(preferences, assets, llm, movie_suggester_chain, genre_mapper_chain, top_k=5):
    """
    '취향 벡터'를 직접 생성하여 Faiss로 영화를 검색하고 추천합니다.
    """
    # 1. 취향 JSON으로 '취향 벡터' 직접 생성
    session_taste_vector = create_taste_vector(preferences, assets, llm, movie_suggester_chain, genre_mapper_chain)

    if session_taste_vector is None:
        return [{"title": "Sorry, I couldn't find good examples for your taste. Please try another preference."}]

    # 2. 생성된 취향 벡터로 '영화' Faiss 인덱스 검색
    movie_index = assets["movie"]["index"]
    #movie_mapping_id_to_title = {v: k for k, v in assets["movie"]["mapping"].items()} # ID -> Title 맵
    movie_mapping_id_to_title = {k: v for k, v in assets["movie"]["mapping"].items()}

    search_vector = session_taste_vector.astype('float32').reshape(1, -1)
    distances, indices = movie_index.search(search_vector, k=top_k)
    
    # 3. 결과 변환
    recommendations = []
    for i in indices[0]:
        # Faiss 인덱스 ID로 영화 제목 찾기
        index = str(i)
        movie_title = movie_mapping_id_to_title.get(index, f"Unknown Movie (ID: {index})")
        recommendations.append({"title": movie_title})
        
    return recommendations


# MODIFIED: This function now just asks a question or triggers the Faiss search.
# The state management logic is moved to the main retriever.

def personalized_recommendation(state, llm, assets, chains):
    """개인화 추천 요청을 Faiss 파이프라인으로 처리"""
    print("\n[Executing Personalized Recommendation with Faiss]")
    preferences = state.get('preferences', {})

    if not any(preferences.values()):
        state['waiting_for_preference'] = True
        return "Of course! To give you a good recommendation, could you tell me about a genre, actor, or a movie you've enjoyed recently?"
    else:
        print(f"🧠 Using preferences to find recommendations: {preferences}")
        return find_movies_with_faiss(
            preferences, assets, llm, chains['movie_suggester'], chains['genre_mapper']
        )

def fact_based_search(query, graph, assets, chains):
    """사실 기반 검색을 수행 (엔티티 추출 -> Fuzzy Matching -> Cypher 실행)"""
    print("\n[Executing Fact-Based Search]")
    extracted_entity = chains['entity_extractor'].invoke({"user_input": query})['text'].strip()
    if not extracted_entity:
        return "I'm sorry, I couldn't identify a movie or person in your question."

    all_known_names = list(assets["movie"]["mapping"].keys()) + \
                      list(assets["actor"]["mapping"].keys()) + \
                      list(assets["director"]["mapping"].keys())
    
    corrected_entity = find_best_match(extracted_entity, all_known_names)
    if not corrected_entity:
        return f"I'm sorry, I couldn't find anything matching '{extracted_entity}'."
    print(f"✅ Corrected Entity: '{corrected_entity}'")
    
    clean_query = query.lower().replace(extracted_entity.lower(), corrected_entity)
    generated_cypher = chains['cypher_gen'].invoke({"schema": graph.schema, "question": clean_query})['text']
    cleaned_cypher = clean_cypher_query(generated_cypher)
    print(f"🧠 Generated Cypher: {cleaned_cypher}")

    try:
        return graph.query(cleaned_cypher)
    except Exception as e:
        print(f"❌ Cypher execution failed: {e}")
        return "Sorry, I ran into an error trying to find that information."

# 2-D. Chit-Chat
def chit_chat(user_query):
    """
    Handles simple, conversational, or off-topic queries.
    """
    print("\n[Handling Chit-Chat]")
    response = "I'm a movie recommendation chatbot. What kind of movie are you looking for? 😊"
    print(f"✅ Response: {response}")
    return response

# MODIFIED: The main controller now manages the conversation state flow.
def hybrid_retriever(user_query, llm, graph, assets, chains, state):
    """쿼리 라우팅 및 대화 상태를 관리하는 메인 컨트롤러"""
    print(f"\n--- User Query: '{user_query}' ---")
    
    if state.get('waiting_for_preference'):
        print("Route: Preference Extraction")
        extracted_prefs_str = chains['preference_extractor'].invoke({"user_input": user_query})['text']
        try:
            state['preferences'] = json.loads(extracted_prefs_str)
            state['waiting_for_preference'] = False
            return personalized_recommendation(state, llm, assets, chains)
        except json.JSONDecodeError:
            return "I had trouble understanding. Could you rephrase your preferences?"
    
    route = chains['router'].invoke({"user_query": user_query})['text'].strip()
    
    if route == "fact_based_search":
        return fact_based_search(user_query, graph, assets, chains)
    
    elif route == "personalized_recommendation":
        print("Route: Personalized Recommendation")
        extracted_prefs_str = chains['preference_extractor'].invoke({"user_input": user_query})['text']
        state['preferences'] = json.loads(extracted_prefs_str)
        return personalized_recommendation(state, llm, assets, chains)
    
    else: # chit_chat or unknown
        return "I'm a movie recommendation chatbot. What kind of movie are you looking for? 😊"