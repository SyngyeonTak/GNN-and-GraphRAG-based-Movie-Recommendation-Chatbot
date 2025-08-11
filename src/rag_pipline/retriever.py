from utils import (
            find_best_match, 
            clean_cypher_query, 
            find_movies_with_faiss,
            format_candidates_for_prompt
        )
import json


# MODIFIED: This function now just asks a question or triggers the Faiss search.
# The state management logic is moved to the main retriever.

def personalized_recommendation(user_query, state, graph, assets, chains, global_graph_nx):
    """개인화 추천 요청을 Faiss 파이프라인으로 처리"""
    print("\n[Executing Personalized Recommendation with Faiss]")
    preferences = state.get('preferences', {})

    if not any(preferences.values()):
        state['waiting_for_preference'] = True
        return "Of course! To give you a good recommendation, could you tell me about a genre, actor, or a movie you've enjoyed recently?"
    else:
        print(f"🧠 Using preferences to find recommendations: {preferences}")
        rerank_dict = find_movies_with_faiss(preferences, assets, graph, chains, global_graph_nx)
        candidates_str = format_candidates_for_prompt(rerank_dict)
        preferences_str = json.dumps(preferences, indent=2)

        personalized_response_chain = chains['personalized_responder']

        inputs = {
            "user_query": user_query,
            "preferences_str": preferences_str,
            "candidates_str": candidates_str
        }
        return personalized_response_chain.invoke(inputs)['text']

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
def hybrid_retriever(user_query, graph, assets, chains, state, global_graph_nx):
    """쿼리 라우팅 및 대화 상태를 관리하는 메인 컨트롤러"""
    print(f"\n--- User Query: '{user_query}' ---")
    
    if state.get('waiting_for_preference'):
        print("Route: Preference Extraction")
        extracted_prefs_str = chains['preference_extractor'].invoke({"user_input": user_query})['text']
        try:
            state['preferences'] = json.loads(extracted_prefs_str)
            state['waiting_for_preference'] = False
            return personalized_recommendation(user_query, state, graph, assets, chains, global_graph_nx)
        except json.JSONDecodeError:
            return "I had trouble understanding. Could you rephrase your preferences?"
    
    route = chains['router'].invoke({"user_query": user_query})['text'].strip().strip("'\"")
    
    if route == "fact_based_search":
        return fact_based_search(user_query, graph, assets, chains)
    
    elif route == "personalized_recommendation":
        print("Route: Personalized Recommendation")
        extracted_prefs_str = chains['preference_extractor'].invoke({"user_input": user_query})['text']
        state['preferences'] = json.loads(extracted_prefs_str)
        return personalized_recommendation(user_query, state, graph, assets, chains, global_graph_nx)
    
    else: # chit_chat or unknown
        return "I'm a movie recommendation chatbot. What kind of movie are you looking for? 😊"