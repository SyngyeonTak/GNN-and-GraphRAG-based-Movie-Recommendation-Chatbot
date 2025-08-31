from utils import (
    find_best_match, 
    clean_cypher_query, 
    find_movies_with_faiss,
    format_candidates_for_prompt,
    find_best_name
)
import json


def personalized_recommendation(user_query, state, graph, assets, chains, global_graph_nx):
    """
    Handle personalized recommendation requests through the Faiss pipeline.
    """
    print("\n[Executing Personalized Recommendation with Faiss]")
    preferences = state.get('preferences', {})

    if not any(preferences.values()):
        state['waiting_for_preference'] = True
        personalized_guide_chain = chains['personalized_guide']
        return personalized_guide_chain.run(
            user_query=user_query,
            preferences=preferences,
            schema=graph.schema
        )
    else:
        print(f"Using preferences to find recommendations: {preferences}")
        rerank_dict = find_movies_with_faiss(preferences, assets, graph, chains, global_graph_nx)
        if not rerank_dict:
            return "Sorry, I couldn't find good examples for your taste. Please try another preference."
        candidates_str = format_candidates_for_prompt(rerank_dict)
        preferences_str = json.dumps(preferences, indent=2)

        personalized_response_chain = chains['personalized_responder']

        inputs = {
            "user_query": user_query,
            "preferences_str": preferences_str,
            "candidates_str": candidates_str
        }
        return personalized_response_chain.invoke(inputs)['text']


def fact_based_search(query: str, graph, assets: dict, chains: dict):
    """
    Handles fact-based queries by:
    1. Extracting structured entities
    2. Performing fuzzy matching
    3. Replacing entity names in the query
    4. Generating Cypher query
    5. Executing Cypher query and formatting the final answer
    """
    import json
    print("\n[Executing Consolidated Fact-Based Search]")
    
    # 1. Extract structured entities (JSON format)
    extractor_chain = chains['entity_extractor']
    entity_json_str = extractor_chain.invoke({"user_input": query})['text']
    try:
        extracted_entities = json.loads(entity_json_str)
    except json.JSONDecodeError:
        return "Sorry, I had trouble understanding your question. Could you please rephrase it?"

    if not any(extracted_entities.values()):
        return "I'm sorry, but I couldn't find any movie or person information in your question."
    print(f"Extracted Entities: {extracted_entities}")

    # 2. Perform Fuzzy Matching
    corrected_entities = {}
    for entity_type, names in extracted_entities.items():
        if not names:
            continue
        entity_asset = assets.get(entity_type)
        if not entity_asset:
            continue
        corrected_names = [find_best_name(name, entity_asset) for name in names]
        valid_names = [name for name in corrected_names if name]
        if valid_names:
            corrected_entities[entity_type] = valid_names

    if not corrected_entities:
        return "I'm sorry, but I couldn't find a match for that in my database."
    print(f"Corrected Entities: {corrected_entities}")

    # 3. Replace original query terms with corrected entity names
    corrected_query = query
    for entity_type, names in corrected_entities.items():
        for original_name, corrected_name in zip(extracted_entities[entity_type], names):
            if original_name.lower() != corrected_name.lower():
                corrected_query = corrected_query.replace(original_name, corrected_name)

    print(f"Corrected Query: {corrected_query}")

    # 4. Generate Cypher query
    cypher_input = {
        "schema": graph.schema,
        "question": (
            f"User's original question: {query}\n"
            f"Corrected question with entity matches: {corrected_query}\n"
            f"Entities recognized: {corrected_entities}\n"
            "Generate a single Cypher query to answer the corrected question."
        )
    }
    generated_cypher = chains['cypher_gen'].invoke(cypher_input)['text']
    cleaned_cypher = clean_cypher_query(generated_cypher)
    print(f"Generated Cypher: {cleaned_cypher}")

    try:
        cypher_result = graph.query(cleaned_cypher)
    except Exception as e:
        print(f"Cypher execution failed: {e}")
        return "Sorry, I ran into an error trying to find that information."

    # 5. Format the final response
    if not cypher_result:
        return "I couldn't find any information for your query. Do you have another question?"
    
    fact_based_response_chain = chains['fact_based_responder']
    final_answer = fact_based_response_chain.invoke({
        "user_query": query,
        "cypher_result": json.dumps(cypher_result)
    })['text']
    
    return final_answer


def chit_chat(user_query, chains):
    """
    Handles simple, conversational, or off-topic queries.
    """
    chit_chat_chain = chains['chit_chatter']
    response = chit_chat_chain.invoke({"user_input": user_query})['text']
    return response


def hybrid_retriever(user_query, graph, assets, chains, state, global_graph_nx):
    """
    Main controller for query routing and conversation state management.
    """
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
    
    else:  # chit_chat or unknown
        return chit_chat(user_query, chains)
