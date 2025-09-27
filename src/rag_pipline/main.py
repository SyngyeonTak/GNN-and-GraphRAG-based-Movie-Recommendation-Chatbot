import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
import pickle

# Import functions from separated files
from chains import (
    get_query_router_chain,
    get_cypher_generation_chain,
    get_subgraph_cypher_chain,
    get_preference_extractor_chain,
    get_entity_extractor_chain,
    get_movie_suggester_chain,
    get_genre_mapper_chain,    
    get_personalized_response_chain,
    get_fact_based_response_chain,
    get_chit_chat_chain,
    get_personalized_guide_chain,
)

from utils import load_recommendation_assets
from graph_utils import create_global_nx_graph
from retriever import hybrid_retriever


def setup_environment():
    """
    Loads API keys and Neo4j credentials from .env, 
    initializes Neo4j graph connection and OpenAI LLM.
    """
    load_dotenv()
    
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    try:
        graph = Neo4jGraph(url=uri, username=user, password=password)
        print("Neo4j database connection successful.")
        print("Graph Schema:\n", graph.schema)
    except Exception as e:
        print(f"Neo4j connection failed: {e}")
        print("Please check your NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables.")
        graph = None

    # Initialize LLM (API key is read from environment)
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-4o-mini")
    print("OpenAI LLM model initialized successfully.")
    
    return llm, graph


def main():
    """
    Sets up the entire program and runs automated test cases 
    using the hybrid retriever pipeline.
    """
    print("Starting Movie Recommendation Chatbot...")
    print("=" * 60)
    
    llm, graph = setup_environment()

    with open('./dataset/graph_snapshot.pkl', 'rb') as f:
        snapshot_data = pickle.load(f)

    # Build global graph in memory
    global_graph_nx = create_global_nx_graph(snapshot_data)
    rec_assets = load_recommendation_assets()
    
    if not all([llm, graph, rec_assets]):
        print("\nExiting program due to setup or asset loading failure.")
        return
    
    chains = {
        'router': get_query_router_chain(llm),
        'cypher_gen': get_cypher_generation_chain(llm),
        'subgraph_gen': get_subgraph_cypher_chain(llm),
        'preference_extractor': get_preference_extractor_chain(llm),
        'entity_extractor': get_entity_extractor_chain(llm),
        'movie_suggester': get_movie_suggester_chain(llm),
        'genre_mapper': get_genre_mapper_chain(llm),
        'personalized_responder': get_personalized_response_chain(llm),
        'personalized_guide': get_personalized_guide_chain(llm),
        'fact_based_responder': get_fact_based_response_chain(llm),
        'chit_chatter': get_chit_chat_chain(llm)
    }
    
    conversation_state = {}
    
    print("\n" + "=" * 60)
    print("Chatbot is ready. Starting test conversations...")
    print("=" * 60)
    
    test_cases = [
        #{"TestCase_ID": "F-01-1", "User_Input": "Can you recommend me a movie?", "Test_Objective": "Basic recommendation"},
        #{"TestCase_ID": "F-01-2", "User_Input": "I like action movies.", "Test_Objective": "Extract user preference"},
        #{"TestCase_ID": "F-01-3", "User_Input": "Recommend me a movie directed by Christopher Nolan.", "Test_Objective": "Director-based recommendation"},
        {"TestCase_ID": "F-01-4", "User_Input": "Recommend me movie like The Dark Knight.", "Test_Objective": "Movie information retrieval"},
        {"TestCase_ID": "F-01-5", "User_Input": "Thanks!", "Test_Objective": "Simple chit-chat response"}
    ]

    # Iterate through each test case
    for case in test_cases:
        test_id = case['TestCase_ID']
        user_input = str(case['User_Input'])

        # Reset conversation state for new scenarios
        if not any(cont in test_id for cont in ['-2', '-3', '-4', '-5']):
            conversation_state = {}
            print("\n" + "=" * 60)
            print("New Scenario: Conversation state has been reset.")

        print(f"--- Running Test: {test_id} ---")
        print(f"Objective: {case['Test_Objective']}")

        # Call hybrid retriever
        response = hybrid_retriever(
            user_query=user_input,
            graph=graph,
            chains=chains,
            state=conversation_state,
            assets=rec_assets,
            global_graph_nx=global_graph_nx
        )

        # Print results
        print(f"\nUser: {user_input}")
        print(f"Chatbot: {response}")
        print(f"Current State: {conversation_state}")
        print("-" * 50)

    print("\n" + "=" * 60)
    print("All test scenarios have been executed.")

if __name__ == "__main__":
    main()
