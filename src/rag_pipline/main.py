import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph

# 분리된 파일에서 함수 임포트

from chains import (
    get_query_router_chain,
    get_cypher_generation_chain,
    get_preference_extractor_chain,
    get_entity_extractor_chain,
    get_movie_suggester_chain,
    get_genre_mapper_chain
)

from retriever import *

# ==================================================================
# 1. Environment Setup
# ==================================================================
def setup_environment():
    """
    Loads API keys and Neo4j credentials from .env, then returns LLM and Graph instances.
    """
    load_dotenv()
    
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    try:
        graph = Neo4jGraph(url=uri, username=user, password=password)
        print("✅ Neo4j database connection successful!")
        print("Graph Schema:\n", graph.schema)
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("Please check your NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables.")
        graph = None

    # The API key is automatically read from the OPENAI_API_KEY environment variable
    llm = ChatOpenAI(openai_api_key = openai_api_key, temperature=0, model_name="gpt-4o-mini")
    print("✅ OpenAI LLM model initialized successfully!")
    
    return llm, graph

# ==================================================================
# 3. Main Execution
# ==================================================================
def main():
    """
    Sets up the entire program and runs test scenarios.
    """
    print("🚀 Starting Movie Recommendation Chatbot...")
    print("="*60)
    
    llm, graph = setup_environment()
    rec_assets = load_recommendation_assets()
    
    if not all([llm, graph, rec_assets]):
        print("\n❌ Exiting program due to setup or asset loading failure.")
        return
        
    if not all([llm, graph, rec_assets]):
        print("\n❌ Exiting due to setup failure.")
        return
        
    chains = {
        'router': get_query_router_chain(llm),
        'cypher_gen': get_cypher_generation_chain(llm),
        'preference_extractor': get_preference_extractor_chain(llm),
        'entity_extractor': get_entity_extractor_chain(llm),
        'movie_suggester': get_movie_suggester_chain(llm),
        'genre_mapper': get_genre_mapper_chain(llm),
    }
    
    # Initialize conversation state
    conversation_state = {}
    
    print("\n" + "="*60)
    print("🤖 Chatbot is ready. Let's start a conversation!")
    print("="*60)
    
    # 3. CSV 파일에서 테스트 케이스 로드
    try:
        test_cases_df = pd.read_csv("./dataset/TC/hybrid_test_cases.csv")
        print(f"✅ Test cases loaded successfully from 'test_cases.csv'. Found {len(test_cases_df)} tests.")
    except FileNotFoundError:
        print("❌ CRITICAL ERROR: 'test_cases.csv' not found. Please create the file.")
        return
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load or read 'test_cases.csv': {e}")
        return

    # 4. 각 테스트 케이스를 순회하며 실행
    conversation_state = {}
    for index, row in test_cases_df.iterrows():
        test_id = row['TestCase_ID']

        #if test_id not in ['R-06', 'R-09', 'P-01-2', 'P-02-2', 'P-03-2']:
        if test_id not in ['P-01-2']:
            continue

        user_input = str(row['User_Input']) # Ensure input is a string

        # 새로운 시나리오가 시작되면 대화 상태 초기화 (멀티턴 대화는 유지)
        if not any(cont in test_id for cont in ['-2', '-3', '-4', '-5']):
            conversation_state = {}
            print("\n" + "="*60)
            print("🔄 New Scenario: Conversation state has been reset.")

        print(f"--- Running Test: {test_id} ---")
        print(f"Objective: {row['Test_Objective']}")

        # 하이브리드 리트리버 호출
        response = hybrid_retriever(
            user_query = user_input,
            llm = llm,
            graph = graph,
            chains = chains,
            state = conversation_state,
            assets = rec_assets
        )

        # 결과 출력
        print(f"\n👤 You: {user_input}")
        print(f"🤖 Chatbot: {response}")
        print(f"Current State: {conversation_state}")
        print("-" * 50)

    print("\n" + "="*60)
    print("✅ All test scenarios have been executed.")

if __name__ == "__main__":
    main()
