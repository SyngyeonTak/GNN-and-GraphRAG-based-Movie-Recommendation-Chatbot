import os
import gradio as gr
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
import pickle
import networkx as nx

# Import necessary functions
from chains import (
    get_query_router_chain, get_cypher_generation_chain, get_subgraph_cypher_chain,
    get_preference_extractor_chain, get_entity_extractor_chain, get_movie_suggester_chain,
    get_genre_mapper_chain, get_personalized_response_chain, get_fact_based_response_chain,
    get_chit_chat_chain, get_personalized_guide_chain,
)
from utils import load_recommendation_assets
from graph_utils import create_global_nx_graph
from retriever import hybrid_retriever

print("Starting Movie Recommendation Chatbot...")

# Load environment variables and initialize objects
load_dotenv()
uri = os.environ.get("NEO4J_URI")
user = os.environ.get("NEO4J_USER")
password = os.environ.get("NEO4J_PASSWORD")
openai_api_key = os.environ.get("OPENAI_API_KEY")

try:
    graph = Neo4jGraph(url=uri, username=user, password=password)
    print("Neo4j database connection successful.")
except Exception as e:
    print(f"Neo4j connection failed: {e}")
    graph = None

llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-4o-mini")
print("OpenAI LLM model initialized successfully.")

# Load dataset and assets
with open('./dataset/graph_snapshot.pkl', 'rb') as f:
    snapshot_data = pickle.load(f)
global_graph_nx = create_global_nx_graph(snapshot_data)
rec_assets = load_recommendation_assets()
print("Recommendation assets loaded successfully.")

# Create chains
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
print("All chains initialized successfully.")

conversation_state = {}
print("="*60)
print("Chatbot is ready to launch with Gradio.")
print("="*60)

def chatbot_response(user_message, chat_history):
    """
    Receive a user message and generate chatbot response.
    This is the core logic function for Gradio UI.
    """
    global conversation_state

    response = hybrid_retriever(
        user_query=user_message,
        graph=graph,
        chains=chains,
        state=conversation_state,
        assets=rec_assets,
        global_graph_nx=global_graph_nx
    )
    return response

with gr.Blocks(theme="soft") as demo:
    gr.Markdown(
        """
        # Movie Recommendation Chatbot
        Get personalized movie recommendations powered by GNN and LLM.
        """
    )
    
    chatbot = gr.Chatbot(
        value=[{"role": "assistant", "content": "Hello! I am your movie recommendation chatbot. Which movie would you like me to find for you?"}],
        label="Movie Recommendation Chatbot",
        type='messages' 
    )

    msg = gr.Textbox(label="Enter your message...", placeholder="Example: 'Find me a heartwarming movie starring Tom Hanks'")
    clear = gr.Button("Start New Conversation")

    def respond(message, chat_history):
        bot_message = chatbot_response(message, chat_history)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    def clear_chat():
        global conversation_state
        conversation_state.clear()
        print("Conversation state has been reset.")
        return [], [{"role": "assistant", "content": "Hello! How can I help you again? Which movie would you like me to find for you?"}]

    clear.click(clear_chat, None, [chatbot, msg], queue=False)

if __name__ == "__main__":
    demo.launch(debug=True)
