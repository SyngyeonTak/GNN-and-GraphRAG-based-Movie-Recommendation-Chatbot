from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.graphs  import Neo4jGraph  # Use the new, recommended package
import faiss  
import json
import numpy as np

# ==================================================================
# Function to load ALL recommendation assets
# ==================================================================
def load_recommendation_assets():
    """
    Loads Faiss index and creates both index-to-title and title-to-index maps.
    """
    print("Loading recommendation assets...")
    try:
        faiss_index = faiss.read_index("movie_embeddings.faiss")
        
        with open("faiss_to_movie_title.json", 'r', encoding='utf-8') as f:
            idx_to_title = json.load(f)
        
        # Create the reverse mapping from title to Faiss index
        # This is crucial for looking up embeddings by title
        title_to_idx = {title: int(idx) for idx, title in idx_to_title.items()}
        
        print(f"✅ Faiss index and mappings loaded successfully.")
        return faiss_index, idx_to_title, title_to_idx
    except Exception as e:
        print(f"❌ Failed to load recommendation assets: {e}")
        return None, None, None
    
# ==================================================================
# NEW: LLM Chain to find representative movies
# ==================================================================
def get_movie_suggester_chain(llm):
    """
    Creates a chain that suggests representative movie titles for a given preference.
    """
    suggester_template = """
    You are an expert movie recommender. Based on the user's preference for a genre, actor, or theme, please suggest 3 to 5 famous and representative movie titles.
    Your answer MUST be ONLY a comma-separated list of movie titles. Do not include any other text, explanation, or numbering.

    User's Stated Preference: "{preference}"
    
    Comma-Separated Movie Titles:
    """
    suggester_prompt = PromptTemplate(template=suggester_template, input_variables=["preference"])
    return LLMChain(llm=llm, prompt=suggester_prompt)

# ==================================================================
# Personalized Recommendation using Faiss (Corrected and Completed Logic)
# ==================================================================
def find_movies_with_faiss(preferences, llm, faiss_index, idx_to_title, title_to_idx, top_k=5):
    """
    Finds top-k similar movies by first using an LLM to find representative movies
    for the user's preferences, and then using their embeddings for Faiss search.
    """
    
    # 1. Use an LLM to suggest representative movies based on the user's preferences.
    # We combine preferences into a single query string for the LLM.
    preference_str = ", ".join(preferences)
    movie_suggester_chain = get_movie_suggester_chain(llm)
    suggested_movies_str = movie_suggester_chain.invoke({"preference": preference_str})['text']
    suggested_movie_titles = [title.strip() for title in suggested_movies_str.split(',')]
    print(f"LLM suggested representative movies: {suggested_movie_titles}")

    # 2. Look up the embeddings for the LLM-suggested movies.
    representative_vectors = []
    for title in suggested_movie_titles: 
        idx = title_to_idx.get(title)
        if idx is not None: 
            # tsy 룰이 너무 naive, hard 함 llm이 추천하는 영화 제목과 내 vector DB에 있는 영화의 entity linking이 필요함
            # tsy from thefuzz import process 같은거 쓰면 된다고 함
            # faiss.reconstruct(id) retrieves the vector for a given index
            representative_vectors.append(faiss_index.reconstruct(idx))
            print(f"Found embedding for representative movie: '{title}'")
        else:
            print(f"Warning: Suggested movie '{title}' not found in our movie list.")

    if not representative_vectors:
        return [{"title": "Sorry, I couldn't find good examples for your taste. Please try another preference."}]

    # 3. Average the vectors to create the 'session_taste_vector'.
    session_taste_vector = np.mean(representative_vectors, axis=0)

    # 4. Perform Faiss search with the new session vector.
    search_vector = session_taste_vector.astype('float32').reshape(1, -1)
    distances, indices = faiss_index.search(search_vector, k=top_k)
    
    # 5. Convert results to movie titles.
    recommendations = []
    for i in indices[0]:
        movie_title = idx_to_title.get(str(i), f"Unknown Movie (ID: {i})")
        recommendations.append({"title": movie_title})
        
    return recommendations

# This is how you would call it from your main chatbot logic
def personalized_recommendation(conversation_state, llm, rec_assets):
    """
    Handles personalized recommendation requests using the LLM-Faiss pipeline.
    """
    print("\n[Executing LLM-driven Personalized Recommendation with Faiss]")
    
    faiss_index, idx_to_title, title_to_idx = rec_assets
    preferences = conversation_state.get('preferences', [])

    if not preferences:
        return "Of course! To give you a good recommendation, could you tell me about a genre, actor, or a movie you've enjoyed recently?"
    else:
        print(f"🧠 Using detected preferences to find recommendations: {preferences}")
        return find_movies_with_faiss(
            preferences, llm, faiss_index, idx_to_title, title_to_idx
        )

def clean_cypher_query(query: str) -> str:
    """
    Cleans the Cypher query generated by the LLM by removing Markdown code blocks.
    """
    if "```cypher" in query:
        # Extracts the query from a ```cypher code block
        query = query.split("```cypher", 1)[1]
    if "```" in query:
        # Removes the closing backticks
        query = query.split("```", 1)[0]
        
    # Removes leading/trailing whitespace and newlines
    return query.strip()

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
# 2. Hybrid Retriever Function Definitions
# ==================================================================

# 2-A. Query Router
def get_query_router_chain(llm):
    """
    Creates a chain to classify the user's query into one of three categories.
    """
    router_prompt_template = """
    You are a highly intelligent query classifier for a movie recommendation chatbot.
    Your task is to analyze the user's query and decide which tool to use.
    Pay close attention to whether the user is asking for a general recommendation or a recommendation with specific constraints (like actors, directors, genres).

    Here are the available tools:

    1. **fact_based_search**: Use this for queries that ask for specific facts OR recommendations with **specific constraints**.
       These queries usually contain named entities like movie titles, actors, directors, or specific properties like genres, release years.
       - Example: "Who directed the movie Inception?"
       - Example: "Show me action movies starring Tom Cruise."
       - Example: "Recommend a comedy movie from the 90s."

    2. **personalized_recommendation**: Use this for queries that are **open-ended**, ask for general suggestions, or are based on the user's personal taste without specific constraints.
       - Example: "What should I watch tonight?"
       - Example: "Recommend a movie for me."

    3. **chit_chat**: Use this for queries that are simple greetings, conversational fillers, or off-topic.
       - Example: "Hello"
       - Example: "What's the weather like today?"

    User Query: "{user_query}"

    Based on the query, which tool should be used? Respond with only the name of the tool.
    Tool:
    """
    router_prompt = PromptTemplate(template=router_prompt_template, input_variables=["user_query"])
    return LLMChain(llm=llm, prompt=router_prompt, verbose=False)

# 2-B. Fact-Based Search
def get_cypher_generation_chain(llm):
    """
    Creates a chain that generates a Cypher query based on the user's question and the graph schema.
    """
    cypher_generation_template = """
    Task: Generate a read-only Cypher query to retrieve information from a Neo4j database based on the user's question.
    Instructions:
    1. Use the provided schema to understand the graph structure.
    2. Respond with ONLY the Cypher query. Do not include any explanations.
    Schema: {schema}
    User Question: {question}
    Cypher Query:
    """
    cypher_prompt = PromptTemplate(template=cypher_generation_template, input_variables=["schema", "question"])
    return LLMChain(llm=llm, prompt=cypher_prompt, verbose=False)

def fact_based_search(user_query: str, graph, cypher_chain):
    """
    Converts a user's question into a Cypher query and executes it against the Neo4j database.
    """
    print("\n[Executing Fact-Based Search]")
    
    # 1. Generate the Cypher query using the LLM
    generated_text = cypher_chain.invoke({
        "schema": graph.schema, 
        "question": user_query
    })['text']

    # 2. Clean the generated text to get a pure Cypher query
    cleaned_query = clean_cypher_query(generated_text)
    
    print(f"🧠 Generated Cypher Query (Cleaned): {cleaned_query}")

    # 3. Execute the cleaned query against the database
    try:
        result = graph.query(cleaned_query)
        print(f"✅ Query executed successfully. Result: {result}")
        return result
    except Exception as e:
        print(f"❌ Query execution failed: {e}")
        return []

# 2-D. Chit-Chat
def chit_chat(user_query):
    """
    Handles simple, conversational, or off-topic queries.
    """
    print("\n[Handling Chit-Chat]")
    response = "I'm a movie recommendation chatbot. What kind of movie are you looking for? 😊"
    print(f"✅ Response: {response}")
    return response

# 2-E. Hybrid Retriever Main Controller
def hybrid_retriever(user_query, llm, graph, router_chain, cypher_chain, conversation_state, rec_assets):
    """
    The main controller that routes the query and manages conversation state.
    """
    print(f"\n--- User Query: '{user_query}' ---")
    
    route = router_chain.invoke({"user_query": user_query})['text'].strip()
    
    if route == "fact_based_search":
        print("Route: Fact-Based Search")
        return fact_based_search(user_query, graph, cypher_chain)
    
    elif route == "personalized_recommendation":
        print("Route: Personalized Recommendation")
        return personalized_recommendation(conversation_state, llm, rec_assets)
    elif route == "chit_chat":
        print("Route: Chit-Chat")
        return chit_chat(user_query)
    
    else:
        response = "I'm sorry, I couldn't understand your request."
        print(f"Route: Unknown. Response: {response}")
        return response


# ==================================================================
# 3. Main Execution
# ==================================================================
def main():
    """
    Sets up the entire program and runs test scenarios.
    """
    print("🚀 Starting Movie Recommendation Chatbot...")
    print("="*60)
    
    # 1. Setup environment: Load keys, connect to DB, initialize LLM
    llm, graph = setup_environment()
    
    # 2. Load recommendation assets: Faiss index and mappings
    rec_assets = load_recommendation_assets()
    
    # Exit if any setup fails
    if not all([llm, graph, rec_assets]):
        print("\n❌ Exiting program due to setup or asset loading failure.")
        return
        
    # 3. Initialize all necessary LangChain chains
    query_router_chain = get_query_router_chain(llm)
    cypher_generation_chain = get_cypher_generation_chain(llm)
    
    # 4. Initialize conversation state
    # This dictionary will hold the user's preferences throughout the conversation
    conversation_state = {}
    
    print("\n" + "="*60)
    print("🤖 Chatbot is ready. Let's start a conversation!")
    print("="*60)
    
    # --- Conversational Scenario ---

    # Turn 1: User asks for a general recommendation (Cold User)
    user_input_1 = "Recommend a movie for me."
    response_1 = hybrid_retriever(user_input_1, llm, graph, query_router_chain, cypher_generation_chain, conversation_state, rec_assets)
    print(f"\n👤 You: {user_input_1}")
    print(f"🤖 Chatbot: {response_1}")
    
    print("\n" + "-"*60)

    # Turn 2: User provides their preferences based on the chatbot's question
    # In a real app, this would come from the user. We'll simulate it here.
    # A real implementation would have another LLM chain to extract these entities.
    print("(Simulating user's response: 'I feel like watching a thrilling action movie.')")
    conversation_state['preferences'] = ['thrilling action']
    
    user_input_2 = "Based on that, what do you suggest?"
    response_2 = hybrid_retriever(user_input_2, llm, graph, query_router_chain, cypher_generation_chain, conversation_state, rec_assets)
    print(f"\n👤 You: {user_input_2}")
    print(f"🤖 Chatbot: {response_2}")
    
    print("\n" + "-"*60)

    # Turn 3: User asks a fact-based question
    user_input_3 = "Who directed the movie Inception?"
    response_3 = hybrid_retriever(user_input_3, llm, graph, query_router_chain, cypher_generation_chain, conversation_state, rec_assets)
    print(f"\n👤 You: {user_input_3}")
    print(f"🤖 Chatbot: {response_3}")

    print("\n" + "="*60)
    print("✅ All test scenarios have been executed.")

if __name__ == "__main__":
    main()