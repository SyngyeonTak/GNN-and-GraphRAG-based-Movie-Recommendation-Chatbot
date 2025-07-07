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

# ==================================================================
# NEW: LLM Chain to extract preferences from user input
# ==================================================================
def get_preference_extractor_chain(llm):
    """
    Creates a chain that extracts structured preferences (actors, genres, keywords)
    from a user's free-form text query.
    """
    extractor_template = """
    You are an expert at understanding user preferences for movies.
    Your task is to extract key entities from the user's statement.
    Extract actors, directors, genres, and any other descriptive keywords.
    Respond with ONLY a JSON object containing the extracted information.
    The keys of the JSON should be "actors", "directors", "genres", and "keywords".
    If no information is found for a category, provide an empty list [].

    Example 1:
    User's Statement: "I want to watch a thrilling action movie."
    JSON Output:
    {
        "actors": [],
        "directors": [],
        "genres": ["action"],
        "keywords": ["thrilling"]
    }

    Example 2:
    User's Statement: "Show me something with Tom Hanks, maybe a drama."
    JSON Output:
    {
        "actors": ["Tom Hanks"],
        "directors": [],
        "genres": ["drama"],
        "keywords": []
    }

    Example 3:
    User's Statement: "I don't know, just something fun."
    JSON Output:
    {
        "actors": [],
        "directors": [],
        "genres": [],
        "keywords": ["fun"]
    }

    User's Statement: "{user_input}"
    JSON Output:
    """
    extractor_prompt = PromptTemplate(template=extractor_template, input_variables=["user_input"])
    return LLMChain(llm=llm, prompt=extractor_prompt)

# MODIFIED: This function now just asks a question or triggers the Faiss search.
# The state management logic is moved to the main retriever.
def personalized_recommendation(conversation_state, llm, rec_assets):
    """
    Handles personalized recommendation requests using the LLM-Faiss pipeline.
    """
    print("\n[Executing LLM-driven Personalized Recommendation with Faiss]")
    
    faiss_index, idx_to_title, title_to_idx = rec_assets
    preferences = conversation_state.get('preferences', [])

    if not preferences:
        # Set a flag to indicate we are waiting for the user's preference
        conversation_state['waiting_for_preference'] = True
        return "Of course! To give you a good recommendation, could you tell me about a genre, actor, or a movie you've enjoyed recently?"
    else:
        print(f"🧠 Using detected preferences to find recommendations: {preferences}")
        # Flatten all preference values into a single list for the next step
        flat_preferences = [item for sublist in preferences.values() for item in sublist]
        return find_movies_with_faiss(
            flat_preferences, llm, faiss_index, idx_to_title, title_to_idx
        )

# ... (기존의 find_movies_with_faiss, clean_cypher_query 등은 그대로 둡니다) ...

# MODIFIED: The main controller now manages the conversation state flow.
def hybrid_retriever(user_query, llm, graph, router_chain, cypher_chain, preference_extractor_chain, conversation_state, rec_assets):
    """
    The main controller that routes the query and manages conversation state,
    including preference extraction.
    """
    print(f"\n--- User Query: '{user_query}' ---")
    
    # Check if we are waiting for the user to state their preferences
    if conversation_state.get('waiting_for_preference'):
        print("Route: Preference Extraction")
        # The user's input is expected to be their preference
        extracted_prefs_str = preference_extractor_chain.invoke({"user_input": user_query})['text']
        
        try:
            # Parse the JSON output from the LLM
            extracted_prefs = json.loads(extracted_prefs_str)
            print(f"🧠 Extracted Preferences: {extracted_prefs}")
            
            # Update the conversation state with the new preferences
            # If preferences already exist, you might want to merge them, but here we'll overwrite for simplicity
            conversation_state['preferences'] = extracted_prefs
            
            # We are no longer waiting for a preference
            conversation_state['waiting_for_preference'] = False
            
            # Now, call the personalized recommendation function again with the updated state
            return personalized_recommendation(conversation_state, llm, rec_assets)
            
        except json.JSONDecodeError:
            conversation_state['waiting_for_preference'] = False
            return "I had a little trouble understanding that. Could you please try rephrasing your preferences?"

    # If not waiting for a preference, use the standard router
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
        
    # Initialize all necessary LangChain chains
    query_router_chain = get_query_router_chain(llm)
    cypher_generation_chain = get_cypher_generation_chain(llm)
    # NEW: Initialize the preference extractor chain
    preference_extractor_chain = get_preference_extractor_chain(llm)
    
    # Initialize conversation state
    conversation_state = {}
    
    print("\n" + "="*60)
    print("🤖 Chatbot is ready. Let's start a conversation!")
    print("="*60)
    
    # --- MODIFIED Conversational Scenario ---

    # Turn 1: User asks for a general recommendation
    user_input_1 = "Recommend a movie for me."
    # MODIFIED: Pass the new chain to the retriever
    response_1 = hybrid_retriever(user_input_1, llm, graph, query_router_chain, cypher_generation_chain, preference_extractor_chain, conversation_state, rec_assets)
    print(f"\n👤 You: {user_input_1}")
    print(f"🤖 Chatbot: {response_1}")
    # At this point, conversation_state['waiting_for_preference'] is True
    print(f"Current State: {conversation_state}")
    
    print("\n" + "-"*60)

    # Turn 2: User provides their preferences based on the chatbot's question
    user_input_2 = "I feel like watching a thrilling action movie starring Tom Cruise."
    # The retriever will now use the preference_extractor_chain because of the state flag
    response_2 = hybrid_retriever(user_input_2, llm, graph, query_router_chain, cypher_generation_chain, preference_extractor_chain, conversation_state, rec_assets)
    print(f"\n👤 You: {user_input_2}")
    print(f"🤖 Chatbot: {response_2}")
    # At this point, the state is updated with extracted preferences
    print(f"Current State: {conversation_state}")

    print("\n" + "-"*60)

    # Turn 3: User asks a fact-based question (the state is maintained but not used here)
    user_input_3 = "Who directed the movie Inception?"
    response_3 = hybrid_retriever(user_input_3, llm, graph, query_router_chain, cypher_generation_chain, preference_extractor_chain, conversation_state, rec_assets)
    print(f"\n👤 You: {user_input_3}")
    print(f"🤖 Chatbot: {response_3}")

    print("\n" + "="*60)
    print("✅ All test scenarios have been executed.")

if __name__ == "__main__":
    main()
