from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain

def get_genre_mapper_chain(llm):
    """
    Maps a user-stated genre to the closest genre available in the database.
    """
    mapper_template = """
    You are a genre mapping expert. A user mentioned the genre '{user_genre}'.
    From the following list of available genres, which one is the closest semantic match?
    Available Genres: {available_genres}
    Respond with ONLY the single best matching genre name from the list. If no good match exists, respond with 'None'.
    """
    mapper_prompt = PromptTemplate(template=mapper_template, input_variables=["user_genre", "available_genres"])
    return LLMChain(llm=llm, prompt=mapper_prompt)

def get_movie_suggester_chain(llm):
    """
    Suggests representative movie titles based on a given preference.
    """
    suggester_template = """
    You are an expert movie recommender. Based on the user's preference for a genre, actor, or theme, please suggest 3 to 5 famous and representative movie titles.
    Your answer MUST be ONLY a comma-separated list of movie titles.

    User's Stated Preference: "{preference}"
    
    Comma-Separated Movie Titles:
    """
    suggester_prompt = PromptTemplate(template=suggester_template, input_variables=["preference"])
    return LLMChain(llm=llm, prompt=suggester_prompt)

def get_preference_extractor_chain(llm):
    """
    Extracts structured user preferences (actors, directors, genres, movies) as JSON.
    """
    extractor_template = """
    You are an expert at understanding user preferences for movies.
    Your task is to extract key entities from the user's statement.

    First, if there are any spelling or name mistakes in the statement (like 'Tam Honks' instead of 'Tom Hanks'),
    correct them silently and proceed as if the correct version was provided.

    Then, extract actors, directors, genres and movies.

    Respond with ONLY a JSON object containing the extracted information.
    If no information is found for a category, provide an empty list [].
    The keys of the JSON should be "actors", "directors", "genres" and "movies".

    Example 1:
    User's Statement: "I want to watch a thrilling action movie."
    JSON Output =
    {{
        "actors": [],
        "directors": [],
        "genres": ["action"],
        "movies": []
    }}

    Example 2:
    User's Statement: "Show me something with Tom Hanks, maybe a drama."
    JSON Output = 
    {{
        "actors": ["Tom Hanks"],
        "directors": [],
        "genres": ["drama"],
        "movies": []
    }}

    Example 3:
    User's Statement: "I don't know, just something fun."
    JSON Output = 
    {{
        "actors": [],
        "directors": [],
        "genres": ["fun"],
        "movies": []
    }}

    Example 4:
    User's Statement: "I like movies like Toy Story"
    JSON Output = 
    {{
        "actors": [],
        "directors": [],
        "genres": [],
        "movies": ["Toy Story"]
    }}

    User's Statement: "{user_input}"
    JSON Output:
    """
    extractor_prompt = PromptTemplate(template=extractor_template, input_variables=["user_input"])
    return LLMChain(llm=llm, prompt=extractor_prompt)

def get_entity_extractor_chain(llm):
    """
    Extracts movie, actor, director, and genre entities from a user query.
    """
    extractor_template = """
    You are an expert at identifying key entities in a user's question about movies.
    Your task is to extract all movie titles, actor names, director names, and genres mentioned.

    Output Format Instructions:
    - Respond with ONLY a valid JSON object.
    - The JSON object must have the following keys: "movie", "actor", "director", "genre".
    - The value for each key must be a list of strings.
    - If no entity of a certain type is found, its value should be an empty list [].

    --- EXAMPLES ---
    Question: "Who directed The Godfather?"
    JSON Output:
    {{
        "movie": ["The Godfather"],
        "actor": [],
        "director": [],
        "genre": []
    }}

    Question: "Show me some crime movies starring Al Pacino"
    JSON Output:
    {{
        "movie": [],
        "actor": ["Al Pacino"],
        "director": [],
        "genre": ["Crime"]
    }}

    Question: "I want to see a film by Christopher Nolan with Leonardo DiCaprio."
    JSON Output:
    {{
        "movie": [],
        "actor": ["Leonardo DiCaprio"],
        "director": ["Christopher Nolan"],
        "genre": []
    }}
    --- END OF EXAMPLES ---

    Question: "{user_input}"
    JSON Output:
    """
    extractor_prompt = PromptTemplate(template=extractor_template, input_variables=["user_input"])
    return LLMChain(llm=llm, prompt=extractor_prompt)

def get_query_router_chain(llm):
    """
    Classifies a user query into one of: fact_based_search, personalized_recommendation, chit_chat.
    """
    router_prompt_template = """
    You are a highly intelligent query classifier for a movie recommendation chatbot.
    Your task is to analyze the user's query and decide which tool to use.
    The available tools are: 'fact_based_search', 'personalized_recommendation', 'chit_chat'.

    - Use 'fact_based_search' for specific questions about movies, actors, directors, or constrained recommendations.
    - Use 'personalized_recommendation' for open-ended recommendations based on taste.
    - Use 'chit_chat' for conversational greetings or off-topic queries.

    User Query: "{user_query}"
    Tool:
    """
    router_prompt = PromptTemplate(template=router_prompt_template, input_variables=["user_query"])
    return LLMChain(llm=llm, prompt=router_prompt, verbose=False)

def get_cypher_generation_chain(llm):
    """
    Stable Cypher generation chain.
    - When the question is about a specific movie, return only movie properties.
    - Enforce DISTINCT and relationship directions.
    - Avoid unnecessary expansions unless explicitly requested.
    """
    cypher_examples = [
        {
            "question": "Find movies starring Tom Hanks.",
            "query": "MATCH (m:Movie)<-[:ACTED_IN]-(a:Actor {{name: 'Tom Hanks'}}) RETURN m.movieId, m.title, m.overview"
        },
        {
            "question": "Find movies directed by Steven Spielberg that star Tom Hanks.",
            "query": """
            MATCH (a:Actor {{name: 'Tom Hanks'}})-[:ACTED_IN]->(m:Movie)
            WITH m
            MATCH (d:Director {{name: 'Steven Spielberg'}})-[:DIRECTED]->(m)
            RETURN m.movieId, m.title
            """
        },
        {
            "question": "What year was The Matrix released?",
            "query": "MATCH (m:Movie {{title: 'Matrix, The'}}) RETURN m.movieId, m.title, m.overview, m.year"
        },
        {
            "question": "Find Sci-Fi movies starring Harrison Ford.",
            "query": "MATCH (a:Actor {{name: 'Harrison Ford'}})-[:ACTED_IN]->(m:Movie)-[:HAS_GENRE]->(:Genre {{name: 'Sci-Fi'}}) RETURN m.movieId, m.title"
        }
    ]

    example_prompt = PromptTemplate(
        template="User Question: {question}\nCypher Query:\n{query}",
        input_variables=["question", "query"]
    )

    cypher_prompt = FewShotPromptTemplate(
        examples=cypher_examples,
        example_prompt=example_prompt,
        prefix="""
        Task: Generate a read-only Cypher query to retrieve information from a Neo4j database.
        You are an expert Neo4j Cypher translator.

        IMPORTANT RULES:
        - If the question is about a specific movie from preferences,
          generate ONLY a query that MATCHes that Movie node and RETURN its properties:
            m.movieId, m.title, m.overview, m.year (Never return only m.)
        - Do NOT add related movies, Genre nodes, or any other expansions 
          unless the user explicitly asks for "similar" or "related movies".
        - For Actor, Director, or Genre queries, you may traverse their relationships.
        - Always use standard property syntax like {{property: 'value'}}.
        - Assume relationship directions:
            * (Movie)-[:HAS_GENRE]->(Genre)
            * (Actor)-[:ACTED_IN]->(Movie)
            * (Director)-[:DIRECTED]->(Movie)
        - Always use DISTINCT when returning multiple nodes to avoid duplicates.

        Schema:
        {schema}
        """,
        suffix="""
        User Question: {question}
        Respond with ONLY the Cypher query.
        Cypher Query:
        """,
        input_variables=["schema", "question"],
        example_separator="\n\n"
    )
    return LLMChain(llm=llm, prompt=cypher_prompt, verbose=False)

def get_subgraph_cypher_chain(llm):
    """
    Generates Cypher to extract a subgraph using undirected patterns (m)-[r]-(n).
    """
    subgraph_cypher_template = """
    Task: Generate a read-only Cypher query to retrieve a subgraph from a Neo4j database.
    
    Instructions:
    1. You will be given a list of movie IDs.
    2. First, find all movies with these IDs using a 'MATCH (m:Movie) WHERE m.movieId IN [...]' clause.
    3. Then, find all nodes directly connected to these movies.
    4. To retrieve all connected nodes regardless of the relationship direction, you MUST use the undirected pattern `(m)-[r]-(n)`. Do NOT use directed patterns like `(m)->(n)` or `(m)<-(n)`.
    5. The query should return the matched movies, the connecting relationships, and the connected neighboring nodes. Use the pattern `RETURN m, r, n`.
    
    Schema:
    {schema}
    
    List of Movie IDs:
    {movie_ids}
    
    Respond with ONLY the Cypher query.
    Cypher Query:
    """
    subgraph_cypher_prompt = PromptTemplate(template=subgraph_cypher_template, input_variables=["schema", "movie_ids"])
    return LLMChain(llm=llm, prompt=subgraph_cypher_prompt, verbose=False)

def get_personalized_response_chain(llm):
    """
    Final natural language response generator after GNN re-ranking and preference extraction.
    """
    final_response_template = """
    You are a helpful and knowledgeable movie recommendation expert.
    Your task is to perform a final re-ranking of candidate movies and generate a natural, compelling response to the user.

    Here is the context:
    1. User's Original Query: This is what the user initially asked. Pay attention to the nuance and tone.
    2. Extracted Preferences: Key entities extracted from the user's query.
    3. Candidate Movies: A list of movies pre-ranked by a graph-based AI (GNN). The GNN score reflects structural importance in the knowledge graph.

    Your step-by-step instructions:
    Step 1: Deeply understand the user's taste by analyzing the "User's Original Query" and "Extracted Preferences". Look for themes, genres, moods, and key entities.
    Step 2: For each movie in "Candidate Movies", read its overview. Re-rank the candidates based on how well their overview semantically matches the user's taste you identified in Step 1. Use the GNN importance score as a secondary factor or a tie-breaker.
    Step 3: Select the top 2-3 best movies from your new ranking.
    Step 4: Craft a friendly and persuasive response. For each recommended movie, briefly explain *why* you are recommending it, connecting its overview to the user's query. Present the results in a clear and appealing format.

    --- CONTEXT ---
    User's Original Query: {user_query}
    Extracted Preferences: {preferences_str}
    Candidate Movies (pre-ranked by GNN): 
    {candidates_str}
    --- END OF CONTEXT ---

    Final Recommendation:
    """
    prompt = PromptTemplate(template=final_response_template, input_variables=["user_query", "preferences_str", "candidates_str"])
    return LLMChain(llm=llm, prompt=prompt)

def get_fact_based_response_chain(llm):
    """
    Formats database query results into a natural response.
    """
    response_formatter_template = """
    You are a friendly and helpful movie chatbot assistant.
    Your task is to answer the user's question based on the data retrieved from a database.
    Format the data into a clear, natural, and helpful answer.

    - If the retrieved data is an empty list or contains no useful information, politely state that you couldn't find a specific answer.
    - If the data is a list of movies, people, or other entities, present it clearly, perhaps using bullet points if appropriate.
    - If the data is a single value (like a year or a name), state the answer directly.
    - Do not just repeat the data structure. Explain it naturally.

    --- CONTEXT ---
    User's Original Question: {user_query}
    Retrieved Data from Database:
    {cypher_result}
    --- END OF CONTEXT ---

    Helpful Answer:
    """
    prompt = PromptTemplate(template=response_formatter_template, input_variables=["user_query", "cypher_result"])
    return LLMChain(llm=llm, prompt=prompt)

def get_chit_chat_chain(llm):
    """
    Stateless chit-chat chain for casual, non-factual user queries.
    """
    chit_chat_template = """
    You are 'CineMate', a friendly, witty, and knowledgeable movie recommendation chatbot.
    Your main identity is a movie expert who loves talking about films. Your tone should be natural and engaging.

    Your goal is to handle a casual user message by following this 3-step strategy:
    1. Acknowledge and directly respond to the user's message.
    2. Find a creative link or bridge from the user's topic to a movie-related topic.
    3. End your response with an open-ended question about movies to guide the user.

    User's Message: "{user_input}"
    CineMate's Witty Response:
    """
    prompt = PromptTemplate(template=chit_chat_template, input_variables=["user_input"])
    return LLMChain(llm=llm, prompt=prompt)

def get_personalized_guide_chain(llm):
    """
    Generates a Cypher query or asks clarifying questions for personalized recommendations.
    """
    prompt_template = PromptTemplate(
        input_variables=["user_query", "preferences", "schema"],
        template="""
            You are an expert Neo4j Cypher generator for a movie recommendation system.

            User Query: {user_query}
            User Preferences: {preferences}

            Task:
            - If preferences are empty or insufficient, respond with a question to elicit more info (genre, actor, movie).
            - If preferences are sufficient, generate a Cypher query to get movies that match user preferences.
            - Use the schema below for reference.
            - Return ONLY the Cypher query if generating one, or the question text if asking for more info.

            Schema:
            {schema}

            Output:
            """
    )
    return LLMChain(llm=llm, prompt=prompt_template, verbose=False)
