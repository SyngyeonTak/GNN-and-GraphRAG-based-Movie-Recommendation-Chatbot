from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain

def get_genre_mapper_chain(llm):
    """사용자가 말한 장르와 가장 의미가 유사한 장르를 DB에서 찾아주는 체인"""
    mapper_template = """
    You are a genre mapping expert. A user mentioned the genre '{user_genre}'.
    From the following list of available genres, which one is the closest semantic match?
    Available Genres: {available_genres}
    Respond with ONLY the single best matching genre name from the list. If no good match exists, respond with 'None'.
    """
    mapper_prompt = PromptTemplate(template=mapper_template, input_variables=["user_genre", "available_genres"])
    return LLMChain(llm=llm, prompt=mapper_prompt)

def get_movie_suggester_chain(llm):
    """주어진 선호도에 대해 대표적인 영화 제목을 추천하는 체인"""
    suggester_template = """
    You are an expert movie recommender. Based on the user's preference for a genre, actor, or theme, please suggest 3 to 5 famous and representative movie titles.
    Your answer MUST be ONLY a comma-separated list of movie titles.

    User's Stated Preference: "{preference}"
    
    Comma-Separated Movie Titles:
    """
    suggester_prompt = PromptTemplate(template=suggester_template, input_variables=["preference"])
    return LLMChain(llm=llm, prompt=suggester_prompt)

def get_preference_extractor_chain(llm):
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
    # 이 라인이 실제 실행되는지 확인하는 것이 중요합니다.
    extractor_prompt = PromptTemplate(template=extractor_template, input_variables=["user_input"])

    # ==========================================================
    # 최종 디버깅: 생성된 프롬프트 객체의 input_variables를 직접 출력합니다.
    print("DEBUG: Final created prompt object's input_variables:", extractor_prompt.input_variables)
    # ==========================================================

    return LLMChain(llm=llm, prompt=extractor_prompt)

def get_entity_extractor_chain(llm):
    """
    쿼리에서 영화, 배우, 감독, 장르 개체를 추출하여
    JSON 형식으로 반환하는 체인
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
    extractor_prompt = PromptTemplate(
        template=extractor_template,
        input_variables=["user_input"]
    )
    
    return LLMChain(llm=llm, prompt=extractor_prompt)

def get_query_router_chain(llm):
    """사용자 쿼리를 세 가지 카테고리 중 하나로 분류하는 체인"""
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

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain

def get_cypher_generation_chain(llm):
    """
    안정적인 Cypher 생성용 LLMChain.
    - 복합 조건은 MATCH 체인 또는 WITH 서브쿼리 스타일로 유도
    - Cypher 문법 중괄호는 LangChain 템플릿 변수 오인 방지를 위해 {{}}로 이스케이프
    """
    # 1. LLM에게 보여줄 예시
    cypher_examples = [
        {
            "question": "Find movies starring Tom Hanks.",
            "query": "MATCH (m:Movie)<-[:ACTED_IN]-(a:Actor {{name: 'Tom Hanks'}}) RETURN m.title, m.overview"
        },
        {
            "question": "Find movies directed by Steven Spielberg that star Tom Hanks.",
            "query": """
            MATCH (a:Actor {{name: 'Tom Hanks'}})-[:ACTED_IN]->(m:Movie)
            WITH m
            MATCH (d:Director {{name: 'Steven Spielberg'}})-[:DIRECTED]->(m)
            RETURN m.title
            """
        },
        {
            "question": "What year was The Matrix released?",
            "query": "MATCH (m:Movie {{title: 'The Matrix'}}) RETURN m.year"
        },
        {
            "question": "Find Sci-Fi movies starring Harrison Ford.",
            "query": "MATCH (a:Actor {{name: 'Harrison Ford'}})-[:ACTED_IN]->(m:Movie)-[:HAS_GENRE]->(g:Genre {{name: 'Sci-Fi'}}) RETURN m.title"
        }
    ]

    # 2. 예시 형식 정의
    example_prompt = PromptTemplate(
        template="User Question: {question}\nCypher Query:\n{query}",
        input_variables=["question", "query"]
    )

    # 3. 최종 프롬프트 구성
    cypher_prompt = FewShotPromptTemplate(
        examples=cypher_examples,
        example_prompt=example_prompt,
        prefix="""
        Task: Generate a read-only Cypher query to retrieve information from a Neo4j database.
        You are an expert Neo4j Cypher translator.
        
        IMPORTANT:
        - For multiple constraints (e.g., a director AND an actor), use either:
          (1) Multiple MATCH statements sharing the same variable, or
          (2) A WITH clause to pass variables between matches.
        - Avoid using `(:Label {{property: value}})` directly inside WHERE without a bound variable.
        - In the Cypher output, use standard property syntax like {{property: 'value'}}.
        
        Here are some examples of user questions and their corresponding Cypher queries.
        
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
    [V2] 서브그래프 추출을 위해 '방향이 없는' 쿼리를 생성하도록
    명확하게 지시하는 프롬프트를 사용하는 체인
    """
    subgraph_cypher_template = """
    Task: Generate a read-only Cypher query to retrieve a subgraph from a Neo4j database.
    
    Instructions:
    1. You will be given a list of movie IDs.
    2. First, find all movies with these IDs using a 'MATCH (m:Movie) WHERE m.movieId IN [...]' clause.
    3. Then, find all nodes directly connected to these movies.
    4. **CRUCIAL**: To retrieve all connected nodes regardless of the relationship direction, you MUST use the undirected pattern `(m)-[r]-(n)`. Do NOT use directed patterns like `(m)->(n)` or `(m)<-(n)`.
    5. The query should return the matched movies, the connecting relationships, and the connected neighboring nodes. Use the pattern `RETURN m, r, n`.
    
    Schema:
    {schema}
    
    List of Movie IDs:
    {movie_ids}
    
    Respond with ONLY the Cypher query.
    Cypher Query:
    """
    
    subgraph_cypher_prompt = PromptTemplate(
        template=subgraph_cypher_template,
        input_variables=["schema", "movie_ids"]
    )
    
    return LLMChain(llm=llm, prompt=subgraph_cypher_prompt, verbose=False)

def get_personalized_response_chain(llm):
    """
    GNN 리랭킹 결과와 사용자 쿼리/선호도를 바탕으로,
    최종 추천 답변을 생성하는 LLMChain을 반환합니다.
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

    Respond with ONLY the final, natural language response to be shown to the user.

    Final Recommendation:
    """
    
    prompt = PromptTemplate(
        template=final_response_template,
        input_variables=["user_query", "preferences_str", "candidates_str"]
    )
    
    return LLMChain(llm=llm, prompt=prompt)

def get_fact_based_response_chain(llm):
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
    
    prompt = PromptTemplate(
        template=response_formatter_template,
        input_variables=["user_query", "cypher_result"]
    )
    
    return LLMChain(llm=llm, prompt=prompt)

def get_chit_chat_chain(llm):
    """
    Creates a stateless Chit-Chat chain that responds to a single user input
    without needing conversation history.
    """
    chit_chat_template = """
    You are 'CineMate', a friendly, witty, and knowledgeable movie recommendation chatbot.
    Your main identity is a movie expert who loves talking about films. Your tone should be natural and engaging.

    Your goal is to handle a casual user message by following this 3-step strategy:
    1. Acknowledge and directly respond to the user's message.
    2. Find a creative link or bridge from the user's topic to a movie-related topic.
    3. End your response with an open-ended question about movies to guide the user.

    --- EXAMPLES ---
    User's Message: "Hey what's up?"
    CineMate's Witty Response: "Hey there! I'm just sorting through my vast library of films. It's what I do best! Are you looking for a movie that perfectly matches your mood today? 😊"

    User's Message: "The weather is great today!"
    CineMate's Witty Response: "That's great to hear! ☀️ A beautiful day like this is perfect for a movie with gorgeous outdoor cinematography. Have you ever seen 'Call Me by Your Name'?"

    User's Message: "I'm so tired lately."
    CineMate's Witty Response: "It sounds like you've had a long day. At the end of a tiring day, sometimes a laugh-out-loud comedy is the perfect remedy. Would you like a recommendation for a fun movie to unwind with?"
    --- END OF EXAMPLES ---
    
    User's Message: "{user_input}"
    CineMate's Witty Response:
    """
    
    prompt = PromptTemplate(
        template=chit_chat_template,
        # The only input is now the user's message
        input_variables=["user_input"]
    )
    
    return LLMChain(llm=llm, prompt=prompt)

def get_personalized_guide_chain(llm):
    """
    user_query가 fact-based인지, personalized recommendation인지 이미 분기된 상태에서,
    personalized recommendation을 처리할 체인.
    - 사용자 선호(preferences)가 충분하지 않으면 질문 유도
    - 충분하면 Cypher 쿼리를 생성
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

    chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)
    return chain