from langchain_core.prompts import PromptTemplate
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
    """쿼리에서 핵심 개체(영화 제목, 인물 이름)를 추출하는 체인"""
    extractor_template = """
    You are an expert at identifying key entities in a user's question about movies.
    Your task is to extract the main movie title, actor's name, or director's name.
    Respond with ONLY the extracted entity.

    Question: "{user_input}"
    Entity:
    """
    extractor_prompt = PromptTemplate(template=extractor_template, input_variables=["user_input"])
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

def get_cypher_generation_chain(llm):
    """사용자 질문과 그래프 스키마를 기반으로 Cypher 쿼리를 생성하는 체인"""
    cypher_generation_template = """
    Task: Generate a read-only Cypher query to retrieve information from a Neo4j database.
    Schema: {schema}
    User Question: {question}
    Respond with ONLY the Cypher query.
    Cypher Query:
    """
    cypher_prompt = PromptTemplate(template=cypher_generation_template, input_variables=["schema", "question"])
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