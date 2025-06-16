# src/populate_db.py

import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm

class Neo4jConnection:
    """Neo4j 데이터베이스와의 연결을 관리하는 클래스"""
    def __init__(self, uri, user, password):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._driver.verify_connectivity()
            print("🎉 Neo4j 데이터베이스에 성공적으로 연결되었습니다.")
        except Exception as e:
            print(f"❌ 데이터베이스 연결에 실패했습니다: {e}")

    def close(self):
        if self._driver is not None:
            self._driver.close()
            print("Neo4j 연결이 닫혔습니다.")

    def execute_query(self, query, parameters=None):
        """Cypher 쿼리를 실행하는 메소드"""
        if self._driver is None:
            print("데이터베이스에 연결되어 있지 않습니다.")
            return
            
        with self._driver.session() as session:
            try:
                result = session.run(query, parameters)
                return [record for record in result]
            except Exception as e:
                print(f"쿼리 실행 중 에러 발생: {e}")
                return None

def clear_database(conn):
    """데이터베이스의 모든 노드와 관계를 삭제합니다."""
    print("기존 데이터베이스를 초기화합니다...")
    conn.execute_query("MATCH (n) DETACH DELETE n")
    print("데이터베이스 초기화 완료.")

def create_constraints(conn):
    """성능 향상 및 데이터 무결성을 위해 제약 조건을 생성합니다."""
    print("제약 조건을 생성합니다...")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
    print("제약 조건 생성 완료.")

def import_movies_and_related_nodes(conn, movies_df):
    """Movie, Genre, Person(Actor, Director) 노드와 그 관계를 임포트합니다."""
    print("영화, 장르, 인물(배우/감독) 노드 및 관계를 임포트합니다...")
    
    # NaN 값을 Python의 None으로 변환
    movies_df = movies_df.where(pd.notna(movies_df), None)

    for _, row in tqdm(movies_df.iterrows(), total=movies_df.shape[0], desc="Importing Movies"):
        query = """
        // 1. Movie 노드 생성 (없으면 만들고, 있으면 속성 업데이트)
        MERGE (m:Movie {movieId: $movieId})
        ON CREATE SET m.title = $title, m.year = $year
        ON MATCH SET m.title = $title, m.year = 'year'

        // 2. Genre 노드 및 HAS_GENRE 관계 생성
        // 리스트의 각 장르에 대해 반복
        WITH m
        UNWIND $genres AS genre_name
        MERGE (g:Genre {name: genre_name})
        MERGE (m)-[:HAS_GENRE]->(g)

        // 3. Director 노드 및 DIRECTED 관계 생성
        // director가 null이 아닌 경우에만 실행
        WITH m
        CALL apoc.do.when($director IS NOT NULL,
            'MERGE (p:Person:Director {name: $director}) MERGE (p)-[:DIRECTED]->(m)',
            '',
            {m: m, director: $director}) YIELD value

        // 4. Actor 노드 및 ACTED_IN 관계 생성
        // actors 리스트가 null이 아닌 경우에만 실행
        WITH m
        CALL apoc.do.when($actors IS NOT NULL,
            'UNWIND $actors AS actor_name
             MERGE (p:Person:Actor {name: actor_name})
             MERGE (p)-[:ACTED_IN]->(m)',
            '',
            {m: m, actors: $actors}) YIELD value
        """
        conn.execute_query(query, parameters=row.to_dict())
    print("영화 관련 데이터 임포트 완료.")

def import_ratings(conn, ratings_df):
    """User 노드와 RATED 관계를 임포트합니다. (배치 처리)"""
    print("사용자 평점(RATED) 관계를 임포트합니다...")

    # 1. User 노드 생성
    print("User 노드를 생성합니다...")
    user_ids = ratings_df['userId'].unique()
    for user_id in tqdm(user_ids, desc="Creating Users"):
        conn.execute_query("MERGE (u:User {userId: $userId})", parameters={'userId': int(user_id)})
    print("User 노드 생성 완료.")
    
    # 2. RATED 관계 생성 (성능을 위해 배치로 처리)
    print("RATED 관계를 생성합니다 (배치 처리 중)...")
    batch_size = 1000
    for i in tqdm(range(0, len(ratings_df), batch_size), desc="Importing Ratings"):
        batch = ratings_df.iloc[i:i + batch_size]
        query = """
        UNWIND $ratings AS rating_row
        MATCH (u:User {userId: rating_row.userId})
        MATCH (m:Movie {movieId: rating_row.movieId})
        MERGE (u)-[r:RATED]->(m)
        ON CREATE SET r.rating = rating_row.rating, r.rated_at = datetime(rating_row.rated_at)
        """
        # rated_at을 ISO 8601 문자열로 변환하여 전달
        batch_dict = batch.to_dict('records')
        for record in batch_dict:
            record['rated_at'] = pd.to_datetime(record['rated_at']).isoformat()
        
        conn.execute_query(query, parameters={'ratings': batch_dict})

    print("평점 관계 임포트 완료.")


def main():
    """메인 실행 함수"""
    # .env 파일에서 환경 변수 로드
    load_dotenv()
    
    # Neo4j 접속 정보
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")

    # 데이터 파일 경로
    PROCESSED_DATA_PATH = './dataset/processed/'
    movies_file = os.path.join(PROCESSED_DATA_PATH, 'movies_processed.csv')
    ratings_file = os.path.join(PROCESSED_DATA_PATH, 'ratings_processed.csv')

    # 데이터 불러오기
    print("처리된 CSV 파일을 불러옵니다...")
    movies_df = pd.read_csv(movies_file)
    # ast.literal_eval을 사용해 문자열화된 리스트를 실제 리스트로 변환
    import ast
    for col in ['genres', 'actors']:
        movies_df[col] = movies_df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else None)
        
    ratings_df = pd.read_csv(ratings_file)
    print("파일 불러오기 완료.")

    # Neo4j 연결
    conn = Neo4jConnection(uri, user, password)
    
    # --- 데이터베이스에 데이터 적재 ---
    # 1. 데이터베이스 초기화 (주의: 기존 모든 데이터가 삭제됩니다)
    clear_database(conn)
    
    # 2. 제약 조건 생성
    create_constraints(conn)
    
    # 3. 영화 및 관련 노드/관계 임포트
    import_movies_and_related_nodes(conn, movies_df)
    
    # 4. 사용자 및 평점 관계 임포트
    import_ratings(conn, ratings_df)
    
    # 연결 종료
    conn.close()
    
    print("\n✅ 모든 데이터가 성공적으로 Neo4j에 임포트되었습니다.")


if __name__ == "__main__":
    main()