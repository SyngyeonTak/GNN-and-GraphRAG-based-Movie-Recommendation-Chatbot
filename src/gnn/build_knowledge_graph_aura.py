import os
import pandas as pd
import ast
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase

# -----------------------
# 1. Aura Connection
# -----------------------
class Neo4jConnection:
    """Neo4j Aura connection manager"""
    def __init__(self, uri, user, password):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._driver.verify_connectivity()
            print(f"✅ Connected to Neo4j Aura: {self._uri}")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j Aura: {e}")

    def close(self):
        if self._driver is not None:
            self._driver.close()
            print("🔌 Connection closed.")

    def execute_query(self, query, parameters=None):
        with self._driver.session() as session:
            try:
                result = session.run(query, parameters)
                return [record for record in result]
            except Exception as e:
                print(f"❌ Query failed: {e}")
                return None

# -----------------------
# 2. Utils
# -----------------------
def clear_database(conn):
    """Delete all nodes & relationships (batch mode for Aura)"""
    print("🧹 Clearing database...")
    delete_query = """
    CALL apoc.periodic.iterate(
      'MATCH (n) RETURN id(n) AS id',
      'MATCH (n) WHERE id(n) = id DETACH DELETE n',
      {batchSize: 5000, iterateList: true}
    )
    """
    conn.execute_query(delete_query)
    print("✅ Database cleared.")

def create_constraints(conn):
    """Create constraints for performance & uniqueness"""
    print("⚙️ Creating constraints...")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Actor) REQUIRE a.name IS UNIQUE")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Director) REQUIRE d.name IS UNIQUE")
    print("✅ Constraints created.")

# -----------------------
# 3. Import Functions
# -----------------------
def import_movies_batch(conn, movies_df, batch_size=500):
    """Import Movie, Genre, Actor, Director in batches"""
    print("🎬 Importing movies, genres, actors, directors...")
    for i in tqdm(range(0, len(movies_df), batch_size), desc="Movies Batch"):
        batch = movies_df.iloc[i:i+batch_size].to_dict("records")
        query = """
        UNWIND $movies AS movie
        MERGE (m:Movie {movieId: movie.movieId})
        ON CREATE SET m.title = movie.title, m.year = movie.year, m.overview = movie.overview
        ON MATCH SET m.title = movie.title, m.year = movie.year, m.overview = movie.overview

        WITH m, movie
        UNWIND coalesce(movie.genres, []) AS genre_name
          MERGE (g:Genre {name: genre_name})
          MERGE (m)-[:HAS_GENRE]->(g)

        WITH m, movie
        UNWIND coalesce(movie.actors, []) AS actor_name
          MERGE (a:Actor {name: actor_name})
          MERGE (a)-[:ACTED_IN]->(m)

        WITH m, movie
        CALL apoc.do.when(movie.director IS NOT NULL,
          'MERGE (d:Director {name: movie.director}) MERGE (d)-[:DIRECTED]->(m)',
          '', {m:m, movie:movie}) YIELD value
        RETURN count(m)
        """
        conn.execute_query(query, {"movies": batch})
    print("✅ Movies import complete.")

def import_ratings_batch(conn, ratings_df, batch_size=1000):
    """Import Users and RATED relationships"""
    print("👤 Importing users & ratings...")

    # 1. Users
    user_ids = ratings_df["userId"].unique()
    for user_id in tqdm(user_ids, desc="Users"):
        conn.execute_query("MERGE (u:User {userId: $userId})", {"userId": int(user_id)})

    # 2. Ratings
    for i in tqdm(range(0, len(ratings_df), batch_size), desc="Ratings Batch"):
        batch = ratings_df.iloc[i:i+batch_size].to_dict("records")
        for r in batch:
            r["rated_at"] = pd.to_datetime(r["rated_at"]).isoformat()
        query = """
        UNWIND $ratings AS row
        MATCH (u:User {userId: row.userId})
        MATCH (m:Movie {movieId: row.movieId})
        MERGE (u)-[r:RATED]->(m)
        ON CREATE SET r.rating = row.rating, r.rated_at = datetime(row.rated_at)
        """
        conn.execute_query(query, {"ratings": batch})
    print("✅ Ratings import complete.")

# -----------------------
# 4. Main
# -----------------------
def main():
    load_dotenv()
    uri = os.getenv("NEO4J_AURA_URI")
    user = os.getenv("NEO4J_AURA_USER")
    password = os.getenv("NEO4J_AURA_PASSWORD")

    conn = Neo4jConnection(uri, user, password)

    # Load CSVs
    movies_df = pd.read_csv("dataset/processed/movies_filtered_decade.csv")
    ratings_df = pd.read_csv("dataset/processed/ratings_filtered_decade.csv")

    # Convert string list cols back to list
    for col in ["genres", "actors"]:
        movies_df[col] = movies_df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

    # Filter movies with both director & actors
    movies_df = movies_df[movies_df["director"].notna() & movies_df["actors"].apply(lambda x: len(x) > 0)]
    ratings_df = ratings_df[ratings_df["movieId"].isin(movies_df["movieId"].unique())]

    print(f"📊 Movies: {len(movies_df)}, Ratings: {len(ratings_df)}")

    # Populate Neo4j
    clear_database(conn)
    create_constraints(conn)
    import_movies_batch(conn, movies_df)
    import_ratings_batch(conn, ratings_df)

    conn.close()
    print("🚀 Export complete! All data imported into Neo4j Aura.")

if __name__ == "__main__":
    main()
