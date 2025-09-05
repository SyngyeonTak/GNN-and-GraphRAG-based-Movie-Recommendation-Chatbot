import os
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
import ast
from neo4j_utils import get_neo4j_connection 

class Neo4jConnection:
    """Manages the connection to the Neo4j database."""
    def __init__(self, uri, user, password):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._driver.verify_connectivity()
            print("Successfully connected to the Neo4j database.")
        except Exception as e:
            print(f"Failed to connect to the database: {e}")

    def close(self):
        if self._driver is not None:
            self._driver.close()
            print("Neo4j connection closed.")

    def execute_query(self, query, parameters=None):
        """Executes a Cypher query."""
        if self._driver is None:
            print("Not connected to the database.")
            return
        with self._driver.session() as session:
            try:
                result = session.run(query, parameters)
                return [record for record in result]
            except Exception as e:
                print(f"An error occurred during query execution: {e}")
                return None

def clear_database(conn):
    """Deletes all nodes and relationships in the database using batches to avoid memory issues."""
    print("Initializing the database (using batch deletion)...")
    delete_query = """
    CALL apoc.periodic.iterate(
      'MATCH (n) RETURN id(n) AS id',
      'MATCH (n) WHERE id(n) = id DETACH DELETE n',
      {batchSize: 10000, iterateList: true}
    )
    """
    conn.execute_query(delete_query)
    print("Database initialization complete.")

def create_constraints(conn):
    """Creates constraints for data integrity and performance."""
    print("Creating constraints...")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
    print("Constraint creation complete.")

def import_movies_and_related_nodes(conn, movies_df):
    """Imports Movie, Genre, Person (Actor, Director) nodes and their relationships."""
    print("Importing Movie, Genre, and Person (Actor/Director) nodes and relationships...")
    
    movies_df = movies_df.where(pd.notna(movies_df), None)

    for _, row in tqdm(movies_df.iterrows(), total=movies_df.shape[0], desc="Importing Movies"):
        query = """
        // 1. Create or update the Movie node
        MERGE (m:Movie {movieId: $movieId})
        ON CREATE SET m.title = $title, m.year = $year, m.overview = $overview,
                      m.avgRating = $avg_rating, m.ratingCount = $rating_count
        ON MATCH SET m.title = $title, m.year = $year, m.overview = $overview,
                      m.avgRating = $avg_rating, m.ratingCount = $rating_count

        // 2. Create Genre nodes and relationships
        WITH m
        UNWIND CASE WHEN $genres IS NULL THEN [] ELSE $genres END AS genre_name
        MERGE (g:Genre {name: genre_name})
        MERGE (m)-[:HAS_GENRE]->(g)

        // 3. Create Actor nodes and relationships
        WITH m
        UNWIND CASE WHEN $actors IS NULL THEN [] ELSE $actors END AS actor_name
        MERGE (a:Actor {name: actor_name})
        SET a:Actor
        MERGE (a)-[:ACTED_IN]->(m)

        // 4. Create Director node and relationship
        WITH m
        CALL apoc.do.when($director IS NOT NULL,
            'MERGE (d:Director {name: $director}) SET d:Director MERGE (d)-[:DIRECTED]->(m)',
            '',
            {m: m, director: $director}) YIELD value        

        RETURN count(m)
        """
        conn.execute_query(query, parameters=row.to_dict())
        
    print("Movie-related data import complete.")

def main():
    """Main execution function."""
    print(os.getcwd())
    conn = get_neo4j_connection()
    
    # Data file path (only movies file is needed now)
    PROCESSED_DATA_PATH = 'dataset/processed/'
    movies_file = os.path.join(PROCESSED_DATA_PATH, 'movies_processed_final.csv')

    # Load data from CSV file
    print("Loading processed CSV file...")
    movies_df = pd.read_csv(movies_file)
    
    # Convert string representations of lists back to actual lists
    for col in ['genres', 'actors']:
        movies_df[col] = movies_df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else None)
    print("File loading complete.")

    # --- Populate the database ---
    clear_database(conn)
    create_constraints(conn)
    import_movies_and_related_nodes(conn, movies_df)
    
    conn.close()
    print("\n✅ All data has been successfully imported into Neo4j.")

if __name__ == "__main__":
    main()
