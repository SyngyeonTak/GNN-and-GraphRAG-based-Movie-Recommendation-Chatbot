import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm
import ast

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
    
    # This query repeatedly finds 10,000 nodes, detaches and deletes them,
    # until no nodes are left. This avoids loading everything into memory at once.
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
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")
    conn.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
    print("Constraint creation complete.")

def import_movies_and_related_nodes(conn, movies_df):
    """Imports Movie, Genre, Person (Actor, Director) nodes and their relationships."""
    print("Importing Movie, Genre, and Person (Actor/Director) nodes and relationships...")
    
    movies_df = movies_df.where(pd.notna(movies_df), None)

    for _, row in tqdm(movies_df.iterrows(), total=movies_df.shape[0], desc="Importing Movies"):
        # MODIFIED: The Cypher query below has been updated to prevent constraint violations.
        query = """
        // 1. Create or update the Movie node
        MERGE (m:Movie {movieId: $movieId})
        ON CREATE SET m.title = $title, m.year = $year, m.overview = $overview
        ON MATCH SET m.title = $title, m.year = $year, m.overview = $overview

        // 2. Create Genre nodes and relationships
        WITH m
        UNWIND $genres AS genre_name
        MERGE (g:Genre {name: genre_name})
        MERGE (m)-[:HAS_GENRE]->(g)

        // 3. Create Director node and relationship (Modified)
        WITH m
        CALL apoc.do.when($director IS NOT NULL,
            // First, MERGE on the Person's name, then SET the Director label.
            'MERGE (p:Person {name: $director}) SET p:Director MERGE (p)-[:DIRECTED]->(m)',
            '',
            {m: m, director: $director}) YIELD value

        // 4. Create Actor nodes and relationships (Modified)
        WITH m
        CALL apoc.do.when($actors IS NOT NULL,
            // First, MERGE on each Person's name, then SET the Actor label.
            'UNWIND $actors AS actor_name
             MERGE (p:Person {name: actor_name})
             SET p:Actor
             MERGE (p)-[:ACTED_IN]->(m)',
            '',
            {m: m, actors: $actors}) YIELD value
            
        RETURN count(m)
        """
        conn.execute_query(query, parameters=row.to_dict())
    print("Movie-related data import complete.")

def import_ratings(conn, ratings_df):
    """Imports User nodes and RATED relationships (in batches)."""
    print("Importing user rating (RATED) relationships...")

    # 1. Create User nodes
    print("Creating User nodes...")
    user_ids = ratings_df['userId'].unique()
    for user_id in tqdm(user_ids, desc="Creating Users"):
        conn.execute_query("MERGE (u:User {userId: $userId})", parameters={'userId': int(user_id)})
    print("User node creation complete.")
    
    # 2. Create RATED relationships (in batches for performance)
    print("Creating RATED relationships (batch processing)...")
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
        # Convert rated_at to ISO 8601 string format for Neo4j's datetime function
        batch_dict = batch.to_dict('records')
        for record in batch_dict:
            record['rated_at'] = pd.to_datetime(record['rated_at']).isoformat()
        
        conn.execute_query(query, parameters={'ratings': batch_dict})

    print("Rating relationship import complete.")


def main():
    """Main execution function."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Neo4j connection details
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")

    # Data file paths
    PROCESSED_DATA_PATH = './dataset/processed/'
    movies_file = os.path.join(PROCESSED_DATA_PATH, 'movies_processed.csv')
    ratings_file = os.path.join(PROCESSED_DATA_PATH, 'ratings_processed.csv')

    # Load data from CSV files
    print("Loading processed CSV files...")
    movies_df = pd.read_csv(movies_file)
    
    # Convert string representations of lists back to actual lists
    for col in ['genres', 'actors']:
        movies_df[col] = movies_df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else None)
        
    ratings_df = pd.read_csv(ratings_file)
    print("File loading complete.")

    # Establish Neo4j connection
    conn = Neo4jConnection(uri, user, password)
    
    # --- Populate the database ---
    # 1. Initialize the database (Warning: deletes all existing data)
    clear_database(conn)
    
    # 2. Create constraints
    create_constraints(conn)
    
    # 3. Import movies and related nodes/relationships
    import_movies_and_related_nodes(conn, movies_df)
    
    # 4. Import users and rating relationships
    import_ratings(conn, ratings_df)
    
    # Close the connection
    conn.close()
    
    print("\n✅ All data has been successfully imported into Neo4j.")


if __name__ == "__main__":
    main()