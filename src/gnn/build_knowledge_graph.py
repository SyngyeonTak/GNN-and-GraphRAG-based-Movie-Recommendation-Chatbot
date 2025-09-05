import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm
import ast
from neo4j_utils import get_neo4j_connection 
from pathlib import Path

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
            // First, MERGE on the Person's name, then SET the Director label.
            'MERGE (d:Director {name: $director}) SET d:Director MERGE (d)-[:DIRECTED]->(m)',
            '',
            {m: m, director: $director}) YIELD value        

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
    #dotenv_path = Path(__file__).parent.parent.parent / '.env'
    #load_dotenv(dotenv_path=dotenv_path)
    print(os.getcwd())
    conn = get_neo4j_connection()
    
    # Data file paths
    PROCESSED_DATA_PATH = 'dataset/processed/'
    movies_file = os.path.join(PROCESSED_DATA_PATH, 'movies_processed_w_id.csv')
    ratings_file = os.path.join(PROCESSED_DATA_PATH, 'ratings_processed_w_id.csv')

    # Load data from CSV files
    print("Loading processed CSV files...")
    movies_df = pd.read_csv(movies_file)
    
    # Convert string representations of lists back to actual lists
    for col in ['genres', 'actors']:
        movies_df[col] = movies_df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else None)
        
    ratings_df = pd.read_csv(ratings_file)
    print("File loading complete.")

    # --- Filter movies that have both director AND actors ---
    print("\nFiltering movies to import...")
    original_movie_count = len(movies_df)

    director_present_condition = movies_df['director'].notna()
    actors_present_condition = movies_df['actors'].apply(lambda x: isinstance(x, list) and len(x) > 0)

    movies_to_import_df = movies_df[director_present_condition & actors_present_condition]
    
    print(f"Original movie count: {original_movie_count}")
    print(f"Movies to import (with director & actors): {len(movies_to_import_df)}")
    print(f"Number of movies excluded: {original_movie_count - len(movies_to_import_df)}")
    
    # --- NEW: Filter ratings for the remaining movies ---
    print("\nFiltering ratings for the selected movies...")
    original_rating_count = len(ratings_df)
    
    # Get the list of movieIds that will actually be imported
    valid_movie_ids = movies_to_import_df['movieId'].unique()
    
    # Keep only the ratings that belong to the valid movies
    ratings_to_import_df = ratings_df[ratings_df['movieId'].isin(valid_movie_ids)]
    
    print(f"Original rating count: {original_rating_count}")
    print(f"Ratings to import (for selected movies): {len(ratings_to_import_df)}")
    print(f"Number of ratings excluded: {original_rating_count - len(ratings_to_import_df)}\n")
    # ----------------------------------------------------------------

    print("File loading complete.")


    # --- Populate the database ---
    # 1. Initialize the database (Warning: deletes all existing data)
    clear_database(conn)
    
    # 2. Create constraints
    create_constraints(conn)
    
    # 3. Import movies and related nodes/relationships
    import_movies_and_related_nodes(conn, movies_to_import_df)
    
    # 4. Import users and rating relationships
    import_ratings(conn, ratings_to_import_df)
    
    # Close the connection
    conn.close()
    
    print("\n✅ All data has been successfully imported into Neo4j.")


if __name__ == "__main__":
    main()