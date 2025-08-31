import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
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
            return []
            
        with self._driver.session() as session:
            try:
                result = session.run(query, parameters)
                return [record for record in result]
            except Exception as e:
                print(f"An error occurred during query execution: {e}")
                return None

def get_neo4j_connection():
    """Loads environment variables and returns a Neo4jConnection instance."""
    #load_dotenv()
    dotenv_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(dotenv_path=dotenv_path)
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    if not all([uri, user, password]):
        raise ValueError("Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in your .env file")
    return Neo4jConnection(uri, user, password)