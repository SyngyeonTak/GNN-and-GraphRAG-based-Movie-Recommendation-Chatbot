from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

try:
    driver.verify_connectivity()
    print("✅ Neo4j 연결 성공!")
except Exception as e:
    print("❌ 연결 실패:", e)

with driver.session(database="neo4j") as session:
    result = session.run("MATCH (n) RETURN count(n) AS cnt")
    print(result.single()["cnt"])