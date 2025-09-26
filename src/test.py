from neo4j import GraphDatabase

uri = "neo4j+s://4961fcbd.databases.neo4j.io"
user = "neo4j"
password = ""

driver = GraphDatabase.driver(uri, auth=(user, password))

with driver.session() as session:
    result = session.run("RETURN 'Aura connection OK' AS msg")
    for record in result:
        print(record["msg"])