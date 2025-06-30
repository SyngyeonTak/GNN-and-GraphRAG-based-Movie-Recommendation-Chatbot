import pickle
from neo4j_utils import get_neo4j_connection

def export_graph_data_to_file():    
    graph_data_for_pickle = {}
    conn = None
    try:
        # 1. connect to the local Neo4j
        conn = get_neo4j_connection() 
        print("Connected to local Neo4j.")

        # 2. extract node data 
        node_types = ['User', 'Movie', 'Genre', 'Actor', 'Director']
        node_mappings = {}
        node_names = {node_type: [] for node_type in node_types}

        for node_type in node_types:
            name_property = ""
            if node_type == 'Movie':
                name_property = ", n.title AS name"
            elif node_type in ['Genre', 'Actor', 'Director']:
                name_property = ", n.name AS name"
            
            query = f"MATCH (n:{node_type}) RETURN elementId(n) AS neo4j_id{name_property}"
            records = conn.execute_query(query)

            node_mappings[node_type] = {record["neo4j_id"]: i for i, record in enumerate(records)}
            if name_property:
                node_names[node_type] = [record["name"] for record in records]
        
        graph_data_for_pickle['node_mappings'] = node_mappings
        graph_data_for_pickle['node_names'] = node_names
        print("Node data extracted.")

        # 3. extract edges
        edge_definitions = [
            ('User', 'RATED', 'Movie'), 
            ('Movie', 'HAS_GENRE', 'Genre'),
            ('Actor', 'ACTED_IN', 'Movie'), 
            ('Director', 'DIRECTED', 'Movie')
        ]
        edge_indices = {}
        for src_type, rel_type, dst_type in edge_definitions:
            query = f"MATCH (src:{src_type})-[r:{rel_type}]->(dst:{dst_type}) RETURN elementId(src) AS src_id, elementId(dst) AS dst_id"
            records = conn.execute_query(query)
            src_nodes = [node_mappings[src_type][r['src_id']] for r in records]
            dst_nodes = [node_mappings[dst_type][r['dst_id']] for r in records]
            edge_key = (src_type.lower(), rel_type.lower(), dst_type.lower())
            edge_indices[edge_key] = [src_nodes, dst_nodes]
        
        graph_data_for_pickle['edge_indices'] = edge_indices
        print("Edge data extracted.")

        # 4. download in Pickle 
        with open('./dataset/graph_snapshot.pkl', 'wb') as f:
            pickle.dump(graph_data_for_pickle, f)
        
        print("\nSuccessfully exported graph data to 'graph_snapshot.pkl'")

    finally:
        if conn:
            conn.close()

def verify_snapshot_file(snapshot_path='graph_snapshot.pkl'):
    """
    Loads the graph snapshot file and prints the number of nodes and relationships
    for each type to verify its contents.
    """
    print(f"--- Verifying Snapshot File: '{snapshot_path}' ---")

    try:
        with open(snapshot_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"\nError: Snapshot file not found at '{snapshot_path}'.")
        print("Please make sure you have run the export script first and the file is in the same directory.")
        return
    except Exception as e:
        print(f"\nAn error occurred while reading the file: {e}")
        return

    # --- 1. Verify Node Counts ---
    print("\n[Node Counts]")
    try:
        node_mappings = data['node_mappings']
        if not node_mappings:
            print("  No node data found.")
        else:
            # Sort items for consistent output order
            for node_type, mapping in sorted(node_mappings.items()):
                # The number of nodes is the number of entries in the mapping dictionary
                count = len(mapping)
                print(f"  - {node_type.capitalize():<10}: {count} nodes")
    except KeyError:
        print("  'node_mappings' key not found in the snapshot file.")


    # --- 2. Verify Relationship Counts ---
    print("\n[Relationship Counts]")
    try:
        edge_indices = data['edge_indices']
        if not edge_indices:
            print("  No relationship data found.")
        else:
            # Sort items for consistent output order
            for edge_key, edge_list in sorted(edge_indices.items()):
                src, rel, dst = edge_key
                # The number of relationships is the length of the source/destination node lists
                count = len(edge_list[0])
                # Format for readability
                formatted_rel = f"({src.capitalize()})-[{rel.upper()}]->({dst.capitalize()})"
                print(f"  - {formatted_rel:<40}: {count} relationships")
    except KeyError:
        print("  'edge_indices' key not found in the snapshot file.")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    #export_graph_data_to_file()
    verify_snapshot_file(snapshot_path='./dataset/graph_snapshot.pkl')