import networkx as nx
from collections import Counter
import numpy as np
import torch
from torch_geometric.data import Data


def create_global_nx_graph(snapshot_data: dict):
    """
    Create a global NetworkX graph from snapshot data.
    Movies are represented as movie_{idx},
    and other entities (Actor, Director, Genre) by their names.
    """
    import networkx as nx

    print("--- Creating global NetworkX graph (Movie=movie_idx, Others=name) ---")
    G = nx.Graph()
    
    node_mappings = snapshot_data['node_mappings']
    node_names = snapshot_data.get('node_names', {})
    edge_indices = snapshot_data['edge_indices']

    internal_to_final_id_map = {}

    # Add Actor, Director, Genre nodes (User 제거됨)
    for node_type in ['Actor', 'Director', 'Genre']:
        if node_type in node_mappings:
            mapping = node_mappings[node_type]
            names_list = node_names.get(node_type, [])
            
            for neo4j_id, node_idx in mapping.items():
                if node_idx < len(names_list):
                    final_node_id = names_list[node_idx]
                else:
                    continue  # Skip if no name available
                
                G.add_node(final_node_id, type=node_type)
                internal_to_final_id_map[(node_type, node_idx)] = final_node_id

    # Add Movie nodes
    movie_mapping = node_mappings.get('Movie', {})
    for neo4j_id, node_idx in movie_mapping.items():
        final_node_id = f"movie_{node_idx+1}"
        G.add_node(final_node_id, type='Movie')
        internal_to_final_id_map[('Movie', node_idx)] = final_node_id

    # Add edges between nodes
    for edge_key, (src_nodes, dst_nodes) in edge_indices.items():
        src_type_str, _, dst_type_str = edge_key
        src_type = src_type_str.capitalize()
        dst_type = dst_type_str.capitalize()

        for src_idx, dst_idx in zip(src_nodes, dst_nodes):
            src_final_id = internal_to_final_id_map.get((src_type, src_idx))
            dst_final_id = internal_to_final_id_map.get((dst_type, dst_idx))

            if src_final_id and dst_final_id:
                G.add_edge(src_final_id, dst_final_id)

    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def extract_subgraph_from_global(global_G, seed_movie_ids, top_k=30, random_k=20):
    """
    Extract a minimal subgraph from the global graph, centered on given seed movies.
    - Selects top_k movies by degree + random_k movies from the rest.
    - Connects them using shortest paths (no extra 1-hop/2-hop expansion).
    """
    import random
    import networkx as nx

    if not seed_movie_ids:
        print("No seed movie IDs provided.")
        return nx.Graph()

    # Step 1. Filter seed movies to candidates with degree
    seed_with_degree = [
        (movie_id, global_G.degree(movie_id))
        for movie_id in seed_movie_ids
        if global_G.has_node(movie_id)
    ]
    if not seed_with_degree:
        print("No valid seed nodes found to create a subgraph.")
        return nx.Graph()

    # Sort by degree (descending)
    seed_with_degree.sort(key=lambda x: x[1], reverse=True)

    # Select Top-K movies
    top_movies = [movie_id for movie_id, _ in seed_with_degree[:top_k]]

    # Random sampling from the remaining
    remaining = [movie_id for movie_id, _ in seed_with_degree[top_k:]]
    random_movies = random.sample(remaining, min(random_k, len(remaining)))

    # Final selected movies
    selected_movies = set(top_movies + random_movies)

    # Step 2. Build minimal subgraph via shortest paths
    nodes_for_subgraph = set(selected_movies)
    for i, m1 in enumerate(selected_movies):
        for m2 in list(selected_movies)[i+1:]:
            try:
                path = nx.shortest_path(global_G, source=m1, target=m2)
                nodes_for_subgraph.update(path)
            except nx.NetworkXNoPath:
                continue

    # Step 3. Return subgraph
    subgraph_nx = global_G.subgraph(nodes_for_subgraph).copy()
    return subgraph_nx

def convert_nx_to_pyg(subgraph_nx, assets):
    """
    Convert a NetworkX subgraph into a PyTorch Geometric Data object,
    attaching embeddings from FAISS indices as node features.
    """
    nodes = list(subgraph_nx.nodes())
    node_map = {node_id: i for i, node_id in enumerate(nodes)}

    # Precompute reverse mappings for FAISS IDs
    mappings = {}
    for entity_type, asset_data in assets.items():
        if entity_type == 'movie':
            mappings[entity_type] = {f'movie_{k}': int(k) for k, v in asset_data['mapping'].items()}
        else:
            mappings[entity_type] = {v: int(k) for k, v in asset_data['mapping'].items()}

    node_features = []
    #embedding_dim = assets['movie']['gnn_index'].d  # Dimension from FAISS index
    embedding_dim = 64  
    for node_id in nodes:
        #try:
        #    node_type = subgraph_nx.nodes[node_id].get('type').lower()
        #    faiss_id = mappings[node_type][node_id]
        #    embedding = assets[node_type]['gnn_index'].reconstruct(faiss_id)
        #    node_features.append(embedding)
        #except KeyError:
        #    print(f"Warning: Node '{node_id}' not found in '{node_type}' mapping. Using zero vector.")
        #    node_features.append(np.zeros(embedding_dim))
        #except Exception as e:
        #    print(f"An error occurred for node '{node_id}': {e}. Using zero vector.")
        #    node_features.append(np.zeros(embedding_dim))

        node_features.append(np.random.randn(embedding_dim))
            
    x = torch.tensor(np.array(node_features), dtype=torch.float)

    # Build edge_index for PyG
    source_nodes = []
    target_nodes = []
    for src, dst in subgraph_nx.edges():
        source_nodes.append(node_map[src])
        target_nodes.append(node_map[dst])
        source_nodes.append(node_map[dst])
        target_nodes.append(node_map[src])

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    return data, nodes
