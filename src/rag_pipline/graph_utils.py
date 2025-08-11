import networkx as nx
from collections import Counter
import numpy as np
import torch
from torch_geometric.data import Data

def create_global_nx_graph(snapshot_data: dict):
    print("--- Creating global NetworkX graph, excluding all duplicate movies ---")
    G = nx.Graph()
    
    node_mappings = snapshot_data['node_mappings']
    node_names = snapshot_data.get('node_names', {})
    edge_indices = snapshot_data['edge_indices']

    internal_to_final_id_map = {}

    # --- 1. 비영화 노드(Actor, Director, Genre, User)는 그대로 추가 ---
    # (이전 코드와 동일)
    for node_type in ['User', 'Actor', 'Director', 'Genre']:
        if node_type in node_mappings:
            mapping = node_mappings[node_type]
            names_list = node_names.get(node_type, [f"user_{i}" for i in range(len(mapping))])
            
            for neo4j_id, node_idx in mapping.items():
                final_node_id = names_list[node_idx] if node_idx < len(names_list) else None
                if not final_node_id:
                    continue
                G.add_node(final_node_id, type=node_type)
                internal_to_final_id_map[(node_type, node_idx)] = final_node_id

    # --- 2. 영화 노드 처리 (중복 완전 제거 로직) ---

    # --- 2a. 모든 영화 제목을 순회하며 중복된 제목을 미리 찾아냄 ---
    movie_titles = node_names.get('Movie', [])
    title_counts = Counter(movie_titles)
    duplicate_titles = {title for title, count in title_counts.items() if count > 1}
    
    if duplicate_titles:
        print(f"Found {len(duplicate_titles)} duplicate titles. These will be excluded.")

    # --- 2b. 영화 노드 추가: 중복된 제목의 영화는 완전히 제외 ---
    movie_mapping = node_mappings.get('Movie', {})
    for neo4j_id, node_idx in movie_mapping.items():
        if node_idx < len(movie_titles):
            title = movie_titles[node_idx]
            
            # ✨ 핵심 변경점: 제목이 중복 리스트에 있으면 건너뜀
            if title in duplicate_titles:
                continue
            
            # 중복이 아닌 영화만 그래프에 추가하고 매핑 정보 저장
            G.add_node(title, type='Movie') # year 정보가 없으므로 속성에서 제외
            internal_to_final_id_map[('Movie', node_idx)] = title

    for edge_key, (src_nodes, dst_nodes) in edge_indices.items():
        src_type_str, _, dst_type_str = edge_key
        src_type = src_type_str.capitalize()
        dst_type = dst_type_str.capitalize()

        for src_idx, dst_idx in zip(src_nodes, dst_nodes):
            src_final_id = internal_to_final_id_map.get((src_type, src_idx))
            dst_final_id = internal_to_final_id_map.get((dst_type, dst_idx))

            if src_final_id and dst_final_id:
                G.add_edge(src_final_id, dst_final_id)

    print(f"Strictly de-duplicated graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def extract_subgraph_from_global(global_G, seed_movie_titles):
    nodes_for_subgraph = set()

    for movie_title in seed_movie_titles:
        if global_G.has_node(movie_title):
            nodes_for_subgraph.add(movie_title)
            nodes_for_subgraph.update(global_G.neighbors(movie_title))
        else:
            print(f"Warning: Seed movie '{movie_title}' not found in the global graph. Skipping.")

    # 추출된 노드가 하나라도 있을 경우에만 서브그래프 생성
    if not nodes_for_subgraph:
        print("No valid seed nodes found to create a subgraph.")
        # 비어있는 그래프를 반환하거나 None을 반환할 수 있습니다.
        return nx.Graph() 

    subgraph_nx = global_G.subgraph(nodes_for_subgraph).copy()
    
    return subgraph_nx

def convert_nx_to_pyg(subgraph_nx, assets):
    nodes = list(subgraph_nx.nodes())
    node_map = {node_id: i for i, node_id in enumerate(nodes)}

    # --- 1. 사전 준비: 이름 -> FAISS ID 역매핑 생성 ---
    # 루프 안에서 매번 생성하지 않도록, 미리 모든 타입의 역매핑을 만들어 둡니다.
    reverse_mappings = {}
    for entity_type, asset_data in assets.items():
        # mapping: {"faiss_id": "name/title"}
        # reverse_mapping: {"name/title": faiss_id}
        reverse_mappings[entity_type] = {v: int(k) for k, v in asset_data['mapping'].items()}

    # --- 2. 노드 특징(feature) 행렬 'x' 생성 ---
    node_features = []
    embedding_dim = assets['movie']['gnn_index'].d # 임베딩 차원은 FAISS 인덱스에서 가져옴

    # 하나의 루프로 통합하여 모든 노드를 처리합니다.
    for node_id in nodes:
        try:
            # a. 노드 타입과 이름(ID)을 가져옵니다.
            node_type = subgraph_nx.nodes[node_id].get('type').lower()

            # b. 미리 만들어둔 역매핑에서 FAISS ID를 찾습니다.
            faiss_id = reverse_mappings[node_type][node_id]

            # c. FAISS 인덱스에서 해당 ID의 임베딩 벡터를 재구성(가져오기)합니다.
            embedding = assets[node_type]['gnn_index'].reconstruct(faiss_id)
            node_features.append(embedding)

        except KeyError:
            # 매핑에 해당 노드 ID가 없는 예외 상황 처리
            print(f"Warning: Node '{node_id}' not found in '{node_type}' mapping. Using zero vector.")
            node_features.append(np.zeros(embedding_dim))
        except Exception as e:
            # 기타 예외 상황 처리
            print(f"An error occurred for node '{node_id}': {e}. Using zero vector.")
            node_features.append(np.zeros(embedding_dim))
            
    # NumPy 리스트를 PyTorch 텐서로 변환
    x = torch.tensor(np.array(node_features), dtype=torch.float)

    # --- 3. 엣지 인덱스 'edge_index' 생성 --- (이전과 동일)
    source_nodes = []
    target_nodes = []
    for src, dst in subgraph_nx.edges():
        source_nodes.append(node_map[src])
        target_nodes.append(node_map[dst])
        source_nodes.append(node_map[dst])
        target_nodes.append(node_map[src])

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    # --- 4. 최종 PyG 데이터 객체 생성 ---
    data = Data(x=x, edge_index=edge_index)
    
    return data, nodes