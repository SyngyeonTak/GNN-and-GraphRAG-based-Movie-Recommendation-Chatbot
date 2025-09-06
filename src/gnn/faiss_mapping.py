import pickle
import json
import os

DUMP_PATH = './dataset/faiss/'
SNAPSHOT_PATH = './dataset/graph_snapshot.pkl'

def load_node_names(snapshot_path=SNAPSHOT_PATH):
    """graph_snapshot.pkl에서 node_names만 불러오기"""
    with open(snapshot_path, 'rb') as f:
        graph_data_from_pickle = pickle.load(f)
    return graph_data_from_pickle['node_names']

def save_mapping(node_type, node_names):
    """리스트 형태의 node_names를 JSON 매핑으로 저장"""
    os.makedirs(DUMP_PATH, exist_ok=True)
    idx_to_name = {i: name for i, name in enumerate(node_names)}

    mapping_path = f"{DUMP_PATH}{node_type}_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(idx_to_name, f, ensure_ascii=False, indent=4)
    print(f"Index-to-name mapping saved to: {mapping_path}")


if __name__ == "__main__":
    node_names = load_node_names()


    for node_type_capitalized, names in node_names.items():
        node_type = node_type_capitalized.lower()
        save_mapping(node_type, names)