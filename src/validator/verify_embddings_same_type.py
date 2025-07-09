import faiss
import json
import numpy as np

# --- 설정 ---
# 검증하고 싶은 노드 타입과 각 타입별 예시 타겟
VERIFICATION_TARGETS = {
    "movie": "Toy Story",
    "actor": "Tom Hanks",
    "director": "Steven Spielberg",
    "genre": "Crime"
}
TOP_K = 10  # 몇 개의 유사 항목을 볼지 설정

def verify_single_embedding(node_type, target_name):
    """지정된 노드 타입의 임베딩 파일과 매핑 파일을 검증합니다."""
    
    faiss_index_path = f"./dataset/faiss/{node_type}_embeddings.faiss"
    mapping_path = f"./dataset/faiss/{node_type}_mapping.json"
    
    print(f"\n{'='*60}")
    print(f"🔬 Verifying Embeddings for Node Type: '{node_type.upper()}'")
    print(f"{'='*60}")

    # --- 1. FAISS 인덱스 로드 ---
    print(f"--- Loading FAISS index from: {faiss_index_path} ---")
    try:
        index = faiss.read_index(faiss_index_path)
        print(f"Index loaded successfully.")
        print(f"Number of vectors in index: {index.ntotal}")
        print(f"Vector dimension: {index.d}")
    except Exception as e:
        print(f"🔴 Error loading FAISS index: {e}")
        return

    # --- 2. 매핑 파일 로드 ---
    print(f"\n--- Loading mapping from: {mapping_path} ---")
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            idx_to_name = json.load(f)
            # JSON은 키를 문자열로 저장하므로, 다시 정수형으로 변환
            idx_to_name = {int(k): v for k, v in idx_to_name.items()}
        print(f"Mapping loaded successfully. Total items: {len(idx_to_name)}")
    except Exception as e:
        print(f"🔴 Error loading mapping file: {e}")
        return

    # --- 3. 기본 구조 검증 ---
    if index.ntotal != len(idx_to_name):
        print("\n[🔴 CRITICAL ERROR] The number of vectors in FAISS and items in the mapping do not match!")
    else:
        print("\n[✔️ SANITY CHECK PASSED] Index and mapping counts match.")

    # --- 4. 유사도 검색 테스트 ---
    print(f"\n--- Performing similarity search for: '{target_name}' ---")
    
    try:
        # 이름으로 인덱스 찾기
        name_to_idx = {v: k for k, v in idx_to_name.items()}
        target_idx = name_to_idx[target_name]
        
        # 해당 인덱스의 벡터를 Faiss에서 가져오기
        target_vector = index.reconstruct(target_idx).reshape(1, -1)
        
        # 유사도 검색 수행 (D: 거리, I: 인덱스)
        distances, indices = index.search(target_vector, TOP_K)
        
        print(f"\nTop {TOP_K} similar items to '{target_name}':")
        print("-" * 50)
        for i, idx in enumerate(indices[0]):
            similar_name = idx_to_name[idx]
            similarity_score = distances[0][i]
            # L2 거리는 작을수록 유사함
            print(f"{i+1:2d}. {similar_name:<40} (Distance: {similarity_score:.4f})")
            
    except KeyError:
        print(f"\n[⚠️ WARNING] Item '{target_name}' not found in the mapping file.")
    except Exception as e:
        print(f"\n[🔴 ERROR] An error occurred during search: {e}")

if __name__ == "__main__":
    # 설정된 모든 타겟에 대해 검증 루프 실행
    for node_type, target_name in VERIFICATION_TARGETS.items():
        verify_single_embedding(node_type, target_name)
