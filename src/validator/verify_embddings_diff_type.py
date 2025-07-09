import faiss
import json
import numpy as np

# --- 설정 ---
# 검증하고 싶은 [소스 -> 타겟] 조합과 소스 노드의 이름
CROSS_VERIFICATION_TARGETS = [
    #{"source_type": "movie", "source_name": "Toy Story", "target_type": "actor"},
    #{"source_type": "actor", "source_name": "Tom Hanks", "target_type": "movie"},
    #{"source_type": "director", "source_name": "Steven Spielberg", "target_type": "genre"},
    {"source_type": "director", "source_name": "Steven Spielberg", "target_type": "movie"}
]
TOP_K = 10  # 몇 개의 유사 항목을 볼지 설정

def find_cross_type_similarity(source_type, source_name, target_type):
    """
    소스 노드의 임베딩을 사용하여 다른 타입의 타겟 노드와의 유사도를 계산합니다.
    (예: 영화 'Toy Story'와 가장 유사한 '배우' 찾기)
    """
    print(f"\n{'='*80}")
    print(f"🔬 Finding Similar Nodes in '{target_type.upper()}' for Source: '{source_name}' ({source_type.upper()})")
    print(f"{'='*80}")

    source_faiss_path = f"././dataset/faiss/{source_type}_embeddings.faiss"
    source_mapping_path = f"././dataset/faiss/{source_type}_mapping.json"
    target_faiss_path = f"././dataset/faiss/{target_type}_embeddings.faiss"
    target_mapping_path = f"././dataset/faiss/{target_type}_mapping.json"

    # --- 1. 소스(Source) 데이터 로드 ---
    try:
        print(f"--- Loading SOURCE index from: {source_faiss_path} ---")
        source_index = faiss.read_index(source_faiss_path)
        with open(source_mapping_path, 'r', encoding='utf-8') as f:
            source_idx_to_name = {int(k): v for k, v in json.load(f).items()}
        print(f"Source ('{source_type}') data loaded successfully.")
    except Exception as e:
        print(f"🔴 Error loading SOURCE data: {e}")
        return

    # --- 2. 타겟(Target) 데이터 로드 ---
    try:
        print(f"\n--- Loading TARGET index from: {target_faiss_path} ---")
        target_index = faiss.read_index(target_faiss_path)
        with open(target_mapping_path, 'r', encoding='utf-8') as f:
            target_idx_to_name = {int(k): v for k, v in json.load(f).items()}
        print(f"Target ('{target_type}') data loaded successfully.")
    except Exception as e:
        print(f"🔴 Error loading TARGET data: {e}")
        return

    # --- 3. 기본 구조 및 차원 검증 ---
    print("\n--- Verifying dimensions ---")
    if source_index.d != target_index.d:
        print(f"🔴 [CRITICAL ERROR] Vector dimensions do not match!")
        print(f"Source ('{source_type}') dimension: {source_index.d}")
        print(f"Target ('{target_type}') dimension: {target_index.d}")
        print("Similarity search between these types is not possible.")
        return
    else:
        print(f"[✔️ SANITY CHECK PASSED] Vector dimensions match: {source_index.d}")

    # --- 4. 소스 벡터 추출 및 교차 유사도 검색 ---
    print(f"\n--- Performing cross-similarity search ---")
    try:
        # 소스 이름으로 인덱스 찾기
        source_name_to_idx = {v: k for k, v in source_idx_to_name.items()}
        source_idx = source_name_to_idx[source_name]

        # 소스 인덱스의 벡터를 Faiss에서 가져오기
        source_vector = source_index.reconstruct(source_idx).reshape(1, -1)
        print(f"Successfully retrieved vector for '{source_name}'.")

        # 소스 벡터를 사용해 *타겟 인덱스*에서 검색 수행
        distances, indices = target_index.search(source_vector, TOP_K)

        print(f"\nTop {TOP_K} similar '{target_type.upper()}' items to '{source_name}':")
        print("-" * 60)
        for i, idx in enumerate(indices[0]):
            similar_name = target_idx_to_name[idx]
            similarity_score = distances[0][i]
            # L2 거리는 작을수록 유사함
            print(f"{i+1:2d}. {similar_name:<40} (Distance: {similarity_score:.4f})")

    except KeyError:
        print(f"\n[⚠️ WARNING] Source item '{source_name}' not found in the '{source_type}' mapping file.")
    except Exception as e:
        print(f"\n[🔴 ERROR] An error occurred during search: {e}")


if __name__ == "__main__":
    # 설정된 모든 조합에 대해 교차 타입 유사도 검증 실행
    for config in CROSS_VERIFICATION_TARGETS:
        find_cross_type_similarity(
            source_type=config["source_type"],
            source_name=config["source_name"],
            target_type=config["target_type"]
        )
