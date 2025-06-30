import faiss
import json
import numpy as np

# --- 설정 ---
FAISS_INDEX_PATH = "movie_embeddings.faiss"
MAPPING_PATH = "faiss_to_movie_title.json"
TOP_K = 10  # 몇 개의 유사 영화를 볼지 설정

# 검증해보고 싶은 영화 제목
TARGET_MOVIE_TITLE = "Forrest Gump"

# --- 검증 스크립트 ---
def verify_embeddings():
    print(f"--- Loading FAISS index from: {FAISS_INDEX_PATH} ---")
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"Index loaded successfully.")
        print(f"Number of vectors in index: {index.ntotal}")
        print(f"Vector dimension: {index.d}")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return

    print(f"\n--- Loading mapping from: {MAPPING_PATH} ---")
    try:
        with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
            idx_to_title = json.load(f)
            # JSON은 키를 문자열로 저장하므로, 다시 정수형으로 변환
            idx_to_title = {int(k): v for k, v in idx_to_title.items()}
        print(f"Mapping loaded successfully. Total movies: {len(idx_to_title)}")
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return

    # --- 기본 구조 검증 ---
    if index.ntotal != len(idx_to_title):
        print("\n[🔴 CRITICAL ERROR] The number of vectors in FAISS and the number of movies in the mapping do not match!")
    else:
        print("\n[✔️ SANITY CHECK PASSED] Index and mapping counts match.")


    # --- 유사도 검색 테스트 ---
    print(f"\n--- Performing similarity search for: '{TARGET_MOVIE_TITLE}' ---")
    
    try:
        # 영화 제목으로 인덱스 찾기
        title_to_idx = {v: k for k, v in idx_to_title.items()}
        target_idx = title_to_idx[TARGET_MOVIE_TITLE]
        
        # 해당 인덱스의 벡터를 Faiss에서 가져오기
        target_vector = index.reconstruct(target_idx).reshape(1, -1)
        
        # 유사도 검색 수행 (D: 거리, I: 인덱스)
        distances, indices = index.search(target_vector, TOP_K)
        
        print(f"\nTop {TOP_K} similar movies to '{TARGET_MOVIE_TITLE}':")
        print("-" * 50)
        for i, idx in enumerate(indices[0]):
            similar_movie_title = idx_to_title[idx]
            similarity_score = distances[0][i]
            # L2 거리는 작을수록 유사하므로, (1 / (1 + distance)) 로 유사도 점수를 변환해 볼 수 있음
            print(f"{i+1:2d}. {similar_movie_title:<40} (Distance: {similarity_score:.4f})")
        
    except KeyError:
        print(f"\n[Error] Movie '{TARGET_MOVIE_TITLE}' not found in the mapping file.")
    except Exception as e:
        print(f"\nAn error occurred during search: {e}")


if __name__ == "__main__":
    verify_embeddings()