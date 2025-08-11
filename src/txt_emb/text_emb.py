import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'


# 3. 텍스트 임베딩 모델 로드
print("Loading sentence transformer model...")
#model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 처리할 파일 목록
mapping_files = {
    "actor": "./dataset/faiss/actor_mapping.json",
    "director": "./dataset/faiss/director_mapping.json",
    "genre": "./dataset/faiss/genre_mapping.json",
    "movie": "./dataset/faiss/movie_mapping.json"
}

for entity_type, file_path in mapping_files.items():
    print(f"--- Processing {entity_type} ---")
    
    # 1. 매핑 파일 로드
    with open(file_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    # 2. 텍스트 목록 추출 (인덱스 순서 유지를 위해 정렬)
    # JSON 키가 문자열 "0", "1", "2" 이므로 정수 변환 후 정렬
    sorted_items = sorted(mapping_data.items(), key=lambda item: int(item[0]))
    names = [item[1] for item in sorted_items]

    # 4. 텍스트 임베딩 생성
    print(f"Generating text embeddings for {len(names)} {entity_type}s...")
    text_embeddings = model.encode(names, show_progress_bar=True)

    # 5. 새로운 FAISS 인덱스 구축
    dimension = text_embeddings.shape[1]  # 임베딩 벡터의 차원
    new_index = faiss.IndexFlatL2(dimension)
    new_index.add(np.array(text_embeddings, dtype=np.float32))

    # 6. 새로운 인덱스 파일 저장
    output_path = f"./dataset/faiss/{entity_type}_text_embeddings.faiss"
    faiss.write_index(new_index, output_path)
    
    print(f"Successfully created '{output_path}' with {new_index.ntotal} vectors.\n")
