import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# ======================
# 1. Neo4j 연결 설정
# ======================
load_dotenv()
AURA_URI = os.environ.get("NEO4J_URI")
AURA_USER = os.environ.get("NEO4J_USER")
AURA_PASS = os.environ.get("NEO4J_PASSWORD")

driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASS))

# ======================
# 2. mapping.json 로드
# ======================
with open("dataset/faiss/movie_mapping.json", "r", encoding="utf-8") as f:
    movie_map = json.load(f)  # {"0": "Inception", "1": "Titanic", ...}

# ======================
# 3. SentenceTransformer 초기화
# ======================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ======================
# 4. Neo4j에서 overview 가져오기 함수
# ======================
def get_overview(tx, movie_name):
    query = """
    MATCH (m:Movie {title: $movie_name})
    RETURN m.overview AS overview
    """
    result = tx.run(query, movie_name=movie_name)
    record = result.single()
    return record["overview"] if record else None

# ======================
# 5. 개별 영화 overview 임베딩 생성
# ======================
start_idx = 0  # ← 디버깅 시작 인덱스
embeddings = []
missing_movies = []

with driver.session() as session:
    for i, (idx, movie_name) in enumerate(tqdm(movie_map.items(), desc="Processing movies")):
        if i < start_idx:
            continue  # 2600 이전은 skip

        print(f"[DEBUG] Processing index {i} → {movie_name}")
        overview = session.read_transaction(get_overview, movie_name)

        # NaN 또는 float 방지
        if not isinstance(overview, str):
            print(f"[WARNING] Overview is not a string at index {i}: {overview}")
            overview = ""

        try:
            emb = model.encode(overview, normalize_embeddings=True)
            embeddings.append(emb)
        except Exception as e:
            print(f"[ERROR] Encoding failed for index {i} ({movie_name}): {e}")
            embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
            missing_movies.append(movie_name)

embeddings = np.array(embeddings, dtype="float32")

# ======================
# 6. FAISS 인덱스 생성
# ======================
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)

# ======================
# 7. FAISS 및 로그 저장
# ======================
faiss.write_index(index, "dataset/faiss/movie_overview_embeddings.faiss")

with open("dataset/faiss/missing_movies.json", "w", encoding="utf-8") as f:
    json.dump(missing_movies, f, ensure_ascii=False, indent=2)

print(f"✅ Saved FAISS index with {len(embeddings)} embeddings")
print(f"⚠️ Missing overviews: {len(missing_movies)} (saved to missing_movies.json)")
