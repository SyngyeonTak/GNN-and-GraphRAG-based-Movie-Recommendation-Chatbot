import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# -----------------------
# 1. 데이터 불러오기
# -----------------------
movies = pd.read_csv("./dataset/processed/movies_filtered_decade.csv")

# overview 중 결측치 제거
movies = movies.dropna(subset=["overview"]).reset_index(drop=True)

# 전체 데이터 사용
sample_movies = movies[:1000]

print("데이터 개수:", len(sample_movies))
print(sample_movies[["movieId", "title", "overview"]].head(5))

# -----------------------
# 2. Sentence-BERT 모델 로드
# -----------------------
# 가볍게 테스트: all-MiniLM-L6-v2
# 검색 특화: multi-qa-mpnet-base-dot-v1
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------
# 3. 영화 overview 임베딩
# -----------------------
docs = sample_movies["overview"].tolist()
doc_emb = model.encode(docs, convert_to_tensor=True, show_progress_bar=True)

# -----------------------
# 4. 쿼리 → 임베딩
# -----------------------
query = "heartwarming WW1 movie"
query_emb = model.encode(query, convert_to_tensor=True)

# -----------------------
# 5. 유사도 계산 (cosine similarity)
# -----------------------
scores = util.pytorch_cos_sim(query_emb, doc_emb)[0].cpu().numpy()

# -----------------------
# 6. 결과 출력 (title + overview + score)
# -----------------------
top_n = 10
top_idx = scores.argsort()[::-1][:top_n]

print(f"\nQuery: {query}\n")
for rank, i in enumerate(top_idx, start=1):
    title = movies.iloc[i]["title"]
    overview = movies.iloc[i]["overview"]
    print(f"{rank}. {scores[i]:.2f} -> {title}")
    print(f"   {overview[:300]}...\n")  # 너무 길면 앞 300자만
