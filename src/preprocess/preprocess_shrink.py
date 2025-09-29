import os
import pandas as pd
import ast

RAW_PROCESSED_PATH = './dataset/processed/'
OUTPUT_PATH = './dataset/processed/'

# -----------------------
# 1. Load Data
# -----------------------
ratings = pd.read_csv(RAW_PROCESSED_PATH + "ratings_processed.csv")
movies = pd.read_csv(RAW_PROCESSED_PATH + "movies_processed.csv")

# -----------------------
# 2. 1차 필터링 (감독 + 배우 조건 충족 영화만)
# -----------------------
print("Filtering movies with both director & actors...")
movies = movies.where(pd.notna(movies), None)

def has_valid_actors(x):
    if x is None:
        return False
    try:
        parsed = ast.literal_eval(x) if isinstance(x, str) else x
        return isinstance(parsed, list) and len(parsed) > 0
    except Exception:
        return False

valid_movies = movies[
    movies["director"].notna() &
    movies["actors"].apply(has_valid_actors)
]
valid_movie_ids = set(valid_movies["movieId"].unique())

ratings = ratings[ratings["movieId"].isin(valid_movie_ids)]
movies = movies[movies["movieId"].isin(valid_movie_ids)]

print(f"After filtering: {len(movies)} movies, {len(ratings)} ratings")

# -----------------------
# 3. 인기 영화 Top-3000 선정
# -----------------------
print("Selecting top-3000 by decade...")

movies["decade"] = (movies["year"] // 10) * 10
movie_popularity = ratings.groupby("movieId").size().reset_index(name="count")
movies_popularity = movies.merge(movie_popularity, on="movieId", how="left")

decades = sorted(movies_popularity["decade"].dropna().unique())
quota = 3000 // len(decades)

selected_ids = []
for decade in decades:
    top_in_decade = (
        movies_popularity[movies_popularity["decade"] == decade]
        .nlargest(quota, "count")["movieId"]
        .tolist()
    )
    selected_ids.extend(top_in_decade)

if len(selected_ids) < 3000:
    remaining = (
        movies_popularity[~movies_popularity["movieId"].isin(selected_ids)]
        .nlargest(3000 - len(selected_ids), "count")["movieId"]
        .tolist()
    )
    selected_ids.extend(remaining)

top_movies = selected_ids[:3000]

ratings_filtered = ratings[ratings["movieId"].isin(top_movies)]
movies_filtered = movies[movies["movieId"].isin(top_movies)]

print(f"After top-3000 by decade: {len(movies_filtered)} movies, {len(ratings_filtered)} ratings")

# -----------------------
# 4. Ratings 샘플링 (200K, stratified)
# -----------------------
target_size = 200_000
movie_weights = ratings_filtered["movieId"].value_counts(normalize=True)
sample_sizes = (movie_weights * target_size).astype(int)

sampled_ratings = []
for movie_id, n in sample_sizes.items():
    movie_ratings = ratings_filtered[ratings_filtered["movieId"] == movie_id]
    if n > 0:
        sampled_ratings.append(movie_ratings.sample(n=n, random_state=42))
sampled_ratings = pd.concat(sampled_ratings)

print(f"Sampled ratings: {len(sampled_ratings)} rows (target={target_size})")


# -----------------------
# 5. ID 재매핑 (movieId, rating_id) - 1부터 시작
# -----------------------
movie_id_map = {old: new+1 for new, old in enumerate(movies_filtered["movieId"].unique())}
movies_filtered["movieId"] = movies_filtered["movieId"].map(movie_id_map)
sampled_ratings["movieId"] = sampled_ratings["movieId"].map(movie_id_map)

sampled_ratings = sampled_ratings.reset_index(drop=True)
sampled_ratings.insert(0, "rating_id", sampled_ratings.index + 1)

# -----------------------
# 6. Save 결과 저장
# -----------------------
movies_filtered.to_csv(OUTPUT_PATH + "movies_filtered_decade.csv", index=False)
sampled_ratings.to_csv(OUTPUT_PATH + "ratings_filtered_decade.csv", index=False)

print("movies_filtered_decade.csv / ratings_filtered.csv 저장 완료 (ID 1부터 시작)")
