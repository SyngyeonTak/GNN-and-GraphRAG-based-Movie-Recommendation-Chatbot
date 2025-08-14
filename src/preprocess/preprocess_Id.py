# src/reindex_ids_1based.py

import pandas as pd
import os
from pathlib import Path
import ast

RAW_PROCESSED_PATH = './dataset/processed/'
OUTPUT_PATH = './dataset/processed/'

def reindex_ids():
    # --- Load processed movie & rating data ---
    movies_file = os.path.join(RAW_PROCESSED_PATH, 'movies_processed.csv')
    ratings_file = os.path.join(RAW_PROCESSED_PATH, 'ratings_processed.csv')

    movies_df = pd.read_csv(movies_file)
    ratings_df = pd.read_csv(ratings_file)

    # Convert stringified lists back to list
    for col in ['genres', 'actors']:
        movies_df[col] = movies_df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

    # --- Filter movies according to Neo4j import condition ---
    filtered_movies = movies_df[
        movies_df['director'].notna() & movies_df['actors'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ].copy()

    # --- Reindex movieId sequentially (1-based) ---
    filtered_movies = filtered_movies.reset_index(drop=True)
    filtered_movies['movieId_old'] = filtered_movies['movieId']  # 기존 id 보관
    filtered_movies['movieId'] = filtered_movies.index + 1  # 1,2,3,... 순차 id 부여

    # --- Map old movieId -> new movieId ---
    movie_id_map = dict(zip(filtered_movies['movieId_old'], filtered_movies['movieId']))

    # --- Filter ratings for valid movies and update movieId ---
    ratings_filtered = ratings_df[ratings_df['movieId'].isin(movie_id_map.keys())].copy()
    ratings_filtered['movieId_old'] = ratings_filtered['movieId']
    ratings_filtered['movieId'] = ratings_filtered['movieId'].map(movie_id_map)

    # --- Reindex userId sequentially (1-based) ---
    user_ids = sorted(ratings_filtered['userId'].unique())
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_ids, start=1)}
    ratings_filtered['userId_old'] = ratings_filtered['userId']
    ratings_filtered['userId'] = ratings_filtered['userId'].map(user_id_map)

    # --- Save updated CSVs ---
    movies_outfile = os.path.join(OUTPUT_PATH, 'movies_processed_w_id.csv')
    ratings_outfile = os.path.join(OUTPUT_PATH, 'ratings_processed_w_id.csv')

    filtered_movies.to_csv(movies_outfile, index=False)
    ratings_filtered.to_csv(ratings_outfile, index=False)

    print(f"Reindexed movies saved to: {movies_outfile}")
    print(f"Reindexed ratings saved to: {ratings_outfile}")
    print(f"Number of movies retained: {len(filtered_movies)}")
    print(f"Number of ratings retained: {len(ratings_filtered)}")
    print(f"Number of unique users: {len(user_id_map)}")

if __name__ == "__main__":
    reindex_ids()
