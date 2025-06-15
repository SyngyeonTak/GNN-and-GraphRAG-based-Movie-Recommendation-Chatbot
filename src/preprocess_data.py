# src/preprocess_data.py

import pandas as pd
import os
import re

# --- 파일 경로 정의 ---
RAW_DATA_PATH = './dataset/raw/ml-32m/'
PROCESSED_DATA_PATH = './dataset/processed/'

def preprocess_data():
    """
    - movies.csv: 'title'에서 연도 분리, 'genres'를 리스트로 변환
    - ratings.csv: 'timestamp'를 datetime으로 변환 (선택적)
    """

    # --- 데이터 불러오기 ---
    movies_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'movies.csv'))
    ratings_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'ratings.csv'))


    # --- movies.csv 전처리 ---
    # 1. 제목(title)에서 연도(year) 분리
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    movies_df['title'] = movies_df.apply(lambda row: re.sub(r'\s*\(\d{4}\)$', '', row['title']).strip(), axis=1)

    # 2. 장르(genres)를 리스트로 변환
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))

    # 3. (no genres listed) 처리
    movies_df['genres'] = movies_df['genres'].apply(lambda x: [] if len(x) == 1 and x[0] == '(no genres listed)' else x)

    # 4. ratings.csv 전처리 - rating timestamp 전처리 ---
    ratings_df['rated_at'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    ratings_df = ratings_df.drop('timestamp', axis=1)

    # 5. 전처리된 데이터 저장 ---
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
        print(f"'{PROCESSED_DATA_PATH}' 폴더 생성 완료.")

    movies_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'movies_processed.csv'), index=False)
    ratings_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'ratings_processed.csv'), index=False)

    print("data processed completed.")


if __name__ == "__main__":
    preprocess_data()