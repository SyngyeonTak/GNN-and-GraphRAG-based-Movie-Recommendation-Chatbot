# src/preprocess_data.py

import pandas as pd
import os
import re
import time
from tmdbv3api import TMDb, Movie, Search
from tqdm import tqdm # 진행률 표시를 위한 라이브러리
from pathlib import Path # 1. pathlib 라이브러리 추가
from dotenv import load_dotenv

# --- 파일 경로 정의 ---
RAW_DATA_PATH = './dataset/raw/ml-32m/' 
PROCESSED_DATA_PATH = './dataset/processed/'

def get_credits(title, year):
    # 제목과 연도를 함께 사용
    movie_api = Movie()
    search_api = Search()
    search_results = search_api.movies(term=title, year=year)
    
    movie_id = None
    try:
        if search_results:
            movie_id = search_results[0].id
            credits = movie_api.credits(movie_id=movie_id)
            director = next((member.name for member in credits.crew if member['job'] == 'Director'), None)
            actors = [member.name for member in list(credits.cast)[:5]]
            return director, actors
        else:
            return None, None
    except Exception as e:
        return None, None

def preprocess_data():
    """
    - movies.csv: 'title'에서 연도 분리, 'genres'를 리스트로 변환, TMDB에서 감독/배우 정보 추가
    - ratings.csv: 'timestamp'를 datetime으로 변환
    """
    # --- 데이터 불러오기 ---
    movies_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'movies.csv'))
    ratings_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'ratings.csv'))

    # --- movies.csv 전처리 ---
    print("기존 movies.csv 파일 전처리를 시작합니다...")
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    movies_df['title'] = movies_df.apply(lambda row: re.sub(r'\s*\(\d{4}\)$', '', row['title']).strip(), axis=1)
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))
    movies_df['genres'] = movies_df['genres'].apply(lambda x: [] if len(x) == 1 and x[0] == '(no genres listed)' else x)
    print("기존 movies.csv 파일 전처리 완료.")

    # --- TMDB API 설정 ---
    dotenv_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=dotenv_path)

    tmdb = TMDb()
    tmdb.api_key = os.environ.get('TMDB_API_KEY')
    print()
    tmdb.language = 'en-US'
    
    # --- 각 영화에 대해 감독 및 배우 정보 가져오기 ---
    print("\nTMDB API를 통해 감독 및 배우 정보 가져오기")

    directors = []
    actors_list = []
    
    #test_df = movies_df.head(50).copy()
    for _, row in tqdm(movies_df.iterrows(), total=movies_df.shape[0]):
        title = row['title']
        year = int(row['year']) if pd.notna(row['year']) else None
        
        director, actors = get_credits(title, year)
        directors.append(director)
        actors_list.append(actors)
        
        time.sleep(0.05)

    # --- 기존 데이터프레임에 새로운 열 추가 ---
    # 테스트한 1000개의 결과만 test_df에 추가합니다.
    movies_df['director'] = directors
    movies_df['actors'] = actors_list

    print("감독 및 배우 정보 추가 완료.")
    
    # 정보 수집 결과 요약
    null_director_count = movies_df['director'].isnull().sum()
    print(f"총 1000개 영화 중 감독 정보를 찾지 못한 영화: {null_director_count}개")

    # --- ratings.csv 전처리 (이 부분은 테스트와 무관하게 전체 처리) ---
    print("\nratings.csv 파일 전처리를 시작합니다...")
    ratings_df['rated_at'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    ratings_df = ratings_df.drop('timestamp', axis=1)
    print("ratings.csv 파일 전처리 완료.")

    # --- 전처리된 데이터 저장 ---
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
        print(f"'{PROCESSED_DATA_PATH}' 폴더 생성 완료.")

    movies_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'movies_processed.csv'), index=False)
    ratings_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'ratings_processed.csv'), index=False)

    print(f"\n모든 데이터 처리가 완료되었으며, '{PROCESSED_DATA_PATH}'에 저장되었습니다.")
    print("최종 movies 데이터 샘플 (테스트 결과):")
    print(movies_df.head())

if __name__ == "__main__":

    preprocess_data()