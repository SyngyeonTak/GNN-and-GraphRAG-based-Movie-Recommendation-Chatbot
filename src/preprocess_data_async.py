# src/preprocess_data.py

import pandas as pd
import os
import re
import asyncio
import aiohttp
from tqdm.asyncio import tqdm # 비동기용 tqdm
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import quote # URL 인코딩을 위한 라이브러리

# .env 파일 로드
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# TMDB API 키 및 헤더
API_KEY = os.environ.get('TMDB_API_KEY')

# 파일 경로 정의
RAW_DATA_PATH = './dataset/raw/ml-32m/'
PROCESSED_DATA_PATH = './dataset/processed/'

# 동시에 보낼 API 요청 수
CONCURRENT_REQUESTS = 100 

# --- 비동기 API 호출 함수들 ---

async def fetch_credits(session, movie_id):
    """주어진 영화 ID로 배우와 감독 정보를 비동기로 가져옵니다."""
    if not movie_id:
        return None, None
    #url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?language=en-US"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?language=en-US&api_key={API_KEY}"
    try:
        async with session.get(url, ssl=False) as response:
            if response.status != 200:
                return None, None # 요청 실패
            data = await response.json()
            director = next((member['name'] for member in data.get('crew', []) if member.get('job') == 'Director'), None)
            actors = [member['name'] for member in data.get('cast', [])[:5]]
            return director, actors
    except Exception:
        return None, None

async def get_movie_data(session, movie_row):
    """DataFrame의 한 행(영화 정보)을 받아 전체 API 처리 과정을 비동기로 수행합니다."""
    title_encoded = quote(movie_row['title'])
    year = movie_row['year']
    
    # 연도 정보가 없으면 API 요청 없이 None 반환
    if pd.isna(year):
        return movie_row['movieId'], None, None

    #search_url = f"https://api.themoviedb.org/3/search/movie?query={title_encoded}&year={int(year)}&language=en-US"
    search_url = f"https://api.themoviedb.org/3/search/movie?query={title_encoded}&year={int(year)}&language=en-US&api_key={API_KEY}"
    try:
        # 1. 영화 검색 API 호출
        async with session.get(search_url, ssl=False) as response:
            if response.status != 200:
                return movie_row['movieId'], None, None
            
            data = await response.json()
            results = data.get('results')
            
            # 2. 검색 결과가 있으면, 첫 번째 결과의 ID로 크레딧 정보 요청
            if results:
                movie_id = results[0].get('id')
                director, actors = await fetch_credits(session, movie_id)
                return movie_row['movieId'], director, actors
            else:
                return movie_row['movieId'], None, None
    except Exception:
        return movie_row['movieId'], None, None

async def main():
    """메인 비동기 실행 함수"""
    print("데이터 전처리를 시작합니다...")
    
    # --- API 키 유효성 검사 ---
    if not API_KEY:
        raise ValueError("TMDB_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
    # --- 데이터 불러오기 ---
    movies_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'movies.csv'))
    ratings_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'ratings.csv'))

    # --- movies.csv 기본 전처리 ---
    print("movies.csv 파일 기본 전처리를 수행합니다...")
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    movies_df['title'] = movies_df.apply(lambda row: re.sub(r'\s*\(\d{4}\)$', '', row['title']).strip(), axis=1)
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))
    movies_df['genres'] = movies_df['genres'].apply(lambda x: [] if len(x) == 1 and x[0] == '(no genres listed)' else x)

    # --- 중간 저장 및 이어하기 로직 ---
    partial_file_path = os.path.join(PROCESSED_DATA_PATH, 'movies_processed_partial.csv')
    if os.path.exists(partial_file_path):
        print(f"기존에 작업하던 파일 '{partial_file_path}'을(를) 발견했습니다. 이어서 작업을 시작합니다.")
        processed_df = pd.read_csv(partial_file_path)
        processed_ids = set(processed_df['movieId'])
    else:
        processed_df = pd.DataFrame()
        processed_ids = set()

    to_process_df = movies_df[~movies_df['movieId'].isin(processed_ids)]

    if to_process_df.empty:
        print("모든 영화의 정보 수집이 이미 완료되었습니다.")
    else:
        print(f"총 {len(movies_df)}개 영화 중 {len(to_process_df)}개의 영화에 대한 정보 수집을 시작합니다.")
        
        # --- 비동기 작업 실행 ---
        async with aiohttp.ClientSession() as session:
            # 처리할 영화 목록을 CONCURRENT_REQUESTS 개수만큼의 배치로 나눔
            for i in tqdm(range(0, len(to_process_df), CONCURRENT_REQUESTS), desc="Processing Batches"):
                batch = to_process_df.iloc[i:i + CONCURRENT_REQUESTS]
                tasks = [get_movie_data(session, row) for _, row in batch.iterrows()]
                results = await asyncio.gather(*tasks)
                
                # --- 결과 정리 및 저장 ---
                new_data = []
                for result in results:
                    if result:
                        movieId, director, actors = result
                        new_data.append({'movieId': movieId, 'director': director, 'actors': actors})
                
                if new_data:
                    new_df = pd.DataFrame(new_data)
                    # 기존 결과와 새로운 결과를 합친 후, partial 파일에 덮어쓰기
                    processed_df = pd.concat([processed_df, new_df], ignore_index=True)
                    processed_df.to_csv(partial_file_path, index=False)
    
    # --- 최종 데이터 병합 및 저장 ---
    print("모든 정보 수집 완료. 최종 파일을 생성합니다.")
    # 원본 데이터에 수집한 director, actors 정보를 movieId 기준으로 병합
    final_movies_df = pd.merge(movies_df, processed_df, on='movieId', how='left')

    # --- ratings.csv 전처리 ---
    print("ratings.csv 파일 전처리를 시작합니다...")
    ratings_df['rated_at'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    ratings_df = ratings_df.drop('timestamp', axis=1)
    
    # --- 최종 파일 저장 ---
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
        
    final_movies_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'movies_processed.csv'), index=False)
    ratings_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'ratings_processed.csv'), index=False)

    print(f"\n모든 데이터 처리가 완료되었으며, '{PROCESSED_DATA_PATH}'에 저장되었습니다.")
    print("최종 movies 데이터 샘플:")
    print(final_movies_df.head())


if __name__ == "__main__":
    # Windows에서 asyncio 실행 시 발생할 수 있는 에러 방지
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())