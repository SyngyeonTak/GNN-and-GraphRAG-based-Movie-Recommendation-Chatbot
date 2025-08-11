# src/preprocess_data.py

import pandas as pd
import os
import re
import asyncio
import aiohttp
from tqdm.asyncio import tqdm # tqdm for asyncio
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import quote # Library for URL encoding

# Load .env file
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# TMDB API Key
API_KEY = os.environ.get('TMDB_API_KEY')

# Define file paths
RAW_DATA_PATH = './dataset/raw/ml-1m_v2/'
PROCESSED_DATA_PATH = './dataset/processed/'

# Number of concurrent API requests
CONCURRENT_REQUESTS = 100 

# --- Asynchronous API Call Functions ---

async def fetch_movie_details(session, movie_id: int):
    """
    Given a movie ID, asynchronously fetches its details (overview, director, actors).
    Uses 'append_to_response' for efficient API calling.

    Args:
        session (aiohttp.ClientSession): The aiohttp client session.
        movie_id (int): The ID of the movie to fetch.

    Returns:
        dict: A dictionary containing the overview, director, and actors.
              Returns None if the request fails.
    """
    if not movie_id:
        return None
        
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US&append_to_response=credits"
    
    try:
        async with session.get(url, ssl=False) as response:
            if response.status != 200:
                return None  # Request failed
            
            data = await response.json()

            # 1. Extract overview
            overview = data.get('overview', None)
            
            # 2. Extract director and actors from within the 'credits' object
            credits_data = data.get('credits', {})
            director = next((member['name'] for member in credits_data.get('crew', []) if member.get('job') == 'Director'), None)
            actors = [member['name'] for member in credits_data.get('cast', [])[:5]]
            
            # 3. Return as a dictionary
            return {
                'overview': overview,
                'director': director,
                'actors': actors
            }
            
    except Exception:
        return None

async def get_movie_data(session, movie_row):
    """
    Takes a DataFrame row (movie info) and performs the entire API process asynchronously.
    """
    # Use the cleaned title, as the original might contain the year
    title_only = re.sub(r'\s*\(\d{4}\)$', '', movie_row['title']).strip()
    title_encoded = quote(title_only)
    year = movie_row['year']
    
    if pd.isna(year):
        return movie_row['movieId'], None

    search_url = f"https://api.themoviedb.org/3/search/movie?query={title_encoded}&primary_release_year={int(year)}&language=en-US&api_key={API_KEY}"
    
    try:
        # 1. Call the movie search API
        async with session.get(search_url, ssl=False) as response:
            if response.status != 200:
                return movie_row['movieId'], None
            
            data = await response.json()
            results = data.get('results')
            
            # 2. If search results exist, request details for the first result
            if results:
                movie_id = results[0].get('id')
                details = await fetch_movie_details(session, movie_id)
                return movie_row['movieId'], details
            else:
                return movie_row['movieId'], None
    except Exception:
        return movie_row['movieId'], None

async def main():
    """Main asynchronous execution function"""
    print("Starting data preprocessing...")
    
    # --- Validate API Key ---
    if not API_KEY:
        raise ValueError("TMDB_API_KEY is not set in the .env file.")
        
    # --- Load data ---
    movies_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'movies.csv'))
    ratings_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'ratings.csv'))

    # --- Basic preprocessing on movies.csv ---
    print("Performing basic preprocessing on movies.csv...")
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))
    movies_df['genres'] = movies_df['genres'].apply(lambda x: [] if len(x) == 1 and x[0] == '(no genres listed)' else x)

    # --- Logic for partial save and resume ---
    partial_file_path = os.path.join(PROCESSED_DATA_PATH, 'movies_processed_partial.csv')
    if os.path.exists(partial_file_path):
        print(f"Found an existing partial file '{partial_file_path}'. Resuming work.")
        processed_df = pd.read_csv(partial_file_path)
        processed_ids = set(processed_df['movieId'].astype(int))
    else:
        processed_df = pd.DataFrame()
        processed_ids = set()

    to_process_df = movies_df[~movies_df['movieId'].isin(processed_ids)]

    if to_process_df.empty:
        print("Information collection for all movies is already complete.")
    else:
        print(f"Starting to collect information for {len(to_process_df)} out of {len(movies_df)} movies.")
        
        # --- Execute asynchronous tasks ---
        async with aiohttp.ClientSession() as session:
            for i in tqdm(range(0, len(to_process_df), CONCURRENT_REQUESTS), desc="Processing Batches"):
                batch = to_process_df.iloc[i:i + CONCURRENT_REQUESTS]
                tasks = [get_movie_data(session, row) for _, row in batch.iterrows()]
                results = await asyncio.gather(*tasks)
                
                # --- Organize and save results ---
                new_data = []
                for result in results:
                    if result:
                        # Process results according to the new return structure
                        movieId, details = result
                        if details:
                            # Combine movieId and details using dictionary unpacking
                            new_row = {'movieId': movieId, **details}
                            new_data.append(new_row)
                        else:
                            # If the API lookup failed, record only the movieId to prevent re-attempts
                            new_data.append({'movieId': movieId, 'overview': None, 'director': None, 'actors': None})
                
                if new_data:
                    new_df = pd.DataFrame(new_data)
                    processed_df = pd.concat([processed_df, new_df], ignore_index=True)
                    processed_df.to_csv(partial_file_path, index=False)
    
    # --- Final data merging and saving ---
    print("All information collected. Creating final files.")
    # Remove year from the title in the original movies_df
    movies_df['title'] = movies_df['title'].apply(lambda row: re.sub(r'\s*\(\d{4}\)$', '', str(row)).strip())

    # Merge the collected information (overview, director, actors)
    final_movies_df = pd.merge(movies_df, processed_df[['movieId', 'overview', 'director', 'actors']], on='movieId', how='left')

    # --- Preprocess ratings.csv ---
    print("Preprocessing ratings.csv...")
    ratings_df['rated_at'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    ratings_df = ratings_df.drop('timestamp', axis=1)
    
    # --- Save final files ---
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        
    final_movies_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'movies_processed.csv'), index=False)
    ratings_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'ratings_processed.csv'), index=False)

    print(f"\nAll data processing is complete and saved to '{PROCESSED_DATA_PATH}'.")
    print("Final movies data sample:")
    print(final_movies_df.head())


if __name__ == "__main__":
    # Prevents errors that can occur when running asyncio on Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())