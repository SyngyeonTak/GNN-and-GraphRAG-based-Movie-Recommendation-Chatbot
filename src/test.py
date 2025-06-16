from pathlib import Path # 1. pathlib 라이브러리 추가
from dotenv import load_dotenv

import os
dotenv_path = Path(__file__).parent.parent / '.env'
    
# 3. 해당 경로의 .env 파일을 로드
load_dotenv(dotenv_path=dotenv_path)
api_key = os.environ.get('TMDB_API_KEY')

print(api_key)