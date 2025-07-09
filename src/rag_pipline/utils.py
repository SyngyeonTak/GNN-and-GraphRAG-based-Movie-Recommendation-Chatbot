import faiss
import json
import re
from thefuzz import process

def find_best_match(query, choices, score_cutoff=85):
    """가장 유사한 항목을 찾아 반환 (일정 점수 이상일 경우)"""
    try:
        # choices가 (이름, ID) 형태의 튜플 리스트일 수 있으므로 이름만 추출
        actual_choices = [choice[0] if isinstance(choice, tuple) else choice for choice in choices]
        best_match, score = process.extractOne(query, actual_choices)

        if score >= score_cutoff:
            return best_match
    except:
        pass
    return None

def load_recommendation_assets(base_path="./dataset/faiss"):
    """
    영화, 배우, 감독, 장르의 Faiss 인덱스와 매핑 파일을 모두 로드합니다.
    """
    print("Loading all recommendation assets...")
    try:
        assets = {}
        entities = ["movie", "actor", "director", "genre"]
        
        for entity in entities:
            index_path = f"{base_path}/{entity}_embeddings.faiss"
            mapping_path = f"{base_path}/{entity}_mapping.json"
            
            assets[entity] = {
                "index": faiss.read_index(index_path),
                "mapping": json.load(open(mapping_path, 'r', encoding='utf-8'))
            }
            print(f"✅ Loaded {entity} assets.")
            
        print("✅ All recommendation assets loaded successfully.")
        return assets
    except Exception as e:
        print(f"❌ Failed to load recommendation assets: {e}")
        return None

def clean_cypher_query(query: str) -> str:
    """
    LLM이 생성한 Cypher 쿼리에서 Markdown 코드 블록을 제거합니다.
    """
    if "```cypher" in query:
        query = query.split("```cypher", 1)[1]
    if "```" in query:
        query = query.split("```", 1)[0]
    return query.strip()