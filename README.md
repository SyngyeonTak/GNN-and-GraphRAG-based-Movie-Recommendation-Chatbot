# 🎬 GNN & GraphRAG 기반 영화 추천 챗봇

> Neo4j (GraphRAG) + GNN + LLM 기반으로 사실 기반 검색과 개인화 추천을 결합한 영화 추천 챗봇

---

## 🚀 프로젝트 로드맵

### **Phase 1: 기초 & 데이터 (지식 그래프 + GNN)**
- **데이터 준비**
  - ✅ MovieLens 32M 데이터셋 (movies, ratings)  
  - ✅ TMDb API 연동으로 배우/감독 메타데이터 수집
  - ✅ 배우·감독 정보가 존재하는 영화 중 **평점 수 기준 상위 3,000편**을 선별  
  - ✅ User rating은 해당 3,000편을 기준으로 약 **100만 개(1M)** 샘플
  - ✅ 데이터 정제  

- **지식 그래프 구축 (Neo4j)**
  - ✅ 노드: `Movie`, `User`, `Genre`, `Actor`, `Director`  
  - ✅ 관계: `RATED`, `HAS_GENRE`, `ACTED_IN`, `DIRECTED`  
  - ✅ Neo4j Browser에서 Cypher 쿼리 검증 완료  

- **GNN 추천 모델 학습**
  - ✅ 그래프 데이터를 PyTorch Geometric (PyG) 객체로 변환  
  - ✅ **Heterogeneous Graph Attention Network (HGAT)** 구현  
  - ✅ 학습 파이프라인 구축 및 손실 수렴 확인   
  - ✅ **FAISS**에 노드 임베딩 저장 (유사도 검색)  
  - ✅ `genre`, `actor`, `director`, `user` 임베딩 추가 (LLM preference 반영)  
  - ✅ 노드 타입별 출력 레이어 공유 (임베딩 공간 정렬)  
  ```mermaid
  flowchart TD
    A["Embedding Layer (128-d)"] --> B1["GATConv Layer 1 (multi-head)"]
    B1 --> B2["GATConv Layer 2"]
    B2 --> C["Shared Output Linear (64-d)"]
    B2 --> D["Decoder: User-Movie link prediction"]
  ```
---

### **Phase 2: LLM 통합 & RAG**
- **하이브리드 검색기**
  - ✅ 쿼리 라우터 (fact / personalized / chit-chat 분류)  
  - ✅ Zero-shot 프롬프트 기반 라우팅  
  - ✅ 상태 기반 통합 검색 로직 구축
 
- **엔티티 텍스트 매칭**
  - ✅ 배우, 감독, 장르, 영화 이름을 JSON 매핑 파일로부터 로드
  - ✅ `SentenceTransformer (all-MiniLM-L6-v2)`를 사용해 텍스트 임베딩 생성
  - ✅ FAISS Index에 저장 후 사용자 쿼리 임베딩과 최근접 탐색 수행
  - ✅ 철자 오류나 표현 차이가 있어도 가장 유사한 엔티티를 안정적으로 매핑

- **사실 기반 검색**
  - ✅ Cypher 쿼리 생성 및 실행  
  - ✅ 결과 파싱 후 답변 생성  

- **개인화 추천**
  - ✅ **유저 Query → Preference 추출 및 Cypher 수행**  
    - 사용자 입력에서 배우, 감독, 장르, 영화 키워드를 추출  
    - 추출된 키워드를 내부 매핑 자산과 비교해 정제  
      - `find_best_name` (텍스트 임베딩 기반 근접 탐색)으로 철자 오류 보정  
      - `genre_mapper_chain` (LLM 기반)으로 장르 의미를 매핑  
    - 정제된 preference를 기반으로 Cypher 쿼리 생성 → Neo4j에서 후보 영화 조회  

  - ✅ **FAISS 기반 후보 확장**  
    - 조회된 영화 후보들을 global GNN 그래프(`global_graph_nx`)에 매핑  
    - `extract_subgraph_from_global`  
      - Seed 영화 노드 선정 (degree 기반 Top-K, 랜덤, embedding 유사도 기반)  
      - 후보 영화들 간 **shortest path**를 연결하여 최소 서브그래프 생성  
    - 해당 subgraph를 PyTorch Geometric 데이터 객체로 변환  

  - ✅ **GAT Attention 기반 노드 중요도 추출**  
    - `run_gat_model`을 통해 **GATRanker** 실행  
    - subgraph 내 각 노드의 **attention score** 산출  
    - Attention score는 "이 노드가 현재 사용자 preference 맥락에서 얼마나 중요한가"를 의미  

  - ✅ **품질 지표(평점 + 인기도) 결합**  
    - `fetch_movie_quality_scores_from_nodes`로 영화별  
      - 평균 평점(`avg_rating`)  
      - 평점 개수(`rating_count`) 조회  
    - `rank_movies_by_attention`에서 점수 결합:  
      - `final_score = α * attention_score + β * quality_score`
      - `attention_score`: GAT 모델에서 학습된 중요도  
      - `quality_score`: (평균 평점 정규화 + 평점 수 정규화)  
      - `α=0.7, β=0.3` → GAT 기반 중요도에 더 높은 가중치  

  - ✅ **영화 설명(overview) 보강**  
    - `fetch_movie_overviews`로 최종 후보의 제목과 줄거리(overview) 조회  
    - `enrich_movies_with_overview`로 결과 아이템을 메타데이터와 함께 보강  

  - ✅ **최종 추천**  
    - Attention × 품질 기반으로 rerank된 영화 중 상위 Top-K (`top_k=5`)를 반환  


---

### **Phase 3: 애플리케이션 & 배포**
- **UI (Gradio)**
  - ✅ Gradio Chatbot UI 구현  
  - ✅ 초기 인사말 메시지 기능 추가  
  - ✅ 백엔드 검색 로직과 연동  

- **개선 작업**
  - ✅ 추천 랭킹 개선 (관련성 × 정규화된 평점/인기도 결합)  
  - ✅ “관련성 + 적정 인기” 균형 달성  

---

## 🛠️ 기술 스택
- **그래프 DB**: Neo4j (지식 그래프, Cypher 쿼리)  
- **GNN**: PyTorch Geometric (HGAT 기반)  
- **벡터 검색**: FAISS (임베딩 유사도 검색)  
- **LLM**: OpenAI GPT (라우팅, Cypher 생성, 개인화 답변)  
- **프레임워크**: LangChain / 커스텀 체인  
- **UI**: Gradio  

---
## 📂 프로젝트 구조

```text
src/
├── gnn/                          # GNN 관련 모듈
│   ├── build_knowledge_graph.py  # Neo4j 기반 지식 그래프 구축
│   ├── export_for_gnn.py         # Neo4j 데이터를 GNN 학습용 포맷으로 변환
│   ├── faiss_mapping.py          # 노드 임베딩 → FAISS 인덱스 매핑
│   ├── neo4j_utils.py            # Neo4j 연동 유틸리티
│   └── train_gnn.py              # GNN 학습 및 임베딩 생성
│
├── preprocess/                   # 데이터 전처리 모듈
│   ├── extension_converter.py    # 확장자/형식 변환
│   ├── preprocess_data_async.py  # 비동기 데이터 전처리
│   └── preprocess_shrink.py      # 데이터 축소/샘플링 전처리
│
├── rag_pipeline/                  # RAG 파이프라인 (챗봇 백엔드)
│   ├── app.py                    # Gradio 앱 실행 진입점
│   ├── chains.py                 # LangChain 체인 정의
│   ├── gnn_encoder.py            # GNN 인코더
│   ├── graph_utils.py            # 그래프 유틸 (NetworkX, Neo4j 헬퍼)
│   ├── main.py                   # 실행 스크립트
│   ├── retriever.py              # Hybrid retriever (fact / personalized / chit-chat)
│   └── utils.py                  # 퍼지 매칭, Cypher 정제 등 유틸 함수
│
├── txt_emb/                       # 텍스트 임베딩 관련
│   └── text_emb.py                # 텍스트 임베딩 추출
│
└── validator/                     # 임베딩 검증 모듈
    ├── verify_embeddings_diff_type.py  # 서로 다른 타입 간 임베딩 검증
    └── verify_embeddings_same_type.py  # 동일 타입 내 임베딩 검증
```

## 💡 주요 기여
- **이종 그래프 GNN 임베딩**과 **GraphRAG**를 결합한 하이브리드 영화 추천 구현  
- **라우터 + 리트리버 구조**로 fact/personalized/chit-chat 쿼리 분리 처리  
- **설명 가능한 추천**: 서브그래프 근거 경로 기반 설명 제공  
- **랭킹 개선**: relevance × popularity normalization  

---

