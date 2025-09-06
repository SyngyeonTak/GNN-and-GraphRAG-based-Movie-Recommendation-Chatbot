# 🎬 GNN & GraphRAG 기반 영화 추천 챗봇

> Neo4j + GNN + LLM 기반으로 사실 기반 검색과 개인화 추천을 결합한 영화 추천 챗봇

---

## 🚀 프로젝트 로드맵

### **Phase 1: 기초 & 데이터 (지식 그래프 + GNN)**
- **데이터 준비**
  - ✅ MovieLens 데이터셋 (movies, ratings, links.csv) 분석  
  - ✅ TMDb API 연동으로 배우/감독 메타데이터 수집  
  - ✅ 데이터 정제 및 `processed/` 폴더 저장  

- **지식 그래프 구축 (Neo4j)**
  - ✅ 노드: `Movie`, `User`, `Genre`, `Actor`, `Director`  
  - ✅ 관계: `RATED`, `HAS_GENRE`, `ACTED_IN`, `DIRECTED`  
  - ✅ Neo4j Browser에서 Cypher 쿼리 검증 완료  

- **GNN 추천 모델 학습**
  - ✅ 그래프 데이터를 PyTorch Geometric (PyG) 객체로 변환  
  - ✅ **Heterogeneous Graph Attention Network (HGAT)** 구현  
  - ✅ 학습 파이프라인 구축 및 손실 수렴 확인  
  - ✅ 추천 성능 평가 (AUC, ranking metric)  
  - ✅ **FAISS**에 노드 임베딩 저장 (유사도 검색)  
  - ✅ `genre`, `actor`, `director`, `user` 임베딩 추가 (LLM preference 반영)  
  - ✅ 노드 타입별 출력 레이어 공유 (임베딩 공간 정렬)  

---

### **Phase 2: LLM 통합 & RAG**
- **하이브리드 검색기**
  - ✅ 쿼리 라우터 (fact / personalized / chit-chat 분류)  
  - ✅ Zero-shot 프롬프트 기반 라우팅  
  - ✅ 상태 기반 통합 검색 로직 구축  

- **사실 기반 검색**
  - ✅ 엔티티 추출 + 퍼지 매칭 (영화/인물 이름 교정)  
  - ✅ Cypher 쿼리 생성 및 실행  
  - ✅ 결과 파싱 후 답변 생성  

- **개인화 추천**
  - ✅ Cold-start 대응 (선호도 질문 시나리오)  
  - ✅ FAISS 유사도 기반 Top-K 영화 검색  
  - ✅ 평점/인기도 정규화 → popularity bias 완화  
  - ✅ 장르 매핑 (LLM chain)  
  - ✅ 배우/감독/영화 이름 교정 (임베딩 매칭)  

- **설명 가능한 추천**
  - ✅ 추천 근거 경로 추출 (evidence path finding)  
  - ✅ 문맥적 서브그래프 생성 및 pruning  
  - ✅ Subgraph 기반 GAT attention 인코딩  
  - ✅ 영화 overview 기반 추가 필터링  

- **최종 답변 생성**
  - ✅ 검색 결과 + 추천 근거를 종합한 답변 생성  
  - ✅ fact-based / chit-chat 전용 체인 분리  
  - ✅ Cypher 생성 오류 개선  

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
src/
├── gnn/ # GNN 관련 모듈
│ ├── build_knowledge_graph.py # Neo4j 기반 지식 그래프 구축
│ ├── export_for_gnn.py # Neo4j 데이터를 GNN 학습용 포맷으로 변환
│ ├── faiss_mapping.py # 노드 임베딩 → FAISS 인덱스 매핑
│ ├── neo4j_utils.py # Neo4j 연동 유틸리티
│ └── train_gnn.py # GNN 학습 및 임베딩 생성
│
├── preprocess/ # 데이터 전처리 모듈
│ ├── extension_converter.py # 확장자/형식 변환
│ ├── preprocess_data_async.py # 비동기 데이터 전처리
│ └── preprocess_shrink.py # 데이터 축소/샘플링 전처리
│
├── rag_pipeline/ # RAG 파이프라인 (챗봇 백엔드)
│ ├── app.py # Gradio 앱 실행 진입점
│ ├── chains.py # LangChain 체인 정의
│ ├── gnn_encoder.py # GNN 인코더
│ ├── graph_utils.py # 그래프 유틸 (NetworkX, Neo4j 헬퍼)
│ ├── main.py # 실행 스크립트
│ ├── retriever.py # Hybrid retriever (fact / personalized / chit-chat)
│ └── utils.py # 퍼지 매칭, Cypher 정제 등 유틸 함수
│
├── txt_emb/ # 텍스트 임베딩 관련
│ └── text_emb.py # 텍스트 임베딩 추출
│
├── validator/ # 임베딩 검증 모듈
│ ├── verify_embeddings_diff_type.py # 서로 다른 타입 간 임베딩 검증
│ └── verify_embeddings_same_type.py # 동일 타입 내 임베딩 검증

## 💡 주요 기여
- **이종 그래프 GNN 임베딩**과 **GraphRAG**를 결합한 하이브리드 영화 추천 구현  
- **라우터 + 리트리버 구조**로 fact/personalized/chit-chat 쿼리 분리 처리  
- **설명 가능한 추천**: 서브그래프 근거 경로 기반 설명 제공  
- **랭킹 개선**: relevance × popularity normalization  

---

