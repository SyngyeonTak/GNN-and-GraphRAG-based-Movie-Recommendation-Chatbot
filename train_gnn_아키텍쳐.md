graph TD
    subgraph "1. 입력 데이터 (Input Data)"
        A[User Nodes]
        B[Movie Nodes]
        C[...]
    end

    subgraph "2. 임베딩 레이어 (Embedding Layer)"
        D(nn.Embedding)
    end

    subgraph "3. GNN 레이어 (GNN Layers)"
        E[첫 번째 GAT 레이어<br>(in: -1, out: hidden_channels)]
        F[두 번째 이후 GAT 레이어<br>(in: hidden * heads, out: hidden)]
    end

    subgraph "4. 최종 출력 레이어 (Output Layer)"
        G[선형 레이어 (nn.Linear)<br>(in: hidden * heads, out: out_channels)]
    end

    subgraph "5. 최종 결과 (Final Result)"
        H[최종 노드 임베딩<br>(링크 예측 등에 사용)]
    end

    %% 데이터 흐름 연결
    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H