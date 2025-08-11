import torch.nn as nn
from torch_geometric.nn import GATConv
import torch
from torch_geometric.data import Data
import networkx as nx

class GATRanker(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GATRanker, self).__init__()
        # 어텐션 스코어를 얻기 위해 heads=1 로 설정
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1)

    def forward(self, x, edge_index):
        # 어텐션 가중치를 반환받기 위해 return_attention_weights=True 로 설정
        h, (edge_index_with_attention, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        return h, alpha
    
# --- 2. 결과 분석: 별도의 순위 계산 함수 ---
# 이 함수는 모델의 '출력값'을 받아 순위를 매기는 것만 책임집니다.
def rank_movies_by_attention(
    attention_scores: torch.Tensor,
    data: Data,
    nodes: list,
    subgraph_nx: nx.Graph
):
    """
    GAT 모델이 출력한 어텐션 스코어를 기반으로 영화 노드의 중요도 순위를 매깁니다.
    """
    num_nodes = data.num_nodes
    node_importance = torch.zeros(num_nodes)
    
    # 노드별 중요도 계산
    node_importance.scatter_add_(0, data.edge_index[1], attention_scores.squeeze())
    
    # 영화 노드만 필터링하여 결과 리스트 생성
    results = []
    for i, importance in enumerate(node_importance):
        node_id = nodes[i]
        
        # 1. 노드에서 'type' 속성을 가져옵니다.
        node_type_attr = subgraph_nx.nodes[node_id].get('type')
        
        # 2. 타입이 'Movie'인지 확인하는 조건문 (수정된 부분)
        is_movie = False
        if isinstance(node_type_attr, list):
            # 타입이 리스트인 경우: 'Movie'가 리스트 안에 포함되어 있는지 확인
            if 'Movie' in node_type_attr:
                is_movie = True
        elif isinstance(node_type_attr, str):
            # 타입이 문자열인 경우: 문자열이 'Movie'와 일치하는지 확인
            if node_type_attr == 'Movie':
                is_movie = True
                
        # 3. 영화 타입이 맞으면 결과에 추가
        if is_movie:
            results.append({'title': node_id, 'importance': importance.item()})
            
    # 중요도 순으로 정렬
    sorted_movies = sorted(results, key=lambda x: x['importance'], reverse=True)
    return sorted_movies