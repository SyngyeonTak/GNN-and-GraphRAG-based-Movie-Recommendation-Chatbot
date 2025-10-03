import torch.nn as nn
from torch_geometric.nn import GATConv
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np


class GATRanker(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GATRanker, self).__init__()
        # Set heads=1 to directly obtain attention scores
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1)

    def forward(self, x, edge_index):
        # Return attention weights with return_attention_weights=True
        h, (edge_index_with_attention, alpha) = self.conv1(
            x, edge_index, return_attention_weights=True
        )
        return h, alpha
    

def rank_movies_by_attention(attention_scores, data, nodes, subgraph_nx,
                             quality_map, alpha=0.7, beta=0.3):
    """
    Rank movie nodes based on GAT attention scores combined with quality metrics.
    - attention_scores: attention values from GAT
    - data: PyTorch Geometric Data object
    - nodes: list of node identifiers
    - subgraph_nx: NetworkX subgraph containing nodes
    - quality_map: dictionary of {movie_id: (avg_rating, n_ratings)}
    - alpha: weight for attention score
    - beta: weight for rating/popularity score
    """
    num_nodes = data.num_nodes
    node_importance = torch.zeros(num_nodes)
    node_importance.scatter_add_(0, data.edge_index[1], attention_scores.squeeze())

    # Find the maximum n_ratings value for normalization
    max_n_ratings = max((val[1] for val in quality_map.values()), default=1)

    results = []
    for i, importance in enumerate(node_importance):
        node_id = nodes[i]
        node_type_attr = subgraph_nx.nodes[node_id].get('type')

        is_movie = (isinstance(node_type_attr, list) and 'Movie' in node_type_attr) or \
                   (isinstance(node_type_attr, str) and node_type_attr == 'Movie')
        if not is_movie:
            continue

        attn_score = importance.item()

        # node_id에서 숫자만 추출
        if isinstance(node_id, str) and node_id.startswith("movie_"):
            movie_key = int(node_id.replace("movie_", ""))
        else:
            movie_key = node_id

        avg_rating, n_ratings = quality_map.get(movie_key, (0.0, 0))

        #avg_rating, n_ratings = quality_map.get(node_id, (0.0, 0))
        # Normalize values
        norm_avg_rating = avg_rating / 5.0
        norm_n_ratings = np.log1p(n_ratings) / np.log1p(max_n_ratings)

        quality_score = 0.5 * norm_avg_rating + 0.5 * norm_n_ratings
        final_score = alpha * attn_score + beta * quality_score

        results.append({
            "movie_id": node_id,
            "importance": attn_score,
            "avg_rating": avg_rating,
            "n_ratings": n_ratings,
            "quality_score": quality_score,
            "final_score": final_score
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)
