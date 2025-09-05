# train_gnn_no_user.py (HGAT embedding training without user nodes)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.transforms import ToUndirected
import numpy as np
import faiss
import json
import pickle

# --- 1. Configuration ---
EMBEDDING_DIM = 128
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64
NUM_LAYERS = 2
NUM_HEADS = 4
EPOCHS = 30
LEARNING_RATE = 0.001
NODE_TYPES_TO_EMBED = ['movie', 'genre', 'actor', 'director']

DUMP_PATH = './dataset/faiss/'

# --- 2. Data Loading ---
def load_data_from_snapshot(snapshot_path='./dataset/graph_snapshot.pkl'):
    print(f"Loading graph data from snapshot: {snapshot_path}")
    with open(snapshot_path, 'rb') as f:
        graph_data_from_pickle = pickle.load(f)

    data = HeteroData()

    node_mappings = graph_data_from_pickle['node_mappings']
    node_names = graph_data_from_pickle['node_names']
    for node_type, mapping in node_mappings.items():
        if len(mapping) > 0:
            data[node_type.lower()].num_nodes = len(mapping)
    print("Node data loaded.")

    edge_indices = graph_data_from_pickle['edge_indices']
    for edge_key, edge_index_list in edge_indices.items():
        data[edge_key].edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    print("Edge data loaded.")

    data = ToUndirected()(data)
    print("Data loading complete.")
    return data, node_names

# --- 3. Model Definition ---
class HGAT(nn.Module):
    def __init__(self, data, embedding_dim, hidden_channels, out_channels, num_layers, num_heads, node_types_to_embed):
        super().__init__()
        self.node_embeds = nn.ModuleDict()
        for node_type in data.node_types:
            self.node_embeds[node_type] = nn.Embedding(data[node_type].num_nodes, embedding_dim)

        self.convs = nn.ModuleList()
        # First Layer
        self.convs.append(HeteroConv({
            edge_type: GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False)
            for edge_type in data.edge_types
        }, aggr='sum'))

        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(HeteroConv({
                edge_type: GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, add_self_loops=False)
                for edge_type in data.edge_types
            }, aggr='sum'))

        self.shared_output_lin = nn.Linear(hidden_channels * num_heads, out_channels)
        self.node_types_to_embed = node_types_to_embed

    def forward(self, data):
        x_dict = {
            node_type: self.node_embeds[node_type](data[node_type].node_id)
            for node_type in data.node_types
        }
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        final_embeddings = {}
        for node_type in self.node_types_to_embed:
            if node_type in x_dict:
                final_embeddings[node_type] = self.shared_output_lin(x_dict[node_type])
        return final_embeddings, x_dict

# --- 4. Save embeddings with FAISS ---
def generate_and_save_embeddings(model, data, node_names, node_types_to_save):
    print("\n--- Generating and Saving Final Embeddings ---")
    model.eval()
    with torch.no_grad():
        final_embeddings_dict, _ = model(data)

        for node_type in node_types_to_save:
            if node_type not in final_embeddings_dict:
                print(f"Warning: Node type '{node_type}' not found in model output. Skipping.")
                continue

            print(f"\nProcessing embeddings for node type: '{node_type}'")
            embeddings = final_embeddings_dict[node_type]
            embeddings_np = embeddings.cpu().detach().numpy().astype('float32')

            # Build FAISS index
            print(f"Building FAISS index for {embeddings_np.shape[0]} '{node_type}' nodes with dimension {OUT_CHANNELS}...")
            index = faiss.IndexFlatL2(OUT_CHANNELS)
            index.add(embeddings_np)

            # Save FAISS index
            faiss_index_path = f"{DUMP_PATH}{node_type}_embeddings.faiss"
            faiss.write_index(index, faiss_index_path)
            print(f"FAISS index saved to: {faiss_index_path}")

            # Save index-to-name mapping
            node_type_capitalized = node_type.capitalize()
            if node_type_capitalized not in node_names:
                print(f"Warning: Could not find names for '{node_type_capitalized}' in node_names dict. Skipping mapping file.")
                continue

            titles = node_names[node_type_capitalized]
            idx_to_name = {i: title for i, title in enumerate(titles)}

            mapping_path = f"{DUMP_PATH}{node_type}_mapping.json"
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(idx_to_name, f, ensure_ascii=False, indent=4)
            print(f"Index-to-name mapping saved to: {mapping_path}")

# --- 5. Main Execution ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    graph_data, node_names = load_data_from_snapshot()

    for node_type in graph_data.node_types:
        if graph_data[node_type].num_nodes > 0:
            graph_data[node_type].node_id = torch.arange(graph_data[node_type].num_nodes)

    model = HGAT(graph_data, EMBEDDING_DIM, HIDDEN_CHANNELS, OUT_CHANNELS, NUM_LAYERS, NUM_HEADS, NODE_TYPES_TO_EMBED).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting model training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        final_embeddings, _ = model(graph_data.to(device))
        # NOTE: No supervision here, placeholder zero loss
        loss = torch.tensor(0.0, requires_grad=True).to(device)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch:03d} finished.")

    print("\nTraining complete. Generating embeddings...")
    generate_and_save_embeddings(model, graph_data.to(device), node_names, NODE_TYPES_TO_EMBED)

if __name__ == "__main__":
    main()