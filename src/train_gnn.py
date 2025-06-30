# train_gnn.py (Final version for server execution)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv 
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from tqdm import tqdm
import numpy as np
import faiss
import json
import pickle # MODIFIED: Import the pickle module.

# REMOVED: No longer connecting directly to Neo4j, so the related utility is not needed.
# from neo4j_utils import get_neo4j_connection 

# --- 1. Configuration ---
# (Configuration section remains unchanged)
EMBEDDING_DIM = 128
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64
NUM_LAYERS = 2
NUM_HEADS = 4 
EPOCHS = 100
LEARNING_RATE = 0.001

# --- 2. Data Loading Function ---

# REMOVED: The load_data_from_neo4j(conn) function is no longer used in this script.
# (This function is only used in export_for_gnn.py on the local machine.)

def load_data_from_snapshot(snapshot_path='graph_snapshot.pkl'):
    """Loads the graph data snapshot from a file."""
    print(f"Loading graph data from snapshot: {snapshot_path}")
    
    with open(snapshot_path, 'rb') as f:
        graph_data_from_pickle = pickle.load(f)

    data = HeteroData()

    # Load node information
    node_mappings = graph_data_from_pickle['node_mappings']
    node_names = graph_data_from_pickle['node_names']
    for node_type, mapping in node_mappings.items():
        # Handle cases where the number of nodes is 0 to prevent errors.
        if len(mapping) > 0:
            data[node_type.lower()].num_nodes = len(mapping)
    print("Node data loaded.")

    # Load edge information
    edge_indices = graph_data_from_pickle['edge_indices']
    for edge_key, edge_index_list in edge_indices.items():
        data[edge_key].edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    print("Edge data loaded.")

    data = ToUndirected()(data)
    print("Data loading complete.")
    return data, node_names

# --- Verification, Model, Training/Testing Functions ---
# (verify_graph_data, HGAT, Decoder, train, test, generate_and_save_embeddings functions are unchanged)

def verify_graph_data(data, node_names):
    """Prints an overview and sample details of the HeteroData object."""
    print("\n" + "="*50)
    print(" Verifying Loaded Graph Data")
    print("="*50)

    # 1. Overview Verification
    print("\n--- Graph Overview ---")
    print(data)
    print("\nNode Counts:")
    for node_type in data.node_types:
        print(f"- {node_type.capitalize()}: {data[node_type].num_nodes}")
    print("\nEdge Counts:")
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        print(f"- ({src.capitalize()})-[{rel.upper()}]->({dst.capitalize()}): {data[edge_type].num_edges}")

    # 2. Sample Verification
    print("\n--- Sample Verification ---")
    target_movie_title = "Forrest Gump"
    
    try:
        movie_titles = node_names['Movie']
        target_movie_idx = movie_titles.index(target_movie_title)
        print(f"Inspecting movie: '{target_movie_title}' (PyG Index: {target_movie_idx})\n")

        # Find Genres associated with this movie
        genre_edge_index = data['movie', 'has_genre', 'genre'].edge_index
        genre_indices = genre_edge_index[1, genre_edge_index[0] == target_movie_idx]
        genres = [node_names['Genre'][i] for i in genre_indices]
        print(f"   - Genres: {genres if genres else 'None found'}")

        # Find Actors associated with this movie
        actor_edge_index = data['actor', 'acted_in', 'movie'].edge_index
        actor_indices = actor_edge_index[0, actor_edge_index[1] == target_movie_idx]
        actors = [node_names['Actor'][i] for i in actor_indices]
        print(f"   - Actors: {actors if actors else 'None found'}")

        # Find the Director of this movie
        director_edge_index = data['director', 'directed', 'movie'].edge_index
        director_indices = director_edge_index[0, director_edge_index[1] == target_movie_idx]
        directors = [node_names['Director'][i] for i in director_indices]
        print(f"   - Director: {directors[0] if directors else 'None found'}")

    except (ValueError, KeyError) as e:
        print(f"Could not find the sample movie '{target_movie_title}' or its relations. Error: {e}")
    except Exception as e:
        print(f"An error occurred during sample verification: {e}")

    print("\n" + "="*50 + " Graph Verification Complete " + "="*50 + "\n")


class HGAT(nn.Module):
    def __init__(self, data, embedding_dim, hidden_channels, out_channels, num_layers, num_heads):
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
        self.lin = nn.Linear(hidden_channels * num_heads, out_channels)

    def forward(self, data):
        x_dict = {
            node_type: self.node_embeds[node_type](data[node_type].node_id)
            for node_type in data.node_types
        }
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['movie']), x_dict

class Decoder(nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super().__init__()
        self.input_dim = hidden_channels * num_heads

    def forward(self, z_dict, edge_label_index):
        user_embed = z_dict['user'][edge_label_index[0]]
        movie_embed = z_dict['movie'][edge_label_index[1]]
        return (user_embed * movie_embed).sum(dim=-1)

def train(model, decoder, data, optimizer):
    model.train(); decoder.train()
    optimizer.zero_grad()
    _, z_dict = model(data)
    pos_out = decoder(z_dict, data['user', 'rated', 'movie'].edge_label_index)
    neg_out = decoder(z_dict, data['user', 'rated', 'movie'].edge_label_index_neg)
    pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, decoder, data):
    from sklearn.metrics import roc_auc_score
    model.eval(); decoder.eval()
    _, z_dict = model(data)
    pos_out = decoder(z_dict, data['user', 'rated', 'movie'].edge_label_index)
    neg_out = decoder(z_dict, data['user', 'rated', 'movie'].edge_label_index_neg)
    out = torch.cat([pos_out, neg_out]).cpu()
    y = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).cpu()
    return roc_auc_score(y, out)

def generate_and_save_embeddings(model, data, node_names):
    """Generates final movie embeddings and saves them using FAISS."""
    print("\n--- Generating and Saving Final Embeddings ---")
    model.eval()
    with torch.no_grad():
        movie_embeddings, _ = model(data)
        movie_embeddings_np = movie_embeddings.cpu().detach().numpy()

        print(f"Building FAISS index for {movie_embeddings_np.shape[0]} movies with dimension {OUT_CHANNELS}...")
        index = faiss.IndexFlatL2(OUT_CHANNELS)
        index.add(movie_embeddings_np)
        
        faiss_index_path = "movie_embeddings.faiss"
        faiss.write_index(index, faiss_index_path)
        print(f"FAISS index saved to: {faiss_index_path}")

        movie_titles = node_names['Movie']
        idx_to_title = {i: title for i, title in enumerate(movie_titles)}
        
        mapping_path = "faiss_to_movie_title.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(idx_to_title, f, ensure_ascii=False, indent=4)
        print(f"Index-to-title mapping saved to: {mapping_path}")


# --- 5. Main Execution Block ---
def main():
    """Main execution function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # MODIFIED: The DB connection logic is completely removed, leaving only the snapshot loading.
    graph_data, node_names = load_data_from_snapshot()

    # Verification step
    verify_graph_data(graph_data, node_names)
    input("Press Enter to continue with training...")

    # Add node_id for embedding lookup
    for node_type in graph_data.node_types:
        if data[node_type].num_nodes > 0:
            graph_data[node_type].node_id = torch.arange(graph_data[node_type].num_nodes)
    
    # Split data for link prediction
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        edge_types=[('user', 'rated', 'movie')],
        rev_edge_types=[('movie', 'rev_rated', 'user')],
    )
    train_data, val_data, test_data = transform(graph_data)
    train_data.to(device); val_data.to(device); test_data.to(device)

    # Initialize model and optimizer
    model = HGAT(train_data, EMBEDDING_DIM, HIDDEN_CHANNELS, OUT_CHANNELS, NUM_LAYERS, NUM_HEADS).to(device)
    decoder = Decoder(HIDDEN_CHANNELS, NUM_HEADS).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
    
    # Full Training Loop
    print("\nStarting model training...")
    for epoch in tqdm(range(1, EPOCHS + 1), desc="Training Progress"):
        loss = train(model, decoder, train_data, optimizer)
        if epoch % 10 == 0:
            val_auc = test(model, decoder, val_data)
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

    # Final Test
    test_auc = test(model, decoder, test_data)
    print(f"\nTraining complete. Final Test AUC: {test_auc:.4f}")

    # Generate and save final embeddings using the full graph data
    generate_and_save_embeddings(model, graph_data.to(device), node_names)


if __name__ == "__main__":
    main()