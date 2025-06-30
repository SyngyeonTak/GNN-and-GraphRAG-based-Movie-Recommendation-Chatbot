# train_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv 
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from tqdm import tqdm
import numpy as np
import faiss
import json # NEW: For saving the mapping file

# Import the custom utility module
from neo4j_utils import get_neo4j_connection 

# --- 1. Configuration ---
# Model Hyperparameters
EMBEDDING_DIM = 128
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64
NUM_LAYERS = 2
NUM_HEADS = 4 
EPOCHS = 100
LEARNING_RATE = 0.001

# --- 2. Data Loading Function ---
def load_data_from_neo4j(conn):
    """Loads data from Neo4j and returns titles/names for verification."""
    print("Loading graph data from Neo4j...")
    data = HeteroData()

    # CHANGED: 'Person' node type is replaced with 'Actor' and 'Director'
    node_types = ['User', 'Movie', 'Genre', 'Actor', 'Director']
    node_mappings = {}
    node_names = {node_type: [] for node_type in node_types}

    for node_type in node_types:
        name_property = ""
        # CHANGED: 'Person' is removed, 'Actor' and 'Director' are checked for 'name' property
        if node_type == 'Movie':
            name_property = ", n.title AS name"
        elif node_type in ['Genre', 'Actor', 'Director']:
            name_property = ", n.name AS name"
        
        query = f"MATCH (n:{node_type}) RETURN elementId(n) AS neo4j_id{name_property}"
        records = conn.execute_query(query)
        
        # Check if records are empty
        if not records:
            print(f"Warning: Found 0 '{node_type}' nodes. Skipping.")
            data[node_type.lower()].num_nodes = 0
            continue

        node_mappings[node_type] = {record["neo4j_id"]: i for i, record in enumerate(records)}
        if name_property:
            node_names[node_type] = [record["name"] for record in records]
        
        data[node_type.lower()].num_nodes = len(node_mappings[node_type])
        print(f"Found {len(node_mappings[node_type])} '{node_type}' nodes.")

    # CHANGED: Edge definitions now use Actor and Director specifically
    edge_definitions = [
        ('User', 'RATED', 'Movie'), 
        ('Movie', 'HAS_GENRE', 'Genre'),
        ('Actor', 'ACTED_IN', 'Movie'), 
        ('Director', 'DIRECTED', 'Movie')
    ]

    for src_type, rel_type, dst_type in edge_definitions:
        query = f"MATCH (src:{src_type})-[r:{rel_type}]->(dst:{dst_type}) RETURN elementId(src) AS src_id, elementId(dst) AS dst_id"
        records = conn.execute_query(query)

        # Check for empty relationships
        if not records:
            print(f"Warning: Found 0 '{rel_type}' relationships. Skipping.")
            continue

        src_nodes = [node_mappings[src_type][r['src_id']] for r in records]
        dst_nodes = [node_mappings[dst_type][r['dst_id']] for r in records]
        edge_key = (src_type.lower(), rel_type.lower(), dst_type.lower())
        data[edge_key].edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        print(f"Found {len(records)} '{rel_type}' relationships.")
        
    data = ToUndirected()(data)
    print("Data loading complete.")
    return data, node_names

# --- Verification Function ---
def verify_graph_data(data, node_names):
    """Prints an overview and sample details of the HeteroData object."""
    print("\n" + "="*50)
    print(" Verifying Loaded Graph Data")
    print("="*50)

    # ... (Overview print part is the same) ...

    # Sample Verification
    print("\n--- Sample Verification ---")
    target_movie_title = "Forrest Gump"
    
    try:
        movie_titles = node_names['Movie']
        target_movie_idx = movie_titles.index(target_movie_title)
        print(f"Inspecting movie: '{target_movie_title}' (PyG Index: {target_movie_idx})\n")

        # Find Genres
        genre_edge_index = data['movie', 'has_genre', 'genre'].edge_index
        genre_indices = genre_edge_index[1, genre_edge_index[0] == target_movie_idx]
        genres = [node_names['Genre'][i] for i in genre_indices]
        print(f"  - Genres: {genres if genres else 'None found'}")

        # CHANGED: Find Actors using ('actor', 'acted_in', 'movie') edge type
        actor_edge_index = data['actor', 'acted_in', 'movie'].edge_index
        actor_indices = actor_edge_index[0, actor_edge_index[1] == target_movie_idx]
        actors = [node_names['Actor'][i] for i in actor_indices]
        print(f"  - Actors: {actors if actors else 'None found'}")

        # CHANGED: Find Director using ('director', 'directed', 'movie') edge type
        director_edge_index = data['director', 'directed', 'movie'].edge_index
        director_indices = director_edge_index[0, director_edge_index[1] == target_movie_idx]
        directors = [node_names['Director'][i] for i in director_indices]
        print(f"  - Director: {directors[0] if directors else 'None found'}")

    except (ValueError, KeyError) as e:
        print(f"Could not find the sample movie '{target_movie_title}' or its relations. Error: {e}")
    except Exception as e:
        print(f"An error occurred during sample verification: {e}")

    print("\n" + "="*50 + " Graph Verification Complete " + "="*50 + "\n")


# --- 3. HGAT Model & Decoder Definition ---
# (This part is unchanged as the model dynamically adapts to node/edge types)
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

# --- 4. Training and Testing Functions ---
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

# --- NEW: Function to generate and save embeddings with FAISS ---
def generate_and_save_embeddings(model, data, node_names):
    """Generates final movie embeddings and saves them using FAISS."""
    print("\n--- Generating and Saving Final Embeddings ---")
    model.eval()
    with torch.no_grad():
        # Get final movie embeddings from the trained model
        movie_embeddings, _ = model(data)
        
        # Move embeddings to CPU and convert to NumPy array
        movie_embeddings_np = movie_embeddings.cpu().detach().numpy()

        # 1. Build the FAISS index
        print(f"Building FAISS index for {movie_embeddings_np.shape[0]} movies with dimension {OUT_CHANNELS}...")
        index = faiss.IndexFlatL2(OUT_CHANNELS)
        index.add(movie_embeddings_np)
        
        # 2. Save the FAISS index to a file
        faiss_index_path = "movie_embeddings.faiss"
        faiss.write_index(index, faiss_index_path)
        print(f"FAISS index saved to: {faiss_index_path}")

        # 3. Create and save the mapping from FAISS index to movie title
        movie_titles = node_names['Movie']
        # The PyG data loader preserves the order, so the i-th embedding corresponds to the i-th movie title
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

    conn = None
    try:
        conn = get_neo4j_connection()
        graph_data, node_names = load_data_from_neo4j(conn)

        # Verification step
        verify_graph_data(graph_data, node_names)
        input("Press Enter to continue with training...")

        # Add node_id for embedding lookup
        for node_type in graph_data.node_types:
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
        
        # NEW: Full Training Loop
        print("\nStarting model training...")
        for epoch in tqdm(range(1, EPOCHS + 1), desc="Training Progress"):
            loss = train(model, decoder, train_data, optimizer)
            if epoch % 10 == 0:
                val_auc = test(model, decoder, val_data)
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

        # Final Test
        test_auc = test(model, decoder, test_data)
        print(f"\nTraining complete. Final Test AUC: {test_auc:.4f}")

        # NEW: Generate and save final embeddings using the full graph data
        # We use the full 'graph_data' to generate embeddings for ALL movies
        generate_and_save_embeddings(model, graph_data.to(device), node_names)

    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()