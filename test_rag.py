import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.30
ASSETS_DIR = "assets"
DATA_DIR = "data"

CATEGORY_SEMANTICS = {
    'airplane': '3D airplane aircraft jet plane supersonic fighter airliner with wings fuselage tail rudder cockpit engines flying',
    'chair': '3D chair seating furniture armchair stool office swivel chair couch bench with four legs seat backrest armrests place to sit',
    'table': '3D table desk dining table workstation furniture coffee table with tabletop surface four legs apron',
    'car': '3D car automobile motor vehicle sports car sedan coupe with wheels chassis bumpers windshield roof doors',
    'guitar': '3D guitar acoustic electric music musical instrument with neck strings fretboard body headstock bouts playing songs',
    'laptop': '3D laptop notebook portable personal computer pc with open screen display keyboard trackpad hinge',
    'bed': '3D bed bedroom furniture mattress frame with headboard footboard corner posts legs pillows place to sleep',
    'bottle': '3D bottle water bottle beverage container flask with base cylinder body shoulder neck cap lip',
    'mug': '3D mug coffee cup ceramic teacup drinking vessel with cylinder body rim base curved handle hot drink',
    'skateboard': '3D skateboard deck skate board with four wheels trucks nose tail rails grip street skating',
    'motorcycle': '3D motorcycle motorbike motor bike chopper scooter with two wheels handlebars gas tank seat engine fork',
    'bathtub': '3D bathtub bath tub jacuzzi bathroom basin with rim floor base walls drainage washing bathing',
    'cap': '3D cap baseball cap hat headwear visor brim crown apex button headband',
    'helmet': '3D helmet safety head protection motorcycle helmet combat dome visor chin strap',
    'knife': '3D knife kitchen knife dagger blade cutting cutlery weapon with sharp blade tip hilt handle pommel',
    'vessel_boat': '3D vessel boat ship sailing yacht sailboat watercraft catamaran with hull deck bow stern keel mast ocean navigation'
}

def resolve_path(filename, parent_dir=""):
    """Finds a file in parent_dir, root, or relative paths."""
    candidates = [
        os.path.join(parent_dir, filename) if parent_dir else filename,
        filename,
        os.path.join("..", parent_dir, filename) if parent_dir else os.path.join("..", filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(parent_dir, filename) if parent_dir else filename

# ------------------------------------------------------------------------------
# Database & Model Loader
# ------------------------------------------------------------------------------
def load_rag_database():
    """Loads wireframe database and precomputed embeddings."""
    db_path = resolve_path("keypointnet_wireframes.json", DATA_DIR)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}. Run 01_dataset_generation.ipynb first.")
        
    with open(db_path, "r", encoding="utf-8") as f:
        wireframes_db = json.load(f)
        
    # Also load sample_wireframes.json if present
    sample_path = resolve_path("sample_wireframes.json", DATA_DIR)
    if not os.path.exists(sample_path):
        sample_path = resolve_path("sample_wireframes.json", "")
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        for k, v in samples.items():
            if k not in wireframes_db:
                wireframes_db[k] = {
                    "category": k,
                    "description": f"3D {k.replace('_', ' ')}: {v['description']}",
                    "anchors": v["anchors"],
                    "edges": v["edges"]
                }
                
    emb_path = resolve_path("wireframes_embeddings.npy", DATA_DIR)
    keys_path = resolve_path("wireframes_keys.json", DATA_DIR)
    
    # Initialize embedding model on CPU
    print("Loading embedding model (all-MiniLM-L6-v2 on CPU)...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    if os.path.exists(emb_path) and os.path.exists(keys_path):
        with open(keys_path, "r", encoding="utf-8") as f:
            db_keys = json.load(f)
        db_embeddings = np.load(emb_path)
    else:
        print("Computing semantic embeddings for database models...")
        db_keys = list(wireframes_db.keys())
        descriptions = []
        for k in db_keys:
            m = wireframes_db[k]
            cat = m.get("category", k.split("_")[0])
            base_sem = CATEGORY_SEMANTICS.get(cat, m.get("description", ""))
            K = len(m["anchors"])
            E = len(m["edges"])
            descriptions.append(f"{base_sem} | Model: {k} ({K} anchors, {E} edges)")
            
        db_embeddings = embedding_model.encode(descriptions, batch_size=128, normalize_embeddings=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        np.save(os.path.join(DATA_DIR, "wireframes_embeddings.npy"), db_embeddings)
        with open(os.path.join(DATA_DIR, "wireframes_keys.json"), "w", encoding="utf-8") as f:
            json.dump(db_keys, f)
            
    return wireframes_db, db_keys, db_embeddings, embedding_model

# ------------------------------------------------------------------------------
# Swarm Math & Subdivision
# ------------------------------------------------------------------------------
def calculate_min_drones_for_human_eye(shape: dict, d_gap: float = 0.25) -> int:
    """Calculates perceptual minimum drone count (N_min) to avoid constellation effect."""
    anchors = np.array(shape["anchors"])
    edges = shape["edges"]
    K = len(anchors)
    edge_drones = sum(max(0, int(np.ceil(np.linalg.norm(anchors[i] - anchors[j]) / d_gap)) - 1) for i, j in edges)
    return K + edge_drones

def build_swarm_coordinates(shape: dict, num_drones: int):
    """Subdivides edges to distribute drones evenly along structural lines."""
    anchors = np.array(shape["anchors"], dtype=float)
    edges = shape["edges"]
    K = len(anchors)
    
    if num_drones < K:
        return None, None
        
    points = list(anchors)
    point_types = ["anchor"] * K
    remaining = num_drones - K
    
    if remaining > 0 and len(edges) > 0:
        edge_lengths = np.array([np.linalg.norm(anchors[j] - anchors[i]) for i, j in edges])
        total_len = np.sum(edge_lengths)
        exact = (edge_lengths / total_len) * remaining
        drones_per_edge = np.floor(exact).astype(int)
        remainder = remaining - np.sum(drones_per_edge)
        
        if remainder > 0:
            best_edges = np.argsort(exact - drones_per_edge)[-remainder:]
            for e in best_edges:
                drones_per_edge[e] += 1
                
        for (i, j), m in zip(edges, drones_per_edge):
            p1, p2 = anchors[i], anchors[j]
            for s in range(1, m + 1):
                t = s / (m + 1)
                points.append((1.0 - t) * p1 + t * p2)
                point_types.append("edge")
                
    return np.array(points), point_types

# ------------------------------------------------------------------------------
# 3D Plotting & Export
# ------------------------------------------------------------------------------
def render_and_save_swarm(shape_key: str, shape: dict, points: np.ndarray, point_types: list, score: float, output_path: str):
    """Renders high-resolution 3D swarm view and saves to assets."""
    anchors = np.array(shape["anchors"])
    edges = shape["edges"]
    
    anchor_mask = np.array(point_types) == "anchor"
    edge_mask = np.array(point_types) == "edge"
    
    fig = plt.figure(figsize=(10, 8), facecolor="#0D1117")
    ax = fig.add_subplot(1, 1, 1, projection="3d", facecolor="#0D1117")
    
    # 1. Guide wireframe edges
    for i, j in edges:
        p1, p2 = anchors[i], anchors[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#00E5FF", alpha=0.35, linewidth=1.2, linestyle="--")
        
    # 2. Edge-interpolated drones (cyan)
    if np.sum(edge_mask) > 0:
        edge_pts = points[edge_mask]
        ax.scatter(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2], color="#00E5FF", s=35, edgecolors="#FFFFFF", linewidth=0.5, alpha=0.95, label=f"Edge Drones ({np.sum(edge_mask)})")
        
    # 3. Corner anchor drones (gold)
    anchor_pts = points[anchor_mask]
    ax.scatter(anchor_pts[:, 0], anchor_pts[:, 1], anchor_pts[:, 2], color="#FFD700", s=70, edgecolors="#FFFFFF", linewidth=1.0, alpha=1.0, label=f"Corner Anchors ({len(anchor_pts)})")
    
    cat = shape.get("category", shape_key.split("_")[0]).upper()
    ax.set_title(f"{cat} Formation (N = {len(points)} Drones)\nModel: {shape_key} | Match Score: {score:.2f}",
                 color="#58A6FF", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_zlim([-1.05, 1.05])
    ax.view_init(elev=22, azim=-55)
    
    ax.grid(True, linestyle=":", alpha=0.25, color="#30363D")
    ax.tick_params(colors="#8B949E", labelsize=8)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor("#161B22")
        pane.set_edgecolor("#30363D")
        
    ax.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9", loc="upper right")
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=160, facecolor="#0D1117", bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------------------
# Main Query Pipeline
# ------------------------------------------------------------------------------
def process_query(prompt: str, target_drones: int = None, threshold: float = SIMILARITY_THRESHOLD):
    wireframes_db, db_keys, db_embeddings, embedding_model = load_rag_database()
    
    print("\n" + "=" * 70)
    print(f"USER QUERY: \"{prompt}\"")
    print("=" * 70)
    
    # 1. Semantic Embedding
    q_vec = embedding_model.encode(prompt, normalize_embeddings=True)
    scores = db_embeddings @ q_vec
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_key = db_keys[best_idx]
    
    # 2. Out-of-domain check
    if best_score < threshold:
        print(f"Cosine Similarity: {best_score:.3f} (Threshold: {threshold:.2f})")
        print("\nRESULT: this formation is not valid\n")
        print("=" * 70)
        return False
        
    shape = wireframes_db[best_key]
    K = len(shape["anchors"])
    n_min = calculate_min_drones_for_human_eye(shape)
    
    if target_drones is None:
        target_drones = n_min
        
    print(f"MATCH FOUND!")
    print(f"  • Shape Model:         {best_key}")
    print(f"  • Category:            {shape.get('category', best_key.split('_')[0])}")
    print(f"  • Cosine Similarity:   {best_score:.3f}")
    print(f"  • Mandatory Anchors:   {K} drones")
    print(f"  • Perceptual N_min:    {n_min} drones (for human visual recognition)")
    print(f"  • Target Swarm Budget: {target_drones} drones")
    
    # 3. Anchor Count Validation
    if target_drones < K:
        print(f"\nERROR: Cannot form shape: target drones ({target_drones}) is less than mandatory anchors ({K}).\n")
        print("=" * 70)
        return False
        
    # 4. Generate Swarm Coordinates
    points, point_types = build_swarm_coordinates(shape, target_drones)
    
    # 5. Render and Save Image to Assets
    os.makedirs(ASSETS_DIR, exist_ok=True)
    out_file = os.path.join(ASSETS_DIR, "test_rag_output.png")
    render_and_save_swarm(best_key, shape, points, point_types, best_score, out_file)
    
    print(f"\nSUCCESS: Swarm formation generated with {len(points)} drones!")
    print(f"  • Saved 3D Image:      {out_file}")
    print("=" * 70 + "\n")
    return True

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test 3D Swarm Wireframe Semantic RAG System")
    parser.add_argument("prompt", nargs="?", type=str, default=None, help="Natural language text description")
    parser.add_argument("--drones", "-n", type=int, default=None, help="Target drone count (default: auto perceptual N_min)")
    parser.add_argument("--threshold", "-t", type=float, default=SIMILARITY_THRESHOLD, help="Cosine similarity threshold (default: 0.30)")
    
    args = parser.parse_args()
    
    prompt = args.prompt
    if not prompt:
        print("\n--- 3D Wireframe RAG Swarm Generator ---")
        prompt = input("Enter text prompt (e.g., 'a supersonic fighter jet'): ").strip()
        if not prompt:
            print("No prompt entered. Exiting.")
            sys.exit(0)
            
        drone_input = input("Enter drone count [Press Enter for automatic perceptual minimum]: ").strip()
        if drone_input.isdigit():
            args.drones = int(drone_input)
            
    process_query(prompt, target_drones=args.drones, threshold=args.threshold)
