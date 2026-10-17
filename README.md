# Text2Swarm 🛸

> Convert text prompts into recognizable 3D drone swarm formations ($N = 15$ to $200$ drones).

Text2Swarm takes a natural language prompt (e.g. `"fighter jet"`, `"an apple"`) or literal text (`"text:HELLO"`), generates a 3D point cloud, and intelligently downsamples it into a clean, recognizable drone formation in 3D space.

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/mhmdMnwr/Text2Swarm.git
cd Text2Swarm

# Install packages
pip install torch torchvision numpy scipy pillow matplotlib plotly
pip install git+https://github.com/openai/point-e.git
```

### 2. Run the Notebook

Open `testing.ipynb` in VS Code or Jupyter Notebook:

```bash
jupyter notebook testing.ipynb
```

Run **Cells 1 through 8** once, then customize **Cell 9**:

```python
PROMPT = "fighter jet"     # e.g., "fighter jet", "an apple", or "text:SWARM"
NUM_POINTS = 35            # Number of drones (15 to 200)
MODE = "auto"              # 'auto', 'edge3d', 'silhouette25d', or 'uniform_fps'
USE_MOCK = True            # True = instant (<0.01s), False = real Point-E 3D diffusion
```

Run **Cell 9** to see the **inline 3D plots** and export the coordinates!

---

## 🎯 How It Works

1. **3D Generation:** Creates a dense point cloud via [OpenAI Point-E](https://github.com/openai/point-e), 2D text extrusion, or instant procedural mock (<0.01s).
2. **Pre-Cleaning:** Removes stray floating points using Statistical Outlier Removal (SOR) and normalizes coordinates to $[-1, 1]^3$.
3. **Smart Downsampling:**
   - **Mode A (`edge3d`):** Prioritizes sharp creases, wingtips, and rims (best for $N \ge 45$).
   - **Mode B (`silhouette25d`):** Extracts outer silhouettes and central spines (best for low point counts $N < 45$).
   - **Mode C (`uniform_fps`):** Uniform surface coverage.
4. **3D Visualization:** Displays side-by-side **True 3D perspective plots** directly in the notebook and exports interactive Plotly 3D HTML files.

---

## 📦 Output Files

Generated formations are saved in `./outputs/`:
- **`.npy`**: Normalized 3D coordinate array of shape `(N, 3)` in `float32`.
- **`.json`**: Drone waypoints, clearance distance metrics, and RGB colors.
- **`.html`**: Standalone rotatable 3D viewer.

```python
import numpy as np
coords = np.load("outputs/fighter_jet_35pts_silhouette25d.npy")
print(coords.shape)  # (35, 3)
```

---

## 📄 License
MIT License.
