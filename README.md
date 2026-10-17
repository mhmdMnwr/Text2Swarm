# Text2Swarm: Text-to-3D Drone Swarm Formation Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Point-E](https://img.shields.io/badge/Point--E-OpenAI-black.svg)](https://github.com/openai/point-e)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly%20%26%20Matplotlib-3f4f75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Text2Swarm** is an end-to-end computational pipeline that transforms natural language prompts or literal text into **sparse, highly recognizable 3D drone swarm coordinate formations** ($N \in [15, 200]$ drones). 

By pairing generative 3D diffusion ([OpenAI Point-E](https://github.com/openai/point-e)) with geometry-aware sparse downsampling algorithms, Text2Swarm preserves sharp silhouettes, wingtips, rims, and structural edges at ultra-low point counts where standard downsampling methods fail.

---

## Key Features

- **Hybrid Prompt Router:**
  - **3D Semantic Prompts** (e.g., `"fighter jet"`, `"an apple"`) &rarr; OpenAI Point-E diffusion model.
  - **Literal Text / Flat Icons** (e.g., `"text:HELLO"`, `"text:SWARM"`) &rarr; 2D morphological edge extraction and multi-layer $Z$-axis extrusion.
  - **Instant Procedural Mocks** (`mock=True`) &rarr; Procedural 1,024-point aircraft generator running in $<0.01$s on CPU for instant development and algorithm testing.
- **Pre-Cleaning & Unit Normalization:**
  - **Statistical Outlier Removal (SOR):** Purges disconnected "floater" diffusion noise points using local $k$-nearest neighbor distance thresholds ($\mu_i > \mu_{\text{global}} + 2.0\sigma_{\text{global}}$).
  - **Aspect-Preserving Bounding Box Normalization:** Centers the formation at $(0, 0, 0)$ and scales uniformly within $[-1.0, 1.0]^3$.
- **Geometry-Preserving Downsampling:**
  - **Mode A &mdash; Edge-Weighted 3D FPS ($N \ge 45$):** Uses local covariance matrix eigenvalues to calculate surface variation ($\kappa$) and extremity ($e$), prioritizing high-curvature creases and tips before Farthest Point Sampling.
  - **Mode B &mdash; 2.5D PCA Silhouette + Spine ($N < 45$):** Computes dominant projection planes via Singular Value Decomposition (SVD), extracts outer boundaries via polar angular binning, and captures primary structural spines with $Z$-depth compression.
  - **Mode C &mdash; Uniform FPS:** Baseline greedy distance maximization for uniform spatial coverage.
- **Hardware-Aware Execution:**
  - Auto-detects CUDA compute capability: enables GPU acceleration when compute capability $\ge 7.0$ (e.g., T4/V100/A100) and automatically falls back to CPU on older GPUs (compute capability $< 7.0$) to avoid PyTorch CUDA kernel errors.
- **Dual Visualizer & Multi-Format Exporter:**
  - Immediate 2D orthographic projections (Top $X\text{--}Y$ and Side $X\text{--}Z$) with Matplotlib.
  - Interactive 3D Plotly dual-subplot comparisons embedded inline within Jupyter / VS Code.
  - Standalone, portable `.html` 3D visualizer exports.
  - Structured `.npy` float32 coordinate arrays and rich `.json` metadata packages with clearance and density metrics.

---

## Architecture Overview

```mermaid
flowchart TD
    A["User Prompt (e.g. 'fighter jet' or 'text:SWARM')"] --> B{"Hybrid Prompt Router"}
    
    B -->|Semantic 3D Prompt| C["OpenAI Point-E Diffusion (base40M-textvec)"]
    B -->|Literal Text ('text:...')| D["2D Morphological Edge Extruder"]
    B -->|USE_MOCK = True| E["Procedural Aircraft Generator (<0.01s)"]
    
    C --> F["Statistical Outlier Removal (SOR)"]
    D --> G["Aspect-Preserving Normalization ([-1, 1]³)"]
    E --> G
    F --> G
    
    G --> H{"Downsample Dispatcher (Target: N drones)"}
    
    H -->|N >= 45 (Mode A)| I["Edge-Weighted 3D FPS\n(k-NN Covariance + Curvature κ + Extremity e)"]
    H -->|N < 45 (Mode B)| J["2.5D PCA Silhouette + Spine\n(SVD Projection + Polar Binning + Spine Extraction)"]
    H -->|Mode C / Baseline| K["Uniform Farthest Point Sampling (FPS)"]
    
    I --> L["Spatial Quality Verification (d_min, d_nn)"]
    J --> L
    K --> L
    
    L --> M["Visualizations: 2D Orthographic Projections (Matplotlib)"]
    L --> N["Interactive 3D Subplots (Plotly HTML)"]
    L --> O["Export: .npy (Float32 Coordinates) & .json (Metadata + Points)"]
```

---

## Downsampling Algorithms

Standard uniform sampling or random decimation loses sharp outlines and thin appendages (such as wings, tail fins, and rims) when reducing dense clouds ($\sim 1,024$ points) to drone swarm budgets ($15 \le N \le 200$). Text2Swarm uses specialized geometric downsampling:

### 1. Mode A: 3D Wireframe Edge-Weighted FPS ($N \ge 45$)
1. **Neighborhood Covariance Analysis:** Constructs a $k$-d tree ($k=20$) and calculates the covariance matrix for each point's neighborhood:
   $$C_i = \frac{1}{k} \sum_{p \in \mathcal{N}(p_i)} (p - \bar{p}_i)(p - \bar{p}_i)^T$$
2. **Surface Variation ($\kappa$):** Evaluates the sorted eigenvalues $\lambda_1 \le \lambda_2 \le \lambda_3$ of $C_i$:
   $$\kappa_i = \frac{\lambda_1}{\lambda_1 + \lambda_2 + \lambda_3}$$
   High $\kappa_i$ marks sharp edges, seams, corners, and surface discontinuities.
3. **Extremity Score ($e$):** Measures normalized Euclidean distance from the global centroid:
   $$e_i = \frac{\|p_i - \bar{p}\|}{\max_j \|p_j - \bar{p}\|}$$
4. **Composite Score & FPS:** Computes $S_i = 0.70 \hat{\kappa}_i + 0.30 \hat{e}_i$. Selects a top candidate pool ($M_{\text{pool}} = \max(\lfloor 0.35M \rfloor, 2N)$) and performs deterministic Farthest Point Sampling across this candidate set.

### 2. Mode B: 2.5D PCA Silhouette + Spine ($N < 45$)
When $N < 45$, 3D surface points become too sparse for human depth perception. Mode B flattens the cloud into a high-impact recognizable silhouette:
1. **Principal Component Analysis (PCA):** Projects the centered point cloud onto its two principal variance axes using SVD:
   $$\text{coords}_{\text{PCA}} = (P - \bar{P}) V^T$$
2. **Polar Angular Binning:** Computes polar angles $\theta = \operatorname{atan2}(Y_{\text{PCA}}, X_{\text{PCA}})$ and partitions points into $B = \max(72, 2N)$ angular slices. Selects points with maximum radial distance $r = \sqrt{X^2 + Y^2}$ in each slice.
3. **Structural Spine Extraction:** Identifies high-curvature interior points ($\approx 15\%$ of points) to form the internal structural skeleton.
4. **$Z$-Depth Compression & FPS:** Compresses the third axis ($Z_{\text{PCA}} \leftarrow 0.45 \cdot Z_{\text{PCA}}$) and executes FPS over the combined silhouette-and-spine pool.

---

## Project Structure

```text
Text2Swarm/
├── testing.ipynb               # Complete interactive Jupyter notebook pipeline
├── point_e_model_cache/        # Cached Point-E model weights (base40M-textvec, ViT-L-14)
├── outputs/                    # Output directory for exported formations and figures
│   ├── <prompt>_<N>pts_<mode>.npy           # (N, 3) float32 coordinate arrays
│   ├── <prompt>_<N>pts_<mode>.json          # Formation metadata, metrics, and colors
│   └── <prompt>_<N>pts_<mode>_comparison.html# Standalone interactive Plotly 3D visualizer
├── .gitignore                  # Git ignore rules for caches, weights, and artifacts
└── README.md                   # Project documentation
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10 or 3.11
- (Optional) NVIDIA GPU with CUDA Compute Capability $\ge 7.0$ (CPU execution is fully supported via automatic fallback or `USE_MOCK=True`)

### 2. Environment Setup

Clone this repository and create a virtual or Conda environment:

```bash
git clone https://github.com/<your-username>/Text2Swarm.git
cd Text2Swarm

# Option A: Using Conda (Recommended)
conda create -n text2swarm python=3.11 -y
conda activate text2swarm

# Option B: Using venv
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install PyTorch and core scientific libraries:

```bash
# PyTorch (Install appropriate CUDA version or CPU build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # Or cpu build

# Core scientific and visualization libraries
pip install numpy scipy pillow matplotlib plotly ipykernel ipython
```

Install OpenAI Point-E:

```bash
pip install git+https://github.com/openai/point-e.git
```

### 4. Model Weights Cache (Optional)

The pipeline automatically downloads the `base40M-textvec` checkpoint upon first run. To use a local directory cache or run offline, Point-E looks in `./point_e_model_cache` or the directory specified by `POINT_E_CACHE_DIR`:

```bash
export POINT_E_CACHE_DIR="./point_e_model_cache"
```

---

## Usage

Open the interactive notebook:

```bash
jupyter notebook testing.ipynb
# Or open in VS Code with the Jupyter extension
```

### Pipeline Configuration

In Cell 9 of [testing.ipynb](file:///home/mnwr/Desktop/Text2Swarm/testing.ipynb), configure your generation parameters:

```python
# --- User Configuration ---
PROMPT = "fighter jet"     # e.g., "fighter jet", "an apple", or "text:SWARM"
NUM_POINTS = 35            # Target number of drones N (15 - 200)
MODE = "auto"              # 'auto', 'edge3d', 'silhouette25d', 'uniform_fps', or 'text'
USE_MOCK = True            # Set True for instant <0.01s testing, False for Point-E diffusion
OUTPUT_DIR = "./outputs"   # Destination directory for exported files
```

### Execution Modes

| Parameter Value | Description | Best For |
|---|---|---|
| `MODE = "auto"` | Automatically selects `silhouette25d` if $N < 45$ or if prompt is `text:...`; otherwise selects `edge3d`. | General use |
| `MODE = "edge3d"` | Edge-weighted 3D wireframe sampling using curvature $\kappa$ and extremity $e$. | Formations with $N \ge 45$ |
| `MODE = "silhouette25d"` | 2.5D PCA silhouette boundary + structural spine with $Z$-axis compression. | Low drone counts ($N < 45$) |
| `MODE = "uniform_fps"` | Standard Farthest Point Sampling without edge weighting. | Uniform volume baseline |
| `PROMPT = "text:WORD"` | Extrudes 2D literal letters into 3D multi-layered coordinates. | Drone light show words/logos |
| `USE_MOCK = True` | Synthesizes a procedural 1,024-point aircraft in $<0.01$s. | Instant CPU testing & prototyping |

---

## Output Data Formats

When running the pipeline, two primary files are generated inside `./outputs/`:

### 1. NumPy Coordinate Matrix (`.npy`)
A binary float32 array of shape `(N, 3)` where every row $[x, y, z]$ represents a normalized 3D drone position in $[-1.0, 1.0]^3$.

```python
import numpy as np
coords = np.load("outputs/fighter_jet_35pts_silhouette25d.npy")
print(coords.shape)  # (35, 3)
```

### 2. Structured Metadata Package (`.json`)
A structured JSON file containing generation metadata, spatial clearance metrics, coordinates, and RGB color channels:

```json
{
  "prompt": "fighter jet",
  "mode": "silhouette25d",
  "num_points": 35,
  "min_distance": 0.1428,
  "mean_nn_distance": 0.2841,
  "points": [
    [0.7241, 0.0012, 0.0514],
    [-0.4120, 0.6512, 0.0189]
  ],
  "colors": [
    [0.10, 0.75, 0.95],
    [0.15, 0.25, 0.65]
  ]
}
```

### Spatial Quality Metrics
- **Minimum Pairwise Distance ($d_{\min}$):** $\min_{i \ne j} \|p_i - p_j\|_2$. Measures drone collision safety buffer.
- **Mean Nearest-Neighbor Distance ($\bar{d}_{\text{nn}}$):** $\frac{1}{N} \sum_{i=1}^N \min_{j \ne i} \|p_i - p_j\|_2$. Measures formation uniformity and spatial distribution.

---

## Scope & Boundaries

> [!NOTE]
> Text2Swarm strictly addresses **static 3D coordinate formation generation, feature extraction, and geometric preservation**.
> 
> Downstream tasks—including temporal trajectory generation, collision avoidance, flight dynamics simulation, wind disturbance rejection, and Hungarian drone-to-waypoint assignment—are handled by subsequent swarm execution modules.

---

## Acknowledgments & References

- [OpenAI Point-E](https://github.com/openai/point-e): *Point-E: A System for Generating 3D Point Clouds from Complex Prompts* (Nichol et al., 2022).
- [Plotly Technologies](https://plotly.com/): Interactive 3D web visualizations.
- [SciPy Spatial](https://docs.scipy.org/doc/scipy/reference/spatial.html): $k$-d tree accelerated neighbor searches.
