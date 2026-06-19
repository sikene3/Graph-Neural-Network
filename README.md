# AML-GNN: Graph Neural Network for Anti-Money Laundering Detection

## 🎥 Project Demo

Watch the system in action!

**Enterprise-grade money laundering detection pipeline** — from raw IBM AML tabular data to interactive GNN-powered investigation dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red.svg)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyG-2.8-green.svg)](https://pyg.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B.svg)](https://streamlit.io)

---

## Overview

This project converts the IBM AMLSim tabular transaction dataset into a **heterogeneous transaction graph**, trains a **GraphSAGE edge classifier** to detect money laundering, and provides an **interactive Streamlit dashboard** for investigators to explore suspicious account neighborhoods.

### Key Results (HI-Small Dataset)

| Metric | Value |
|--------|-------|
| Nodes (accounts) | 515,080 |
| Edges (transactions) | 5,078,345 |
| Fraud rate | 0.0632% |
| **Test Recall** | **0.7143** |
| **Test AUPRC** | **0.0423** |

---

## Project Structure

```
.
├── data/
│   ├── raw/                          # Original IBM AMLSim CSV/TXT files
│   │   ├── HI-Small_Trans.csv        # ~5M transactions
│   │   ├── HI-Small_Patterns.txt     # Laundering labels
│   │   ├── HI-Small_accounts.csv     # Account metadata
│   │   └── ...                       # Medium/Large variants
│   └── processed/
│       └── graph/                    # Preprocessed graph artifacts
│           ├── nodes_mapped.parquet   # 515K nodes (node_id → account)
│           └── edges_labeled.parquet  # 5M edges + features + labels
├── artifacts/                        # Trained model & metrics
│   ├── aml_gnn_model.pth             # GraphSAGE model weights
│   └── training_metrics.json         # KPI metrics for dashboard
├── src/                              # Source code
│   ├── parse_aml_graph.py            # Step 1: CSV → Graph preprocessing
│   ├── train_gnn.py                  # Step 2: GNN training
│   └── app_aml_investigator.py       # Step 3: Streamlit dashboard
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Raw CSV/TXT     │────▶│  parse_aml_graph  │────▶│  train_gnn.py       │
│  (IBM AMLSim)    │     │  (Chunked ETL)    │     │  (GraphSAGE + MLP)  │
└─────────────────┘     └────────┬─────────┘     └──────────┬──────────┘
                                 │                          │
                          nodes_mapped.parquet      aml_gnn_model.pth
                          edges_labeled.parquet     training_metrics.json
                                 │                          │
                                 └──────────┬───────────────┘
                                            ▼
                                 ┌─────────────────────┐
                                 │  app_aml_investigator│
                                 │  (Streamlit + PyVis) │
                                 └─────────────────────┘
```

### Step 1 — Graph Preprocessing (`parse_aml_graph.py`)

- **Memory-efficient**: Chunked CSV reading (500K rows/chunk) with PyArrow-backed string dtypes
- **Node mapping**: Extracts 515K unique accounts → sequential integer IDs (0 to N-1)
- **Edge labeling**: Matches transactions against `Patterns.txt` via composite-key hashing
- **Output**: Snappy-compressed Parquet files (~80 MB total)

### Step 2 — GNN Training (`train_gnn.py`)

- **Model**: 2-layer GraphSAGE encoder + 3-layer MLP edge predictor
- **Node features**: Log-normalized in/out degree
- **Edge features**: Log-normalized amounts + one-hot encoded currency/format
- **Mini-batching**: `RandomNodeLoader` (200 partitions) — no `pyg-lib` required
- **Loss**: `BCEWithLogitsLoss` with `pos_weight=1581.5` to handle 0.06% fraud rate
- **Metrics**: Precision, Recall, F1, AUPRC (via `torchmetrics`)
- **Early stopping**: Patience=10 on validation AUPRC

### Step 3 — Investigation Dashboard (`app_aml_investigator.py`)

- **KPI sidebar**: AUPRC, Recall, Precision, F1, dataset statistics
- **Node search**: By integer ID or account hash string
- **Subgraph extraction**: 1-hop or 2-hop neighborhood
- **GNN inference**: Real-time edge scoring on the extracted subgraph
- **PyVis visualization**: Interactive directed graph with color-coded risk levels
- **Transaction table**: Suspicious edges sorted by model confidence

---

## Getting Started

### Prerequisites

- Python 3.10+
- 8+ GB RAM (for loading the full edge table)
- CUDA-capable GPU optional (CPU training supported)

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Preprocess raw data into graph format
python src/parse_aml_graph.py

# 2. Train the GNN model
python src/train_gnn.py

# 3. Launch the investigation dashboard
streamlit run src/app_aml_investigator.py
```

### Using Other Dataset Sizes

The pipeline supports HI-Small, HI-Medium, HI-Large, LI-Small, LI-Medium, and LI-Large variants. To switch datasets, modify the `CSV_PATH` and `PATTERNS_PATH` constants in `src/parse_aml_graph.py`.

---

## Model Architecture

```
                    ┌──────────────────────┐
  Node Features     │   node_encoder       │
  (in/out degree)   │   Linear(2 → 64)     │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │   SAGEConv(64 → 128) │  ← 1-hop neighborhood aggregation
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │   SAGEConv(128 → 64) │  ← 2-hop neighborhood aggregation
                    └──────────┬───────────┘
                               ▼
  ┌─────────────────────────────────────────────────┐
  │  Edge Predictor                                 │
  │  [src_emb(64) ‖ dst_emb(64) ‖ edge_attr(39)]   │
  │  → Linear(167 → 128) → ReLU → Dropout          │
  │  → Linear(128 → 64)  → ReLU → Dropout          │
  │  → Linear(64 → 1)                               │
  └─────────────────────────────────────────────────┘
```

---

## Dashboard Features

| Feature | Description |
|---------|-------------|
| Dark theme | Enterprise-grade dark UI |
| KPI cards | AUPRC, Recall, Precision, F1 in sidebar |
| Dual search | Search by Node ID or Account Hash |
| Hop control | 1-hop (direct) or 2-hop (extended) neighborhood |
| Risk coloring | Red (≥0.5), Orange (0.1–0.5), Gray (<0.1) |
| Interactive graph | PyVis force-directed layout with tooltips |
| Suspicious table | Filtered view of high-risk transactions |

---

## Limitations & Future Work

- **Node features**: Currently degree-only; could incorporate account metadata from `accounts.csv`
- **Temporal dynamics**: Timestamps are not used; a temporal GNN (TGN) could capture evolving patterns
- **Heterogeneous graph**: Banks could be modeled as a separate node type with RGCN
- **Scalability**: `RandomNodeLoader` works on CPU; `NeighborLoader` with `pyg-lib` would be faster on GPU
- **Class imbalance**: 0.06% fraud rate limits precision; oversampling or focal loss may help

---

## License

This project is for educational and research purposes. The IBM AMLSim dataset is publicly available from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering).
