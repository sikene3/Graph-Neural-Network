#!/usr/bin/env python3
"""
app_aml_investigator.py — Enterprise AML Investigation Center
=============================================================

Streamlit dashboard for interactive GNN-based money-laundering
investigation.  Loads the preprocessed graph, the trained GraphSAGE
model, and training metrics, then provides:

  - KPI sidebar (AUPRC, Recall, Precision, etc.)
  - Node search (by integer ID or account hash)
  - 1-hop / 2-hop subgraph extraction
  - GNN inference on the subgraph to highlight suspicious edges
  - PyVis interactive network visualization
  - Suspicious-transaction table

Usage:
  streamlit run app_aml_investigator.py
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyvis.network import Network
from torch_geometric.nn import SAGEConv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NODES_PATH   = PROJECT_ROOT / "data/processed/graph/nodes_mapped.parquet"
EDGES_PATH   = PROJECT_ROOT / "data/processed/graph/edges_labeled.parquet"
MODEL_PATH   = PROJECT_ROOT / "artifacts/aml_gnn_model.pth"
METRICS_PATH = PROJECT_ROOT / "artifacts/training_metrics.json"

# ---------------------------------------------------------------------------
# Model architecture (must match train_gnn.py exactly)
# ---------------------------------------------------------------------------

HIDDEN_DIM      = 64
ENCODER_OUT_DIM = 64
PRED_HIDDEN     = 128
DROPOUT         = 0.3
EDGE_FEAT_DIM   = 39   # 2 continuous + 15 currency + 15 currency + 7 format


class AMLGraphSAGE(nn.Module):
    """
    GraphSAGE encoder + MLP edge classifier.
    Identical to the class in train_gnn.py — required for weight loading.
    """

    def __init__(
        self,
        node_feat_dim: int = 2,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        encoder_out_dim: int = ENCODER_OUT_DIM,
        pred_hidden: int = PRED_HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.dropout = dropout
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim * 2)
        self.conv2 = SAGEConv(hidden_dim * 2, encoder_out_dim)

        pred_in_dim = encoder_out_dim * 2 + edge_feat_dim
        self.predictor = nn.Sequential(
            nn.Linear(pred_in_dim, pred_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden, pred_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden // 2, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.node_encoder(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=False)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=False)
        x = self.conv2(x, edge_index)
        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index)
        src_h = h[edge_index[0]]
        dst_h = h[edge_index[1]]
        edge_repr = torch.cat([src_h, dst_h, edge_attr], dim=-1)
        return self.predictor(edge_repr).squeeze(-1)


# ---------------------------------------------------------------------------
# Caching: load heavy assets once per session
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model() -> AMLGraphSAGE:
    """Load the trained GraphSAGE model from disk."""
    model = AMLGraphSAGE()
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data
def load_edges() -> pd.DataFrame:
    """Load the full edges parquet file."""
    return pd.read_parquet(EDGES_PATH)


@st.cache_data
def load_nodes() -> pd.DataFrame:
    """Load the nodes mapping parquet file."""
    return pd.read_parquet(NODES_PATH)


@st.cache_data
def load_metrics() -> dict:
    """Load training metrics JSON."""
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def compute_global_stats(edges_df: pd.DataFrame) -> dict:
    """
    Precompute global statistics needed for feature normalisation
    and node-degree features.  Called once, cached.
    """
    num_nodes = 515080

    # --- Node degrees (log1p) ---
    src = torch.tensor(edges_df["src"].values, dtype=torch.long)
    dst = torch.tensor(edges_df["dst"].values, dtype=torch.long)
    in_deg = torch.zeros(num_nodes, dtype=torch.float32)
    out_deg = torch.zeros(num_nodes, dtype=torch.float32)
    ones = torch.ones(len(edges_df), dtype=torch.float32)
    in_deg.scatter_add_(0, dst, ones)
    out_deg.scatter_add_(0, src, ones)
    in_deg_log = torch.log1p(in_deg)
    out_deg_log = torch.log1p(out_deg)

    # --- Amount normalisation stats ---
    amt_recv = torch.tensor(edges_df["Amount Received"].values, dtype=torch.float32)
    amt_paid = torch.tensor(edges_df["Amount Paid"].values, dtype=torch.float32)
    amt_recv_log = torch.log1p(amt_recv)
    amt_paid_log = torch.log1p(amt_paid)

    return {
        "in_deg_log": in_deg_log,
        "out_deg_log": out_deg_log,
        "amt_recv_mean": amt_recv_log.mean().item(),
        "amt_recv_std":  amt_recv_log.std().item(),
        "amt_paid_mean": amt_paid_log.mean().item(),
        "amt_paid_std":  amt_paid_log.std().item(),
        "num_currency_cats": 15,
        "num_format_cats": 7,
    }


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------

def extract_subgraph(
    edges_df: pd.DataFrame,
    seed_node: int,
    hops: int = 1,
    max_edges: int = 2000,
) -> Tuple[pd.DataFrame, set]:
    """
    Extract the k-hop neighborhood of *seed_node* from the full edge list.

    Returns:
      sub_edges : DataFrame of edges in the neighborhood
      node_set  : set of all node IDs appearing in the subgraph
    """
    node_set = {seed_node}
    frontier = {seed_node}

    for _ in range(hops):
        mask_src = edges_df["src"].isin(frontier)
        mask_dst = edges_df["dst"].isin(frontier)
        hop_edges = edges_df[mask_src | mask_dst]

        if len(hop_edges) == 0:
            break

        new_nodes = set(hop_edges["src"].unique()) | set(hop_edges["dst"].unique())
        frontier = new_nodes - node_set
        node_set |= new_nodes

        if len(node_set) > 5000:
            break

    # Collect all edges where both endpoints are in node_set
    mask = edges_df["src"].isin(node_set) & edges_df["dst"].isin(node_set)
    sub_edges = edges_df[mask].copy()

    if len(sub_edges) > max_edges:
        sub_edges = sub_edges.sample(n=max_edges, random_state=42)

    return sub_edges, node_set


# ---------------------------------------------------------------------------
# Feature preparation for inference
# ---------------------------------------------------------------------------

def prepare_subgraph_features(
    sub_edges: pd.DataFrame,
    node_set: set,
    global_stats: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Given a subgraph edge DataFrame and the set of nodes, build:
      - x          : (num_local_nodes, 2) node features
      - edge_index : (2, num_local_edges) remapped edge indices
      - edge_attr  : (num_local_edges, 39) normalised edge features
      - id_map     : {global_node_id → local_idx} for remapping
    """
    # --- Node ID → local index mapping ---
    sorted_nodes = sorted(node_set)
    id_map = {gid: lid for lid, gid in enumerate(sorted_nodes)}
    num_local_nodes = len(sorted_nodes)

    # --- Node features (degree-based) ---
    in_deg_log = global_stats["in_deg_log"]
    out_deg_log = global_stats["out_deg_log"]
    node_ids_t = torch.tensor(sorted_nodes, dtype=torch.long)
    x = torch.stack([
        in_deg_log[node_ids_t],
        out_deg_log[node_ids_t],
    ], dim=-1)  # (num_local_nodes, 2)

    # --- Edge index (remapped) ---
    src_local = sub_edges["src"].map(id_map).values
    dst_local = sub_edges["dst"].map(id_map).values
    edge_index = torch.tensor(
        np.stack([src_local, dst_local], axis=0),
        dtype=torch.long,
    )  # (2, E_local)

    # --- Edge features ---
    feat_list = []

    # Continuous amounts: log1p → z-score
    amt_recv = torch.tensor(
        sub_edges["Amount Received"].values, dtype=torch.float32
    )
    amt_paid = torch.tensor(
        sub_edges["Amount Paid"].values, dtype=torch.float32
    )
    amt_recv_log = torch.log1p(amt_recv)
    amt_paid_log = torch.log1p(amt_paid)
    amt_recv_norm = (
        (amt_recv_log - global_stats["amt_recv_mean"])
        / (global_stats["amt_recv_std"] + 1e-8)
    )
    amt_paid_norm = (
        (amt_paid_log - global_stats["amt_paid_mean"])
        / (global_stats["amt_paid_std"] + 1e-8)
    )
    feat_list.append(amt_recv_norm.unsqueeze(-1))
    feat_list.append(amt_paid_norm.unsqueeze(-1))

    # Categorical: one-hot
    for col, num_cats in [
        ("Receiving Currency", global_stats["num_currency_cats"]),
        ("Payment Currency",   global_stats["num_currency_cats"]),
        ("Payment Format",     global_stats["num_format_cats"]),
    ]:
        codes = torch.tensor(sub_edges[col].values, dtype=torch.long)
        one_hot = F.one_hot(codes, num_classes=num_cats).float()
        feat_list.append(one_hot)

    edge_attr = torch.cat(feat_list, dim=-1)  # (E_local, 39)

    return x, edge_index, edge_attr, id_map


# ---------------------------------------------------------------------------
# PyVis graph rendering
# ---------------------------------------------------------------------------

def build_pyvis_graph(
    sub_edges: pd.DataFrame,
    id_map: dict,
    probs: np.ndarray,
    seed_node: int,
    nodes_df: pd.DataFrame,
    height: str = "600px",
) -> str:
    """
    Build an interactive PyVis network HTML string.

    Nodes are sized by degree in the subgraph.
    Edges are colored by laundering probability:
      - RED    if prob >= 0.5 or ground-truth label == 1
      - ORANGE if 0.1 <= prob < 0.5
      - GRAY   otherwise
    """
    net = Network(height=height, width="100%", directed=True)
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
        "smooth": { "type": "continuous" }
      }
    }
    """)

    # --- Node degree in subgraph ---
    src_counts = sub_edges["src"].value_counts().to_dict()
    dst_counts = sub_edges["dst"].value_counts().to_dict()
    degree = {}
    for n in id_map:
        degree[n] = src_counts.get(n, 0) + dst_counts.get(n, 0)
    max_deg = max(degree.values()) if degree else 1

    # Account lookup dict
    acc_lookup = dict(zip(nodes_df["node_id"], nodes_df["account"]))

    # --- Add nodes ---
    for gid, lid in id_map.items():
        deg = degree.get(gid, 0)
        size = 5 + 30 * (deg / max(max_deg, 1))
        is_seed = gid == seed_node
        color = "#FFD700" if is_seed else "#4A90D9"
        label = f"Node {gid}" if is_seed else str(gid)
        title = (
            f"<b>Account:</b> {acc_lookup.get(gid, 'N/A')}<br>"
            f"<b>Node ID:</b> {gid}<br>"
            f"<b>Degree (subgraph):</b> {deg}"
        )
        net.add_node(
            lid,
            label=label,
            title=title,
            color=color,
            size=size,
            borderWidth=3 if is_seed else 1,
            borderWidthSelected=5,
        )

    # --- Add edges ---
    for i, (_, row) in enumerate(sub_edges.iterrows()):
        src_lid = id_map[row["src"]]
        dst_lid = id_map[row["dst"]]
        prob = probs[i]
        is_laundering = int(row["is_laundering"])

        if is_laundering == 1 or prob >= 0.5:
            color = "red"
            width = 3
        elif prob >= 0.1:
            color = "orange"
            width = 2
        else:
            color = "rgba(180,180,180,0.4)"
            width = 1

        title = (
            f"<b>Amount Received:</b> ${row['Amount Received']:,.2f}<br>"
            f"<b>Amount Paid:</b> ${row['Amount Paid']:,.2f}<br>"
            f"<b>Laundering Label:</b> {is_laundering}<br>"
            f"<b>Model Score:</b> {prob:.4f}"
        )
        net.add_edge(
            src_lid, dst_lid,
            title=title,
            color=color,
            width=width,
            arrows="to",
        )

    return net.generate_html()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Enterprise AML Investigation Center",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Custom dark theme via CSS ---
    st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .metric-card {
        background: #1A1C23;
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
        border-left: 3px solid #4A90D9;
    }
    .metric-value { font-size: 22px; font-weight: 700; color: #4A90D9; }
    .metric-label { font-size: 12px; color: #8B949E; text-transform: uppercase; }
    .suspicious { color: #FF4444; font-weight: 600; }
    .clean { color: #4CAF50; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Enterprise AML Investigation Center")
    st.caption("Graph Neural Network — Money Laundering Detection")

    # --- Load assets ---
    with st.spinner("Loading model and data..."):
        model = load_model()
        edges_df = load_edges()
        nodes_df = load_nodes()
        metrics = load_metrics()
        global_stats = compute_global_stats(edges_df)

    # ===================================================================
    # Sidebar — KPI cards
    # ===================================================================
    with st.sidebar:
        st.header("Model Performance")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUPRC", f"{metrics['test_auprc']:.4f}")
            st.metric("Precision", f"{metrics['test_precision']:.4f}")
        with col2:
            st.metric("Recall", f"{metrics['test_recall']:.4f}")
            st.metric("F1-Score", f"{metrics['test_f1']:.4f}")

        st.divider()
        st.header("Dataset Statistics")
        st.metric("Total Nodes", f"{metrics['total_nodes']:,}")
        st.metric("Total Edges", f"{metrics['total_edges']:,}")
        st.metric("Fraud Rate", f"{metrics['fraud_rate_pct']:.4f}%")
        st.metric("Pos Weight", f"{metrics['pos_weight']:.1f}")

        st.divider()
        st.header("Model Info")
        st.text(f"Architecture: {metrics['model']}")
        st.text(f"Epochs trained: {metrics['epochs_trained']}")
        st.text(f"Device: {metrics['device']}")
        st.text(f"Best Val AUPRC: {metrics['best_val_auprc']:.4f}")

        st.divider()
        st.caption("Built with PyTorch Geometric + Streamlit")

    # ===================================================================
    # Main area — Search & Investigation
    # ===================================================================
    st.header("Account Investigation")

    # --- Search mode selector ---
    search_mode = st.radio(
        "Search by:",
        ["Node ID (integer)", "Account Hash (string)"],
        horizontal=True,
    )

    if search_mode == "Node ID (integer)":
        node_input = st.text_input(
            "Enter Node ID (0 – 515079)",
            placeholder="e.g. 11214",
            help="Integer ID from the node mapping table.",
        )
        try:
            seed_node = int(node_input) if node_input.strip() else None
        except ValueError:
            seed_node = None
            if node_input.strip():
                st.warning("Please enter a valid integer Node ID.")
    else:
        account_input = st.text_input(
            "Enter Account Hash",
            placeholder="e.g. 800737690",
            help="Account hash string from the transaction data.",
        )
        if account_input.strip():
            match = nodes_df[nodes_df["account"] == account_input.strip()]
            if len(match) > 0:
                seed_node = int(match.iloc[0]["node_id"])
                st.success(f"Found Node ID: {seed_node}")
            else:
                seed_node = None
                st.error("Account hash not found in node mapping.")
        else:
            seed_node = None

    # --- Hop selection ---
    hops = st.slider("Neighborhood hops", 1, 2, 1,
                     help="1-hop = direct transactions; 2-hop = includes neighbors of neighbors.")

    if seed_node is not None and st.button("Investigate", type="primary"):
        with st.spinner(f"Extracting {hops}-hop neighborhood for Node {seed_node}..."):
            sub_edges, node_set = extract_subgraph(edges_df, seed_node, hops=hops)

        if len(sub_edges) == 0:
            st.warning("No transactions found for this node.")
            return

        st.success(
            f"Subgraph extracted: **{len(node_set):,}** nodes, "
            f"**{len(sub_edges):,}** edges"
        )

        # --- Run GNN inference ---
        with st.spinner("Running GNN inference on subgraph..."):
            x, edge_index, edge_attr, id_map = prepare_subgraph_features(
                sub_edges, node_set, global_stats
            )
            with torch.no_grad():
                logits = model(x, edge_index, edge_attr)
                probs = torch.sigmoid(logits).numpy()

        # --- Tabs for graph + table ---
        tab_graph, tab_table = st.tabs(["Network Graph", "Transaction Table"])

        with tab_graph:
            st.subheader("Transaction Network")
            st.caption(
                "Red edges = high laundering risk (model score ≥ 0.5 or known label). "
                "Orange = moderate risk (0.1–0.5). Gray = low risk."
            )

            html = build_pyvis_graph(
                sub_edges, id_map, probs, seed_node, nodes_df,
                height="650px",
            )
            st.components.v1.html(html, height=680, scrolling=True)

            # Legend
            cols = st.columns(4)
            cols[0].markdown("🔴 **High Risk** (score ≥ 0.5)")
            cols[1].markdown("🟠 **Moderate Risk** (0.1–0.5)")
            cols[2].markdown("⚪ **Low Risk** (< 0.1)")
            cols[3].markdown("🟡 **Seed Node**")

        with tab_table:
            st.subheader("Suspicious Transactions")

            # Attach model scores to the edge DataFrame
            display_df = sub_edges.copy()
            display_df["model_score"] = probs
            display_df["predicted_risk"] = np.where(
                probs >= 0.5, "HIGH",
                np.where(probs >= 0.1, "MEDIUM", "LOW"),
            )

            # Map node IDs to account hashes for readability
            acc_map = dict(zip(nodes_df["node_id"], nodes_df["account"]))

            # Show only suspicious (ground-truth or high model score)
            suspicious = display_df[
                (display_df["is_laundering"] == 1) | (display_df["model_score"] >= 0.5)
            ].copy()

            if len(suspicious) == 0:
                st.info("No suspicious transactions detected in this neighborhood.")
            else:
                suspicious["Sender Account"] = suspicious["src"].map(acc_map)
                suspicious["Receiver Account"] = suspicious["dst"].map(acc_map)

                st.dataframe(
                    suspicious[[
                        "Sender Account", "Receiver Account",
                        "Amount Received", "Amount Paid",
                        "is_laundering", "model_score", "predicted_risk",
                    ]].sort_values("model_score", ascending=False),
                    use_container_width=True,
                    column_config={
                        "model_score": st.column_config.NumberColumn(format="%.4f"),
                        "Amount Received": st.column_config.NumberColumn(format="$%.2f"),
                        "Amount Paid": st.column_config.NumberColumn(format="$%.2f"),
                        "is_laundering": st.column_config.NumberColumn(
                            format="%d", width="small"
                        ),
                    },
                )

                # Summary stats
                known_fraud = suspicious["is_laundering"].sum()
                high_risk = (suspicious["predicted_risk"] == "HIGH").sum()
                st.metric("Known Laundering", int(known_fraud))
                st.metric("Model High-Risk Flags", int(high_risk))

            # Option to view all edges
            with st.expander("View All Transactions in Neighborhood"):
                display_df["Sender Account"] = display_df["src"].map(acc_map)
                display_df["Receiver Account"] = display_df["dst"].map(acc_map)
                st.dataframe(
                    display_df[[
                        "Sender Account", "Receiver Account",
                        "Amount Received", "Amount Paid",
                        "is_laundering", "model_score", "predicted_risk",
                    ]].sort_values("model_score", ascending=False),
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
