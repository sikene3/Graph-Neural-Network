#!/usr/bin/env python3
"""
train_gnn.py — GraphSAGE Edge Classification for AML Detection
==============================================================

Loads the preprocessed nodes_mapped.parquet and edges_labeled.parquet,
constructs a PyG Data object, and trains a GraphSAGE encoder + MLP
edge-classifier to detect money-laundering transactions.

Key design decisions for the ~5 M edge / ~515 k node scale:
  - RandomNodeLoader for subgraph mini-batch training (no pyg-lib needed).
  - BCEWithLogitsLoss with pos_weight to handle ~0.06 % fraud rate.
  - Precision, Recall, F1, and AUPRC as evaluation metrics.
  - Early stopping on validation AUPRC.
  - Saves best model to aml_gnn_model.pth.

Usage:
  python train_gnn.py
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import SAGEConv
from torchmetrics import (
    AveragePrecision,
    Precision,
    Recall,
    F1Score,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NODES_PATH   = Path("nodes_mapped.parquet")
EDGES_PATH   = Path("edges_labeled.parquet")
MODEL_PATH   = Path("aml_gnn_model.pth")

# GNN architecture
HIDDEN_DIM       = 64
ENCODER_OUT_DIM  = 64
NUM_CONV_LAYERS  = 2
DROPOUT          = 0.3

# Edge predictor MLP
PRED_HIDDEN      = 128

# Training
NUM_EPOCHS       = 50
NUM_PARTS        = 200           # number of subgraph partitions per epoch
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-5
PATIENCE         = 10            # early stopping

# Data split ratios
TRAIN_RATIO      = 0.8
VAL_RATIO        = 0.1
TEST_RATIO       = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_gnn")

# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_and_prepare_data() -> Data:
    """
    Reads nodes_mapped.parquet and edges_labeled.parquet, builds node
    features (degree-based), one-hot encodes categorical edge features,
    normalises continuous edge features, and returns a PyG Data object
    with train/val/test edge masks.
    """
    log.info("Loading nodes from %s ...", NODES_PATH)
    nodes_df = pd.read_parquet(NODES_PATH)
    num_nodes = len(nodes_df)
    log.info("  → %d nodes loaded.", num_nodes)

    log.info("Loading edges from %s ...", EDGES_PATH)
    edges_df = pd.read_parquet(EDGES_PATH)
    num_edges = len(edges_df)
    log.info("  → %d edges loaded.", num_edges)

    # --- Edge index ---
    src = torch.tensor(edges_df["src"].values, dtype=torch.long)
    dst = torch.tensor(edges_df["dst"].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)  # (2, E)

    # --- Edge labels ---
    y = torch.tensor(edges_df["is_laundering"].values, dtype=torch.float32)

    # --- Node features (degree-based) ---
    in_deg = torch.zeros(num_nodes, dtype=torch.float32)
    out_deg = torch.zeros(num_nodes, dtype=torch.float32)
    ones = torch.ones(num_edges, dtype=torch.float32)
    in_deg.scatter_add_(0, dst, ones)
    out_deg.scatter_add_(0, src, ones)

    # Log-scale to compress range, then stack
    in_deg = torch.log1p(in_deg).unsqueeze(-1)
    out_deg = torch.log1p(out_deg).unsqueeze(-1)
    x = torch.cat([in_deg, out_deg], dim=-1)  # (N, 2)
    # Will be projected to HIDDEN_DIM by the model's node_encoder

    # --- Edge features ---
    edge_feat_list = []

    # Continuous: log-transform amounts, then z-score normalise
    for col in ["Amount Received", "Amount Paid"]:
        vals = torch.tensor(edges_df[col].values, dtype=torch.float32)
        vals = torch.log1p(vals)
        mean, std = vals.mean(), vals.std()
        vals = (vals - mean) / (std + 1e-8)
        edge_feat_list.append(vals.unsqueeze(-1))

    # Categorical: one-hot encode (codes stored as int16 in parquet)
    cat_cols = ["Receiving Currency", "Payment Currency", "Payment Format"]
    for col in cat_cols:
        codes = torch.tensor(edges_df[col].values, dtype=torch.long)
        num_cats = int(codes.max().item()) + 1
        one_hot = F.one_hot(codes, num_classes=num_cats).float()
        edge_feat_list.append(one_hot)

    edge_attr = torch.cat(edge_feat_list, dim=-1)  # (E, edge_dim)
    log.info("  → Edge attribute dimension: %d", edge_attr.size(1))

    # --- Train / val / test edge masks ---
    perm = torch.randperm(num_edges)
    train_end = int(TRAIN_RATIO * num_edges)
    val_end   = train_end + int(VAL_RATIO * num_edges)

    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask   = torch.zeros(num_edges, dtype=torch.bool)
    test_mask  = torch.zeros(num_edges, dtype=torch.bool)

    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    # --- Assemble Data object ---
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
    )

    # --- Log split statistics ---
    pos_train = y[train_mask].sum().item()
    pos_val   = y[val_mask].sum().item()
    pos_test  = y[test_mask].sum().item()
    log.info("=" * 55)
    log.info("DATA SPLIT SUMMARY")
    log.info("  Total nodes  : %d", num_nodes)
    log.info("  Total edges  : %d", num_edges)
    log.info("  Train edges  : %d  (pos: %d, neg: %d)",
             train_mask.sum().item(), pos_train, train_mask.sum().item() - pos_train)
    log.info("  Val edges    : %d  (pos: %d, neg: %d)",
             val_mask.sum().item(), pos_val, val_mask.sum().item() - pos_val)
    log.info("  Test edges   : %d  (pos: %d, neg: %d)",
             test_mask.sum().item(), pos_test, test_mask.sum().item() - pos_test)
    log.info("  Fraud rate   : %.4f %%", 100 * y.sum().item() / num_edges)
    log.info("=" * 55)

    return data


# ---------------------------------------------------------------------------
# 2. Model definition
# ---------------------------------------------------------------------------

class AMLGraphSAGE(nn.Module):
    """
    GraphSAGE encoder + MLP edge classifier.

    Encoder:
      node_encoder : Linear(2, HIDDEN_DIM)  — projects degree features
      conv1        : SAGEConv(HIDDEN_DIM, HIDDEN_DIM * 2)
      conv2        : SAGEConv(HIDDEN_DIM * 2, ENCODER_OUT_DIM)

    Predictor:
      MLP that takes [src_emb || dst_emb || edge_attr] and outputs a
      single logit for binary classification.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        encoder_out_dim: int = ENCODER_OUT_DIM,
        pred_hidden: int = PRED_HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.dropout = dropout

        # Project raw node features (degree) to hidden_dim
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        # GNN layers
        self.conv1 = SAGEConv(hidden_dim, hidden_dim * 2)
        self.conv2 = SAGEConv(hidden_dim * 2, encoder_out_dim)

        # Edge predictor: [src(64) || dst(64) || edge_attr(E)] → logit
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
        """Run the GNN encoder, return node embeddings."""
        x = self.node_encoder(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass: encode nodes, then predict logits for every
        edge in the batch.  Returns (E_batch,) tensor of logits.
        """
        h = self.encode(x, edge_index)

        src_h = h[edge_index[0]]   # (E_batch, encoder_out_dim)
        dst_h = h[edge_index[1]]   # (E_batch, encoder_out_dim)

        edge_repr = torch.cat([src_h, dst_h, edge_attr], dim=-1)
        logits = self.predictor(edge_repr).squeeze(-1)  # (E_batch,)
        return logits


# ---------------------------------------------------------------------------
# 3. Data loaders
# ---------------------------------------------------------------------------

def create_loaders(data: Data) -> Tuple[RandomNodeLoader, RandomNodeLoader, RandomNodeLoader]:
    """
    Creates RandomNodeLoader instances for train / val / test.

    RandomNodeLoader partitions the graph into subgraphs by randomly
    assigning nodes to partitions.  Each batch is a subgraph containing
    a subset of nodes and all edges between them.  Edge masks are
    preserved, so we can compute loss only on train_mask edges.

    Train loader shuffles partitions each epoch; val/test are fixed.
    """
    train_loader = RandomNodeLoader(data, num_parts=NUM_PARTS, shuffle=True)
    val_loader   = RandomNodeLoader(data, num_parts=NUM_PARTS, shuffle=False)
    test_loader  = RandomNodeLoader(data, num_parts=NUM_PARTS, shuffle=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 4. Training & evaluation utilities
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: AMLGraphSAGE,
    loader: RandomNodeLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train one epoch, return average loss over train_mask edges."""
    model.train()
    total_loss = 0.0
    total_edges = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        mask = batch.train_mask
        if mask.sum() == 0:
            continue

        loss = criterion(logits[mask], batch.y[mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        total_edges += mask.sum().item()

    return total_loss / max(total_edges, 1)


@torch.no_grad()
def evaluate(
    model: AMLGraphSAGE,
    loader: RandomNodeLoader,
    mask_name: str,
    device: torch.device,
) -> dict:
    """
    Evaluate the model on all batches from *loader*, collecting
    predictions and labels for edges where *mask_name* is True.
    Returns a dict of metrics.
    """
    model.eval()

    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        mask = getattr(batch, mask_name)  # e.g. batch.val_mask
        if mask.sum() == 0:
            continue

        probs = torch.sigmoid(logits[mask])
        all_preds.append(probs.cpu())
        all_labels.append(batch.y[mask].cpu().long())

    if not all_preds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auprc": 0.0}

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    precision = Precision(task="binary")(preds, labels).item()
    recall    = Recall(task="binary")(preds, labels).item()
    f1        = F1Score(task="binary")(preds, labels).item()
    auprc     = AveragePrecision(task="binary")(preds, labels).item()

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "auprc":     auprc,
    }


# ---------------------------------------------------------------------------
# 5. Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== AML GNN Training ===")
    log.info("Device: %s", DEVICE)

    # --- Load & prepare data ---
    data = load_and_prepare_data()

    # --- Compute pos_weight for BCEWithLogitsLoss ---
    num_pos = data.y.sum().item()
    num_neg = data.num_edges - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], device=DEVICE)
    log.info("pos_weight for BCE loss: %.1f", pos_weight.item())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Create model ---
    model = AMLGraphSAGE(
        node_feat_dim=data.x.size(1),
        edge_feat_dim=data.edge_attr.size(1),
    ).to(DEVICE)

    log.info("Model parameters: %d",
             sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --- Create loaders ---
    train_loader, val_loader, test_loader = create_loaders(data)

    # --- Training loop with early stopping ---
    best_auprc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, "val_mask", DEVICE)

        log.info(
            "Epoch %3d | Loss: %.4f | "
            "Val P: %.4f R: %.4f F1: %.4f AUPRC: %.4f",
            epoch, train_loss,
            val_metrics["precision"], val_metrics["recall"],
            val_metrics["f1"], val_metrics["auprc"],
        )

        # Early stopping on AUPRC
        if val_metrics["auprc"] > best_auprc:
            best_auprc = val_metrics["auprc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            log.info("  ↑ New best model (AUPRC=%.4f)", best_auprc)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("Early stopping after %d epochs without improvement.", PATIENCE)
                break

    # --- Restore best model & evaluate on test set ---
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, "test_mask", DEVICE)

    log.info("=" * 55)
    log.info("FINAL TEST RESULTS")
    log.info("  Precision : %.4f", test_metrics["precision"])
    log.info("  Recall    : %.4f", test_metrics["recall"])
    log.info("  F1-Score  : %.4f", test_metrics["f1"])
    log.info("  AUPRC     : %.4f", test_metrics["auprc"])
    log.info("=" * 55)

    # --- Save model ---
    torch.save(model.state_dict(), MODEL_PATH)
    log.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
