#!/usr/bin/env python3
"""
parse_aml_graph.py — Memory-Efficient IBM AML Tabular-to-Graph Converter
=======================================================================

Converts the HI-Small AML transaction CSV (~5M rows) into a graph
representation suitable for PyTorch Geometric edge-classification GNNs.

Outputs:
  nodes_mapped.parquet  — (node_id, account_string) mapping table
  edges_labeled.parquet — (src, dst, edge_attr..., is_laundering) edge table

Design principles:
  - Chunked CSV reading with tight dtypes to stay under ~2 GB RAM.
  - PyArrow-backed string columns for compact storage and fast hashing.
  - Two-pass approach: pass 1 collects unique accounts → builds mapping;
    pass 2 streams edges, maps accounts to ints, labels from Patterns.txt.
  - Parquet output with Snappy compression for disk efficiency.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATH      = Path("HI-Small_Trans.csv")
PATTERNS_PATH = Path("HI-Small_Patterns.txt")
NODES_OUT     = Path("nodes_mapped.parquet")
EDGES_OUT     = Path("edges_labeled.parquet")

CHUNKSIZE     = 500_000          # rows per chunk — tune for your RAM
LOG_INTERVAL  = 1_000_000        # rows between progress logs

# Dtypes for the CSV columns (tight, memory-conscious)
DTYPE_MAP = {
    "Timestamp":          "string[pyarrow]",
    "From Bank":          "string[pyarrow]",
    "Account":            "string[pyarrow]",
    "To Bank":            "string[pyarrow]",
    "Account.1":          "string[pyarrow]",
    "Amount Received":    "float32",
    "Receiving Currency": "category",
    "Amount Paid":        "float32",
    "Payment Currency":   "category",
    "Payment Format":     "category",
    "Is Laundering":      "int8",
}

# Columns to keep in the final edge parquet
EDGE_FEATURE_COLS = [
    "Amount Received", "Amount Paid",
    "Receiving Currency", "Payment Currency", "Payment Format",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("parse_aml")

# ---------------------------------------------------------------------------
# Step 1 — Build the laundering-label lookup set from Patterns.txt
# ---------------------------------------------------------------------------

def build_pattern_label_set(patterns_path: Path) -> set:
    """
    Reads Patterns.txt (skip the header line), builds a set of composite
    keys that uniquely identify each laundering transaction.

    Composite key format:
        "Timestamp|From Bank|Account|To Bank|Account.1|Amount Received|Receiving Currency"

    Returns a Python set of those keys for O(1) membership testing.
    """
    log.info("Loading laundering patterns from %s ...", patterns_path)

    # Patterns.txt has the same CSV schema as the main file, with a
    # descriptive header line that we skip.
    pat_df = pd.read_csv(
        patterns_path,
        skiprows=1,
        header=None,
        names=list(DTYPE_MAP.keys()),
        dtype={k: v for k, v in DTYPE_MAP.items() if k != "Is Laundering"},
    )

    # Build composite key: all fields that together identify a row
    def _make_key(row: pd.Series) -> str:
        return (
            f"{row['Timestamp']}|"
            f"{row['From Bank']}|"
            f"{row['Account']}|"
            f"{row['To Bank']}|"
            f"{row['Account.1']}|"
            f"{row['Amount Received']:.2f}|"
            f"{row['Receiving Currency']}"
        )

    keys = set()
    for _, row in pat_df.iterrows():
        keys.add(_make_key(row))

    log.info("  → %d unique laundering-pattern keys built.", len(keys))
    return keys


# ---------------------------------------------------------------------------
# Step 2 — First pass: collect all unique account strings
# ---------------------------------------------------------------------------

def collect_unique_accounts(csv_path: Path) -> pd.array:
    """
    Streams the CSV in chunks, collecting every distinct account string
    from both the 'Account' (sender) and 'Account.1' (receiver) columns.

    Returns a PyArrow-backed StringArray of unique accounts.
    """
    log.info("Pass 1/2 — Collecting unique accounts from %s ...", csv_path)

    reader = pd.read_csv(
        csv_path,
        chunksize=CHUNKSIZE,
        dtype=DTYPE_MAP,
        usecols=["Account", "Account.1"],
    )

    unique_set: set[str] = set()
    total_rows = 0

    for chunk in reader:
        # Drop NAs (shouldn't exist, but be safe)
        senders = chunk["Account"].dropna().tolist()
        receivers = chunk["Account.1"].dropna().tolist()
        unique_set.update(senders)
        unique_set.update(receivers)

        total_rows += len(chunk)
        if total_rows % LOG_INTERVAL < CHUNKSIZE:
            log.info("  … scanned %d rows, %d unique accounts so far.",
                     total_rows, len(unique_set))

    log.info("  → %d total rows scanned, %d unique accounts found.",
             total_rows, len(unique_set))

    # Convert to PyArrow-backed StringArray for compact storage
    return pd.array(list(unique_set), dtype="string[pyarrow]")


# ---------------------------------------------------------------------------
# Step 3 — Build node mapping and save nodes_mapped.parquet
# ---------------------------------------------------------------------------

def build_and_save_nodes(accounts: pd.array, output_path: Path) -> dict:
    """
    Given a sorted array of unique account strings, assigns each a
    sequential integer ID (0 … N-1), saves the mapping as a Parquet file,
    and returns a Python dict {account_str → node_id} for fast lookups.
    """
    log.info("Building node mapping for %d accounts ...", len(accounts))

    # Sort for deterministic ordering
    sorted_acc = pd.array(sorted(accounts), dtype="string[pyarrow]")
    node_ids = np.arange(len(sorted_acc), dtype=np.int32)

    nodes_df = pd.DataFrame({
        "node_id":  node_ids,
        "account":  sorted_acc,
    })
    nodes_df.to_parquet(output_path, compression="snappy", index=False)
    log.info("  → Saved %s (%d nodes).", output_path, len(nodes_df))

    # Build dict for O(1) lookup during edge streaming
    mapping = {str(acc): int(nid) for acc, nid in zip(sorted_acc, node_ids)}
    return mapping


# ---------------------------------------------------------------------------
# Step 4 — Second pass: stream edges, map accounts, label, save
# ---------------------------------------------------------------------------

def stream_and_save_edges(
    csv_path: Path,
    account_to_id: dict,
    pattern_keys: set,
    output_path: Path,
) -> None:
    """
    Streams the CSV a second time.  For each row:
      - Maps sender & receiver accounts to integer node IDs.
      - Builds a composite key and checks membership in pattern_keys
        to set the is_laundering label.
      - Accumulates edge DataFrames and periodically writes them to
        a single Parquet file (row-group append via directory of files
        that we concatenate at the end, or via streaming Parquet writer).

    Strategy: collect chunks in a list, then pd.concat + to_parquet at the
    end.  With ~5 M edges × ~10 columns of tight dtypes, the final
    DataFrame is ~200 MB — well within RAM.
    """
    log.info("Pass 2/2 — Streaming edges from %s ...", csv_path)

    reader = pd.read_csv(
        csv_path,
        chunksize=CHUNKSIZE,
        dtype=DTYPE_MAP,
    )

    edge_chunks: list[pd.DataFrame] = []
    total_rows = 0
    laundering_count = 0

    for chunk in reader:
        n = len(chunk)
        total_rows += n

        # --- Map accounts to integer node IDs ---
        src_ids = chunk["Account"].map(account_to_id)
        dst_ids = chunk["Account.1"].map(account_to_id)

        # --- Build composite key for pattern matching ---
        composite_key = (
            chunk["Timestamp"].astype(str)
            + "|" + chunk["From Bank"].astype(str)
            + "|" + chunk["Account"].astype(str)
            + "|" + chunk["To Bank"].astype(str)
            + "|" + chunk["Account.1"].astype(str)
            + "|" + chunk["Amount Received"].map("{:.2f}".format, na_action="ignore")
            + "|" + chunk["Receiving Currency"].astype(str)
        )

        is_laundering = composite_key.isin(pattern_keys).astype("int8")
        laundering_count += is_laundering.sum()

        # --- Assemble edge DataFrame ---
        edge_df = pd.DataFrame({
            "src":             src_ids.astype("int32"),
            "dst":             dst_ids.astype("int32"),
            "is_laundering":   is_laundering,
        })

        # Attach edge-feature columns (convert categories to codes for
        # compact storage; the training script can one-hot or embed them)
        for col in EDGE_FEATURE_COLS:
            series = chunk[col]
            if isinstance(series.dtype, pd.CategoricalDtype):
                edge_df[col] = series.cat.codes.astype("int16")
            else:
                edge_df[col] = series

        edge_chunks.append(edge_df)

        if total_rows % LOG_INTERVAL < CHUNKSIZE:
            log.info("  … processed %d rows, %d laundering so far.",
                     total_rows, laundering_count)

    # --- Concatenate and save ---
    log.info("Concatenating %d chunks ...", len(edge_chunks))
    all_edges = pd.concat(edge_chunks, ignore_index=True)
    del edge_chunks  # free memory

    all_edges.to_parquet(output_path, compression="snappy", index=False)
    log.info("  → Saved %s (%d edges).", output_path, len(all_edges))

    # --- Integrity report ---
    legit_count = len(all_edges) - laundering_count
    log.info("=" * 55)
    log.info("DATA INTEGRITY REPORT")
    log.info("  Unique nodes (accounts) : %d", len(account_to_id))
    log.info("  Total edges (transactions): %d", len(all_edges))
    log.info("  Laundering edges          : %d (%.4f%%)",
             laundering_count, 100 * laundering_count / len(all_edges))
    log.info("  Legitimate edges          : %d (%.4f%%)",
             legit_count, 100 * legit_count / len(all_edges))
    log.info("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== IBM AML Graph Parser ===")

    # 1. Build pattern-key lookup set
    pattern_keys = build_pattern_label_set(PATTERNS_PATH)

    # 2. First pass — collect unique accounts
    accounts = collect_unique_accounts(CSV_PATH)

    # 3. Build node mapping & save nodes_mapped.parquet
    account_to_id = build_and_save_nodes(accounts, NODES_OUT)

    # 4. Second pass — stream edges, label, save edges_labeled.parquet
    stream_and_save_edges(CSV_PATH, account_to_id, pattern_keys, EDGES_OUT)

    log.info("Done. Output files: %s, %s", NODES_OUT, EDGES_OUT)


if __name__ == "__main__":
    main()
