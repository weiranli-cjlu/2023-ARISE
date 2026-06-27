import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

try:
    from torch_geometric.utils import from_scipy_sparse_matrix
except Exception:  # torch_geometric is recommended, but keep a small fallback for portability.
    from_scipy_sparse_matrix = None


DEFAULT_DATA_DIR = "~/datasets/GAD/mat"


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        return [to_tuple(mx) for mx in sparse_mx]
    return to_tuple(sparse_mx)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation."""
    if not sp.issparse(features):
        features = sp.csr_matrix(features)
    rowsum = np.array(features.sum(1), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_inv = np.power(rowsum, -1).flatten()
    r_inv[~np.isfinite(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(features.todense(), dtype=np.float32), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj, dtype=np.float32)
    rowsum = np.array(adj.sum(1), dtype=np.float32)
    adj_raw = adj
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().astype(np.float32)
    return norm_adj, adj_raw.astype(np.float32)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def _get_first_key(data: Dict, keys: Sequence[str], required: bool = True):
    for key in keys:
        if key in data:
            return data[key]
    if required:
        raise KeyError(f"None of keys {keys} found in .mat file. Available keys: {list(data.keys())}")
    return None


def _resolve_mat_path(dataset: str, data_dir: Optional[str] = None) -> Path:
    """Resolve dataset path. Default location is ~/datasets/GAD/mat/<dataset>.mat."""
    data_root = Path(os.path.expanduser(data_dir or DEFAULT_DATA_DIR))
    dataset_path = Path(os.path.expanduser(dataset))
    if dataset_path.suffix == ".mat" and dataset_path.exists():
        return dataset_path

    mat_path = data_root / f"{dataset_path.stem}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Cannot find dataset file: {mat_path}\n"
            f"Please put .mat files under {data_root} or pass --data_dir /path/to/mat_dir."
        )
    return mat_path


def load_mat(dataset, data_dir: Optional[str] = None, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset from ~/datasets/GAD/mat by default.

    Compatible field names:
      - adjacency: Network / A / adj
      - attributes: Attributes / X / attr / features
      - anomaly label: Label / gnd / label / y
      - class label: Class / class / labels. If missing, zeros are used because ARISE
        does not use class labels during training/testing.
    """
    mat_path = _resolve_mat_path(dataset, data_dir)
    data = sio.loadmat(str(mat_path))

    label = _get_first_key(data, ["Label", "gnd", "label", "y"])
    attr = _get_first_key(data, ["Attributes", "X", "attr", "features", "Feat"])
    network = _get_first_key(data, ["Network", "A", "adj"])

    adj = sp.csr_matrix(network, dtype=np.float32)
    feat = sp.csr_matrix(attr, dtype=np.float32)

    class_raw = _get_first_key(data, ["Class", "class", "labels"], required=False)
    if class_raw is not None:
        labels_dense = np.squeeze(np.array(class_raw, dtype=np.int64))
        # Some datasets are 1-based, while others are already 0-based.
        if labels_dense.size > 0 and labels_dense.min() == 1:
            labels_dense = labels_dense - 1
        num_classes = int(np.max(labels_dense)) + 1 if labels_dense.size > 0 else 1
        labels = dense_to_one_hot(labels_dense, num_classes)
    else:
        labels = dense_to_one_hot(np.zeros(adj.shape[0], dtype=np.int64), 1)

    ano_labels = np.squeeze(np.array(label)).astype(np.int64)
    if "str_anomaly_label" in data:
        str_ano_labels = np.squeeze(np.array(data["str_anomaly_label"]))
        attr_ano_labels = np.squeeze(np.array(data["attr_anomaly_label"]))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train:num_train + num_val]
    idx_test = all_idx[num_train + num_val:]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels


def adj_to_edge_index(adj: sp.spmatrix) -> torch.Tensor:
    """Convert scipy adjacency matrix to PyG-style edge_index without using DGL."""
    adj = sp.coo_matrix(adj)
    if from_scipy_sparse_matrix is not None:
        edge_index, _ = from_scipy_sparse_matrix(adj)
        return edge_index.long()

    row = torch.from_numpy(adj.row).long()
    col = torch.from_numpy(adj.col).long()
    return torch.stack([row, col], dim=0)


def build_neighbor_dict(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Build CPU neighbor lists from a PyG edge_index tensor."""
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    for u, v in zip(src, dst):
        if u != v:
            neighbors[u].append(v)
    return neighbors


def _single_rwr_trace(seed: int, neighbors: List[List[int]], restart_prob: float, max_steps: int) -> List[int]:
    """A lightweight random-walk-with-restart sampler implemented with Python/Torch data.

    The implementation deliberately stays on CPU because subgraph generation is a
    sampling/preprocessing step and avoids DGL CUDA compatibility issues on RTX 50 series.
    """
    cur = seed
    trace = [seed]
    for _ in range(max_steps - 1):
        if random.random() < restart_prob:
            cur = seed
        nbrs = neighbors[cur]
        if len(nbrs) == 0:
            cur = seed
        else:
            cur = nbrs[random.randrange(len(nbrs))]
        trace.append(cur)
    return trace


def _unique_keep_order(nodes: Sequence[int]) -> List[int]:
    seen = set()
    out = []
    for n in nodes:
        if n not in seen:
            seen.add(n)
            out.append(int(n))
    return out


def generate_rwr_subgraph(edge_index_or_neighbors, num_nodes: Optional[int] = None, subgraph_size: int = 4,
                          restart_prob: float = 0.9) -> List[List[int]]:
    """Generate per-node RWR subgraphs without DGL.

    Args:
        edge_index_or_neighbors: PyG edge_index tensor with shape [2, E] or prebuilt neighbor list.
        num_nodes: number of nodes, required when edge_index is given.
        subgraph_size: returned subgraph size. The seed node is always placed at the last position
            to preserve ARISE's original tensor layout.
        restart_prob: restart probability for retry walks.

    Returns:
        A list whose i-th element contains ``subgraph_size`` node ids. The last id is i.
    """
    if isinstance(edge_index_or_neighbors, torch.Tensor):
        if num_nodes is None:
            num_nodes = int(edge_index_or_neighbors.max().item()) + 1
        neighbors = build_neighbor_dict(edge_index_or_neighbors, num_nodes)
    else:
        neighbors = edge_index_or_neighbors
        if num_nodes is None:
            num_nodes = len(neighbors)

    reduced_size = subgraph_size - 1
    subgraphs: List[List[int]] = []

    for seed in range(num_nodes):
        # First pass: low restart probability collects local context more effectively than DGL's
        # deprecated contrib sampler on modern software stacks.
        trace = _single_rwr_trace(seed, neighbors, restart_prob=restart_prob, max_steps=max(subgraph_size * 5, 8))
        nodes = [n for n in _unique_keep_order(trace) if n != seed]

        retry_time = 0
        while len(nodes) < reduced_size and retry_time < 10:
            trace = _single_rwr_trace(seed, neighbors, restart_prob=restart_prob, max_steps=max(subgraph_size * 8, 16))
            nodes = [n for n in _unique_keep_order(trace) if n != seed]
            retry_time += 1

        if len(nodes) < reduced_size:
            # Isolated or tiny-component nodes: pad with the seed itself. This keeps tensor shapes
            # valid and mirrors the original code's repeated-node fallback.
            nodes = (nodes + [seed] * reduced_size)[:reduced_size]
        else:
            nodes = nodes[:reduced_size]

        nodes.append(seed)
        subgraphs.append(nodes)

    return subgraphs


def build_weight_lookup(adj: sp.spmatrix) -> List[Dict[int, float]]:
    """Build row-wise sparse weight dictionaries for tiny subgraph extraction.

    This avoids materializing a dense N*N adjacency matrix on CPU/GPU.  The
    returned object is optimized for queries like weight_lookup[u].get(v) where
    each mini-batch only needs a few node pairs.
    """
    adj = adj.tocsr().astype(np.float32)
    lookup: List[Dict[int, float]] = []
    for row in range(adj.shape[0]):
        start, end = adj.indptr[row], adj.indptr[row + 1]
        cols = adj.indices[start:end]
        vals = adj.data[start:end]
        lookup.append({int(c): float(v) for c, v in zip(cols, vals)})
    return lookup
