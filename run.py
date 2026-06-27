import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange

from model import Model
from utils import (
    adj_to_edge_index,
    build_neighbor_dict,
    build_weight_lookup,
    generate_rwr_subgraph,
    load_mat,
    normalize_adj,
    preprocess_features,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class PreparedData:
    """Data shared by all trials for one dataset.

    The original script reloaded the .mat file, rebuilt neighbors and moved a
    dense N*N adjacency matrix to GPU for every run/trial.  This object keeps
    immutable dataset-level objects once and reuses them across runs.
    """

    adj_lookup: List[dict]
    adj_raw: sp.csr_matrix
    features_cpu: torch.Tensor
    labels: np.ndarray
    degree_ave: float
    nb_nodes: int
    ft_size: int
    neighbor_list: List[List[int]]


def parse_args():
    parser = argparse.ArgumentParser(description="ARISE low-memory PyTorch/PyG-style implementation.")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat",
                        help="Directory containing <dataset>.mat files. Default: ~/datasets/GAD/mat")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=5, help="Target number of trials. Existing completed trials are skipped.")
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--drop_prob", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--subgraph_size", type=int, default=4)
    parser.add_argument("--readout", type=str, default="avg", choices=["max", "min", "avg", "weighted_sum"])
    parser.add_argument("--auc_test_rounds", type=int, default=64,
                        help="Number of stochastic test rounds. Use 256 for final paper-level evaluation if resources allow.")
    parser.add_argument("--negsamp_ratio", type=int, default=1)
    parser.add_argument("--rwr_restart_prob", type=float, default=0.9)
    parser.add_argument("--subgraph_resample_interval", type=int, default=1,
                        help="Regenerate RWR subgraphs every N epochs/test rounds. Larger values reduce CPU cost; 1 preserves old behavior.")
    parser.add_argument("--amp", action="store_true",
                        help="Use CUDA automatic mixed precision to reduce GPU memory. Disabled by default for numerical stability.")
    parser.add_argument("--save_model_path", type=str, default="best_model.pt")
    parser.add_argument("--save_best_model", action="store_true",
                        help="Save the best model checkpoint to --save_model_path. By default the best state is kept in memory to reduce IO.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory for trial scores, trial metrics and checkpoints.")
    parser.add_argument("--summary_csv", type=str, default=None,
                        help="Final summary csv path. Default: <results_dir>/summary.csv")
    parser.add_argument("--rerun_completed", action="store_true",
                        help="Ignore completed trial files and rerun all trials.")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _subgraphs_to_numpy(idx: Sequence[int], subgraphs) -> np.ndarray:
    return np.asarray([subgraphs[i] for i in idx], dtype=np.int64)


def make_batch_tensors(idx, subgraphs, adj_lookup, features_cpu, ft_size, subgraph_size, device):
    """Build only the mini-batch subgraph tensors instead of storing dense N*N adjacency.

    Final tensor layout intentionally matches the previous implementation:
      - adjacency: original subgraph block + one zero row/column whose self-loop is 1
      - feature: [neighbors, zero-vector, seed]
    """
    cur_batch_size = len(idx)
    sub_idx_np = _subgraphs_to_numpy(idx, subgraphs)

    # Build the small batched adjacency on CPU.  subgraph_size is usually 3-8,
    # so dictionary lookup is far cheaper in memory than dense global adjacency.
    ba_np = np.zeros((cur_batch_size, subgraph_size + 1, subgraph_size + 1), dtype=np.float32)
    for b, nodes in enumerate(sub_idx_np):
        for r, src in enumerate(nodes):
            row_weights = adj_lookup[int(src)]
            if not row_weights:
                continue
            for c, dst in enumerate(nodes):
                value = row_weights.get(int(dst))
                if value is not None:
                    ba_np[b, r, c] = value
    ba_np[:, -1, -1] = 1.0

    flat_idx = torch.from_numpy(sub_idx_np.reshape(-1))
    bf_src = features_cpu.index_select(0, flat_idx).view(cur_batch_size, subgraph_size, ft_size)
    bf = torch.zeros((cur_batch_size, subgraph_size + 1, ft_size), dtype=features_cpu.dtype)
    if subgraph_size > 1:
        bf[:, :subgraph_size - 1, :] = bf_src[:, :-1, :]
    bf[:, -1:, :] = bf_src[:, -1:, :]

    ba = torch.from_numpy(ba_np).to(device=device, non_blocking=True)
    bf = bf.to(device=device, non_blocking=True)
    return ba, bf


def make_labels(cur_batch_size: int, negsamp_ratio: int, device):
    lbl = torch.empty((cur_batch_size * (1 + negsamp_ratio), 1), dtype=torch.float32, device=device)
    lbl[:cur_batch_size].fill_(1.0)
    lbl[cur_batch_size:].zero_()
    return lbl


def iter_batches(indices, batch_size):
    """Yield batches while avoiding a final singleton batch.

    The discriminator uses in-batch cyclic-shift negative sampling. A batch of
    one node cannot provide a real in-batch negative sample, so when the last
    chunk has length 1 we merge it into the previous chunk.
    """
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches[-1])
        batches.pop()
    for batch in batches:
        if batch:
            yield batch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_dict(args):
    """Fields that define whether an old trial can be reused."""
    return {
        "dataset": args.dataset,
        "data_dir": os.path.abspath(os.path.expanduser(args.data_dir)),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "num_epoch": args.num_epoch,
        "drop_prob": args.drop_prob,
        "batch_size": args.batch_size,
        "subgraph_size": args.subgraph_size,
        "readout": args.readout,
        "auc_test_rounds": args.auc_test_rounds,
        "negsamp_ratio": args.negsamp_ratio,
        "rwr_restart_prob": args.rwr_restart_prob,
        "subgraph_resample_interval": getattr(args, "subgraph_resample_interval", 1),
        "amp": bool(getattr(args, "amp", False)),
        "memory_mode": "sparse_subgraph_cpu_features_v2",
    }


def config_key(args) -> str:
    raw = json.dumps(config_dict(args), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def trial_score_path(results_dir: Path, dataset: str, cfg_key: str, trial: int) -> Path:
    score_dir = ensure_dir(results_dir / f"{dataset}_scores")
    return score_dir / f"{cfg_key}_trial_{trial:03d}.npz"


def metric_fields():
    return [
        "datetime", "config_key", "dataset", "trial", "seed", "num_epoch", "auc_test_rounds",
        "auc", "auprc", "alpha", "score_file",
    ]


def summary_fields():
    return [
        "datetime", "config_key", "dataset", "trials", "auc", "auprc",
    ]


def append_csv_row(path: Path, fieldnames, row):
    ensure_dir(path.parent)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_trial_metrics(metrics_csv: Path, cfg_key: str, results_dir: Path, dataset: str):
    """Read completed trial metrics and verify that score files still exist."""
    completed = {}
    if not metrics_csv.exists():
        return completed

    with metrics_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("config_key") != cfg_key or row.get("dataset") != dataset:
                continue
            try:
                trial = int(row["trial"])
                auc_value = float(row["auc"])
                auprc_value = float(row["auprc"])
            except (KeyError, TypeError, ValueError):
                continue

            score_file = Path(row.get("score_file", ""))
            if not score_file.is_absolute():
                score_file = results_dir / score_file
            if not score_file.exists():
                score_file = trial_score_path(results_dir, dataset, cfg_key, trial)
            if not score_file.exists():
                continue

            completed[trial] = {
                "trial": trial,
                "seed": int(row.get("seed", trial)),
                "auc": auc_value,
                "auprc": auprc_value,
                "alpha": float(row.get("alpha", "nan")),
                "score_file": str(score_file),
            }
    return completed


def format_metric(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "nan±nan(nan)"
    percent = values * 100.0
    return f"{np.mean(percent):.2f}±{np.std(percent):.2f}({np.max(percent):.2f})"


def compute_auc_auprc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_score = np.asarray(y_score).reshape(-1)
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both positive and negative samples to compute AUC/AUPRC.")
    auc_value = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc_value = auc(recall, precision)
    return float(auc_value), float(auprc_value)


def load_and_preprocess(args) -> PreparedData:
    """Load dataset once and keep memory-heavy objects sparse/CPU-side."""
    adj, features, _, _, _, _, ano_label, _, _ = load_mat(args.dataset, data_dir=args.data_dir)
    adj = adj.tocsr().astype(np.float32)

    degree = np.asarray(adj.sum(axis=0)).reshape(-1)
    degree_ave = float(np.mean(degree))

    features, _ = preprocess_features(features)
    features = np.asarray(features, dtype=np.float32)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    edge_index = adj_to_edge_index(adj)
    neighbor_list = build_neighbor_dict(edge_index, nb_nodes)

    norm_adj, adj_raw = normalize_adj(adj)
    norm_adj = (norm_adj + sp.eye(norm_adj.shape[0], dtype=np.float32, format="csr")).tocsr().astype(np.float32)
    adj_lookup = build_weight_lookup(norm_adj)

    adj_raw = adj_raw.tocsr().astype(np.float32)
    adj_raw.setdiag(0)
    adj_raw.eliminate_zeros()

    return PreparedData(
        adj_lookup=adj_lookup,
        adj_raw=adj_raw,
        features_cpu=torch.from_numpy(features),
        labels=np.asarray(ano_label).reshape(-1).astype(np.int64),
        degree_ave=degree_ave,
        nb_nodes=nb_nodes,
        ft_size=ft_size,
        neighbor_list=neighbor_list,
    )


def maybe_generate_subgraphs(data: PreparedData, args, round_id: int, cached_subgraphs):
    interval = max(1, int(getattr(args, "subgraph_resample_interval", 1)))
    if cached_subgraphs is None or round_id % interval == 0:
        return generate_rwr_subgraph(
            data.neighbor_list,
            num_nodes=data.nb_nodes,
            subgraph_size=args.subgraph_size,
            restart_prob=args.rwr_restart_prob,
        )
    return cached_subgraphs


def copy_state_to_cpu(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def autocast_context(use_amp: bool):
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()


def make_grad_scaler(use_amp: bool):
    if not use_amp:
        return None
    if hasattr(torch, "amp"):
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler()


def compute_structure_scores(nodes_embed: torch.Tensor, adj_raw: sp.csr_matrix, degree_ave: float, nb_nodes: int) -> np.ndarray:
    """Compute topology anomaly scores without materializing an N*N similarity matrix.

    For L2-normalized embeddings z_i:
        sum_{i != j} z_i^T z_j = ||sum_i z_i||_2^2 - n
    This is exactly the same aggregate used by the old code, but it reduces the
    memory footprint from O(N^2) to O(N*d).
    """
    features_norm = F.normalize(nodes_embed, p=2, dim=1).detach().cpu().numpy().astype(np.float32)

    try:
        net = nx.from_scipy_sparse_array(adj_raw)
    except AttributeError:  # networkx < 3.0
        net = nx.from_scipy_sparse_matrix(adj_raw)
    net.remove_edges_from(nx.selfloop_edges(net))

    k_init = max(1, int(degree_ave))
    score_sum = np.zeros(nb_nodes, dtype=np.float32)
    num_score_vectors = 0

    while True:
        core_nodes = list(nx.k_core(net, k_init))
        if len(core_nodes) == 0:
            break

        sub_net = net.subgraph(core_nodes)
        for component in nx.connected_components(sub_net):
            core_temp = np.fromiter(component, dtype=np.int64)
            core_temp_size = int(core_temp.size)
            if core_temp_size <= 1:
                continue

            z = features_norm[core_temp]
            sum_vec = z.sum(axis=0, dtype=np.float64)
            diag_sum = float(np.einsum("ij,ij->i", z, z, dtype=np.float64).sum())
            similar_num = core_temp_size * (core_temp_size - 1)
            similar_temp = float(np.dot(sum_vec, sum_vec) - diag_sum)
            if similar_num == 0 or similar_temp <= 0:
                continue

            score_value = core_temp_size / (similar_temp / similar_num)
            score_sum[core_temp] += np.float32(score_value)
            num_score_vectors += 1
        k_init += 1

    if num_score_vectors == 0:
        return np.zeros(nb_nodes, dtype=np.float32)

    stru_ano_score = score_sum / float(num_score_vectors)
    return MinMaxScaler().fit_transform(stru_ano_score.reshape(-1, 1)).reshape(-1).astype(np.float32)


def run_one_trial(args, trial: int, device, results_dir: Path, cfg_key: str, is_tune: bool = False,
                  data: PreparedData = None):
    seed = trial
    set_seed(seed)

    if data is None:
        data = load_and_preprocess(args)

    model = Model(data.ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([args.negsamp_ratio], dtype=torch.float32, device=device),
    )
    use_amp = bool(getattr(args, "amp", False) and device.type == "cuda")
    scaler = make_grad_scaler(use_amp)

    best = float("inf")
    best_state = copy_state_to_cpu(model)
    cached_subgraphs = None

    # Train model
    for epoch in trange(args.num_epoch, desc="Epoch", position=1 if is_tune else 0, leave=not is_tune):
        model.train()
        all_idx = list(range(data.nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0.0
        seen_nodes = 0
        cached_subgraphs = maybe_generate_subgraphs(data, args, epoch, cached_subgraphs)

        for idx in iter_batches(all_idx, args.batch_size):
            optimiser.zero_grad(set_to_none=True)
            cur_batch_size = len(idx)
            seen_nodes += cur_batch_size
            lbl = make_labels(cur_batch_size, args.negsamp_ratio, device)
            ba, bf = make_batch_tensors(
                idx, cached_subgraphs, data.adj_lookup, data.features_cpu, data.ft_size, args.subgraph_size, device
            )

            with autocast_context(use_amp):
                logits, _ = model(bf, ba)
                loss_all = b_xent(logits, lbl)
                loss = torch.mean(loss_all)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()

            last_loss = float(loss.detach().cpu())
            total_loss += last_loss * cur_batch_size

        mean_loss = total_loss / max(seen_nodes, 1)
        if mean_loss < best:
            best = mean_loss
            best_state = copy_state_to_cpu(model)

    if getattr(args, "save_best_model", False) and not is_tune:
        torch.save(best_state, args.save_model_path)

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    # Test model. Accumulate the mean directly instead of storing [rounds, nodes].
    attr_score_sum = np.zeros(data.nb_nodes, dtype=np.float64)
    nodes_embed = torch.zeros([data.nb_nodes, args.embedding_dim], dtype=torch.float32, device=device)
    cached_subgraphs = None

    with torch.no_grad():
        for test_round in trange(args.auc_test_rounds, desc="Test", position=1 if is_tune else 0, leave=not is_tune):
            all_idx = list(range(data.nb_nodes))
            random.shuffle(all_idx)
            cached_subgraphs = maybe_generate_subgraphs(data, args, test_round, cached_subgraphs)

            for idx in iter_batches(all_idx, args.batch_size):
                cur_batch_size = len(idx)
                ba, bf = make_batch_tensors(
                    idx, cached_subgraphs, data.adj_lookup, data.features_cpu, data.ft_size, args.subgraph_size, device
                )
                with autocast_context(use_amp):
                    logits, batch_embed = model(bf, ba)
                    logits = torch.sigmoid(torch.squeeze(logits))
                if test_round == args.auc_test_rounds - 1:
                    nodes_embed[idx] = batch_embed.float()

                pos_logits = logits[:cur_batch_size]
                neg_logits = logits[cur_batch_size:].view(args.negsamp_ratio, cur_batch_size).mean(dim=0)
                attr_ano_score = -(pos_logits - neg_logits).detach().float().cpu().numpy()
                attr_score_sum[idx] += attr_ano_score

    attr_ano_score_final = (attr_score_sum / max(1, args.auc_test_rounds)).astype(np.float32)
    attr_ano_score_final = MinMaxScaler().fit_transform(attr_ano_score_final.reshape(-1, 1)).reshape(-1).astype(np.float32)

    stru_ano_score_final = compute_structure_scores(nodes_embed, data.adj_raw, data.degree_ave, data.nb_nodes)

    alpha_list = list(np.arange(0, 1, 0.2))
    rate_auc = []
    for alpha in alpha_list:
        final_scores_rate = alpha * attr_ano_score_final + (1 - alpha) * stru_ano_score_final
        auc_temp, _ = compute_auc_auprc(data.labels, final_scores_rate)
        rate_auc.append(auc_temp)
    max_alpha = alpha_list[rate_auc.index(max(rate_auc))]
    final_scores_rate = max_alpha * attr_ano_score_final + (1 - max_alpha) * stru_ano_score_final
    best_auc, best_auprc = compute_auc_auprc(data.labels, final_scores_rate)

    score_file = "None"
    if not is_tune:
        score_file = trial_score_path(results_dir, args.dataset, cfg_key, trial)
        np.savez_compressed(
            score_file,
            y_true=data.labels.astype(np.int64),
            y_score=np.asarray(final_scores_rate).reshape(-1).astype(np.float32),
            attr_score=np.asarray(attr_ano_score_final).reshape(-1).astype(np.float32),
            stru_score=np.asarray(stru_ano_score_final).reshape(-1).astype(np.float32),
            alpha=np.asarray(max_alpha, dtype=np.float32),
            auc=np.asarray(best_auc, dtype=np.float32),
            auprc=np.asarray(best_auprc, dtype=np.float32),
        )

    del model, optimiser, nodes_embed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "trial": trial,
        "seed": seed,
        "auc": best_auc,
        "auprc": best_auprc,
        "alpha": max_alpha,
        "score_file": str(score_file),
    }


def main():
    args = parse_args()
    if args.lr is None:
        args.lr = 3e-3
    if args.num_epoch is None:
        args.num_epoch = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(os.path.expanduser(args.results_dir))
    ensure_dir(results_dir)
    cfg_key = config_key(args)
    metrics_csv = results_dir / "trial_metrics.csv"
    summary_csv = Path(os.path.expanduser(args.summary_csv)) if args.summary_csv else results_dir / "summary.csv"

    print("Loading and preprocessing dataset once...")
    data = load_and_preprocess(args)

    completed = {} if args.rerun_completed else read_trial_metrics(metrics_csv, cfg_key, results_dir, args.dataset)
    target_trials = list(range(1, args.runs + 1))
    pending_trials = [trial for trial in target_trials if trial not in completed]

    if completed and not args.rerun_completed:
        print(f"Found {len(completed)} completed trial(s) for this config. Will run {len(pending_trials)} remaining trial(s).")
    else:
        print(f"Will run {len(pending_trials)} trial(s).")
    print("Config key:", cfg_key)
    print("Memory mode: sparse subgraph adjacency + CPU feature cache")

    trial_results = {trial: completed[trial] for trial in completed if trial in target_trials}
    for trial in tqdm(pending_trials, desc="Trials"):
        result = run_one_trial(args, trial, device, results_dir, cfg_key, data=data)
        trial_results[trial] = result
        append_csv_row(metrics_csv, metric_fields(), {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "config_key": cfg_key,
            "dataset": args.dataset,
            "trial": trial,
            "seed": result["seed"],
            "num_epoch": args.num_epoch,
            "auc_test_rounds": args.auc_test_rounds,
            "auc": f"{result['auc']:.8f}",
            "auprc": f"{result['auprc']:.8f}",
            "alpha": result["alpha"],
            "score_file": os.path.relpath(result["score_file"], start=results_dir),
        })

    missing = [trial for trial in target_trials if trial not in trial_results]
    if missing:
        raise RuntimeError(f"Missing trial results after running: {missing}")

    ordered_results = [trial_results[trial] for trial in target_trials]
    all_auc = [r["auc"] for r in ordered_results]
    all_auprc = [r["auprc"] for r in ordered_results]
    auc_text = format_metric(all_auc)
    auprc_text = format_metric(all_auprc)

    summary_row = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "config_key": cfg_key,
        "dataset": args.dataset,
        "trials": args.runs,
        "auc": auc_text,
        "auprc": auprc_text,
    }
    append_csv_row(summary_csv, summary_fields(), summary_row)

    print("\n==============================")
    print(f"FINAL TESTING AUC:   {auc_text}")
    print(f"FINAL TESTING AUPRC: {auprc_text}")
    print("Summary CSV:", summary_csv)
    print("Trial metrics CSV:", metrics_csv)
    print("==============================")


if __name__ == "__main__":
    main()
