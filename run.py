import argparse
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

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
from utils import adj_to_edge_index, build_neighbor_dict, generate_rwr_subgraph, load_mat, normalize_adj, preprocess_features

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = argparse.ArgumentParser(description="ARISE without DGL; RWR is implemented with PyG-style edge_index + PyTorch.")
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
    parser.add_argument("--auc_test_rounds", type=int, default=256)
    parser.add_argument("--negsamp_ratio", type=int, default=1)
    parser.add_argument("--rwr_restart_prob", type=float, default=0.9)
    parser.add_argument("--save_model_path", type=str, default="best_model.pt")
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
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_batch_tensors(idx, subgraphs, adj, features, ft_size, subgraph_size, device):
    cur_batch_size = len(idx)
    # Vectorized extraction. This is much faster than looping over nodes with
    # repeated advanced indexing on the dense global adjacency matrix.
    sub_idx = torch.as_tensor([subgraphs[i] for i in idx], dtype=torch.long, device=device)
    ba = adj[sub_idx.unsqueeze(2), sub_idx.unsqueeze(1)]
    bf = features[sub_idx]

    added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size), device=device)
    added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1), device=device)
    added_adj_zero_col[:, -1, :] = 1.0
    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size), device=device)

    ba = torch.cat((ba, added_adj_zero_row), dim=1)
    ba = torch.cat((ba, added_adj_zero_col), dim=2)
    bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
    return ba, bf


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


def load_and_preprocess(args, device):
    # Load and preprocess data. Default path: ~/datasets/GAD/mat/<dataset>.mat
    adj, features, _, _, _, _, ano_label, _, _ = load_mat(args.dataset, data_dir=args.data_dir)

    degree = np.asarray(adj.sum(axis=0)).reshape(-1)
    degree_ave = float(np.mean(degree))

    features, _ = preprocess_features(features)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    # PyG-style graph representation for DGL-free RWR subgraph sampling.
    edge_index = adj_to_edge_index(adj)
    neighbor_list = build_neighbor_dict(edge_index, nb_nodes)

    adj, adj_raw = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj_raw = np.asarray(adj_raw.todense())

    features = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    adj = torch.as_tensor(np.asarray(adj), dtype=torch.float32, device=device)

    return adj, adj_raw, features, ano_label, degree_ave, nb_nodes, ft_size, neighbor_list


def run_one_trial(args, trial: int, device, results_dir: Path, cfg_key: str, is_tune: bool=False):
    seed = trial
    set_seed(seed)

    adj, adj_raw, features, ano_label, degree_ave, nb_nodes, ft_size, neighbor_list = load_and_preprocess(args, device)

    model = Model(ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([args.negsamp_ratio], dtype=torch.float32, device=device),
    )
    batch_num = nb_nodes // args.batch_size + 1
    best = 1e9

    # Train model
    for epoch in trange(args.num_epoch, desc=f"Epoch", position=1 if is_tune else 0, leave=not is_tune):
        model.train()
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0.0
        last_loss = 0.0
        last_batch_size = 0

        subgraphs = generate_rwr_subgraph(
            neighbor_list,
            num_nodes=nb_nodes,
            subgraph_size=args.subgraph_size,
            restart_prob=args.rwr_restart_prob,
        )

        for batch_idx in range(batch_num):
            is_final_batch = batch_idx == (batch_num - 1)
            if not is_final_batch:
                idx = all_idx[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
            else:
                idx = all_idx[batch_idx * args.batch_size:]
            if len(idx) == 0:
                continue

            optimiser.zero_grad()
            cur_batch_size = len(idx)
            last_batch_size = cur_batch_size
            lbl = torch.cat(
                (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))
            ).unsqueeze(1).to(device)

            ba, bf = make_batch_tensors(idx, subgraphs, adj, features, ft_size, args.subgraph_size, device)
            logits, _ = model(bf, ba)
            loss_all = b_xent(logits, lbl)
            loss = torch.mean(loss_all)
            loss.backward()
            optimiser.step()

            last_loss = float(loss.detach().cpu())
            if not is_final_batch:
                total_loss += last_loss

        mean_loss = (total_loss * args.batch_size + last_loss * last_batch_size) / nb_nodes
        if mean_loss < best:
            best = mean_loss
            torch.save(model.state_dict(), args.save_model_path)

    # Test model
    model.load_state_dict(torch.load(args.save_model_path, map_location=device))
    model.eval()

    multi_round_attr_ano_score = np.zeros((args.auc_test_rounds, nb_nodes), dtype=np.float32)
    nodes_embed = torch.zeros([nb_nodes, args.embedding_dim], dtype=torch.float32, device=device)

    for test_round in trange(args.auc_test_rounds, desc="Test", position=1 if is_tune else 0, leave=not is_tune):
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        subgraphs = generate_rwr_subgraph(
            neighbor_list,
            num_nodes=nb_nodes,
            subgraph_size=args.subgraph_size,
            restart_prob=args.rwr_restart_prob,
        )

        for batch_idx in range(batch_num):
            is_final_batch = batch_idx == (batch_num - 1)
            if not is_final_batch:
                idx = all_idx[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
            else:
                idx = all_idx[batch_idx * args.batch_size:]
            if len(idx) == 0:
                continue

            cur_batch_size = len(idx)
            ba, bf = make_batch_tensors(idx, subgraphs, adj, features, ft_size, args.subgraph_size, device)

            with torch.no_grad():
                logits, batch_embed = model(bf, ba)
                logits = torch.sigmoid(torch.squeeze(logits))
                if test_round == args.auc_test_rounds - 1:
                    nodes_embed[idx] = batch_embed

            pos_logits = logits[:cur_batch_size]
            neg_logits = logits[cur_batch_size:].view(args.negsamp_ratio, cur_batch_size).mean(dim=0)
            attr_ano_score = -(pos_logits - neg_logits).detach().cpu().numpy()
            multi_round_attr_ano_score[test_round, idx] = attr_ano_score

    # Attribute anomaly scores
    attr_ano_score_final = np.mean(multi_round_attr_ano_score, axis=0)
    attr_scaler = MinMaxScaler()
    attr_ano_score_final = attr_scaler.fit_transform(attr_ano_score_final.reshape(-1, 1)).reshape(-1)

    # Topology anomaly scores
    features_norm = F.normalize(nodes_embed, p=2, dim=1)
    features_similarity = torch.matmul(features_norm, features_norm.transpose(0, 1)).detach().cpu().numpy()

    k_init = max(1, int(degree_ave))
    net = nx.from_numpy_array(adj_raw)
    net.remove_edges_from(nx.selfloop_edges(net))
    adj_raw_no_loop = nx.to_numpy_array(net)
    multi_round_stru_ano_score = []

    while True:
        list_temp = list(nx.k_core(net, k_init))
        if len(list_temp) == 0:
            break
        core_adj = adj_raw_no_loop[list_temp, :][:, list_temp]
        core_graph = nx.from_numpy_array(core_adj)
        list_temp = np.array(list_temp)
        for component in nx.connected_components(core_graph):
            core_temp = list_temp[list(component)]
            core_temp_size = len(core_temp)
            if core_temp_size <= 1:
                continue
            sim_block = features_similarity[np.ix_(core_temp, core_temp)]
            similar_num = core_temp_size * (core_temp_size - 1)
            similar_temp = float(sim_block.sum() - np.trace(sim_block))
            if similar_num == 0 or similar_temp == 0:
                continue
            scores_temp = np.zeros(nb_nodes, dtype=np.float32)
            scores_temp[core_temp] = core_temp_size / (similar_temp / similar_num)
            multi_round_stru_ano_score.append(scores_temp)
        k_init += 1

    if len(multi_round_stru_ano_score) == 0:
        stru_ano_score_final = np.zeros(nb_nodes, dtype=np.float32)
    else:
        multi_round_stru_ano_score = np.array(multi_round_stru_ano_score)
        multi_round_stru_ano_score = np.mean(multi_round_stru_ano_score, axis=0)
        stru_scaler = MinMaxScaler()
        stru_ano_score_final = stru_scaler.fit_transform(multi_round_stru_ano_score.reshape(-1, 1)).reshape(-1)

    alpha_list = list(np.arange(0, 1, 0.2))
    rate_auc = []
    for alpha in alpha_list:
        final_scores_rate = alpha * attr_ano_score_final + (1 - alpha) * stru_ano_score_final
        auc_temp, _ = compute_auc_auprc(ano_label, final_scores_rate)
        rate_auc.append(auc_temp)
    max_alpha = alpha_list[rate_auc.index(max(rate_auc))]
    final_scores_rate = max_alpha * attr_ano_score_final + (1 - max_alpha) * stru_ano_score_final
    best_auc, best_auprc = compute_auc_auprc(ano_label, final_scores_rate)

    score_file = "None"
    if not is_tune:
        score_file = trial_score_path(results_dir, args.dataset, cfg_key, trial)
        np.savez_compressed(
            score_file,
            y_true=np.asarray(ano_label).reshape(-1).astype(np.int64),
            y_score=np.asarray(final_scores_rate).reshape(-1).astype(np.float32),
            attr_score=np.asarray(attr_ano_score_final).reshape(-1).astype(np.float32),
            stru_score=np.asarray(stru_ano_score_final).reshape(-1).astype(np.float32),
            alpha=np.asarray(max_alpha, dtype=np.float32),
            auc=np.asarray(best_auc, dtype=np.float32),
            auprc=np.asarray(best_auprc, dtype=np.float32),
        )

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

    completed = {} if args.rerun_completed else read_trial_metrics(metrics_csv, cfg_key, results_dir, args.dataset)
    target_trials = list(range(1, args.runs + 1))
    pending_trials = [trial for trial in target_trials if trial not in completed]

    if completed and not args.rerun_completed:
        print(f"Found {len(completed)} completed trial(s) for this config. Will run {len(pending_trials)} remaining trial(s).")
    else:
        print(f"Will run {len(pending_trials)} trial(s).")
    print("Config key:", cfg_key)

    trial_results = {trial: completed[trial] for trial in completed if trial in target_trials}
    for trial in pending_trials:
        result = run_one_trial(args, trial, device, results_dir, cfg_key)
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
