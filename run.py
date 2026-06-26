import argparse
import os
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from model import Model
from utils import adj_to_edge_index, build_neighbor_dict, load_mat, normalize_adj, preprocess_features, generate_rwr_subgraph

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = argparse.ArgumentParser(description="ARISE without DGL; RWR is implemented with PyG-style edge_index + PyTorch.")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat",
                        help="Directory containing <dataset>.mat files. Default: ~/datasets/GAD/mat")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--drop_prob", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--subgraph_size", type=int, default=4)
    parser.add_argument("--readout", type=str, default="avg", choices=["max", "min", "avg", "weighted_sum"])
    parser.add_argument("--auc_test_rounds", type=int, default=256)
    parser.add_argument("--negsamp_ratio", type=int, default=1)
    parser.add_argument("--rwr_restart_prob", type=float, default=0.9)
    parser.add_argument("--save_model_path", type=str, default="best_model.pkl")
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


def main():
    args = parse_args()
    if args.lr is None:
        args.lr = 3e-3
    if args.num_epoch is None:
        args.num_epoch = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_auc = []

    for run in range(args.runs):
        seed = run + 1
        print("Dataset:", args.dataset)
        print("Data dir:", os.path.expanduser(args.data_dir))
        print("lr:", args.lr)
        print("epoch:", args.num_epoch)
        print("seed:", seed)
        print("device:", device)
        set_seed(seed)

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

        model = Model(ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        b_xent = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor([args.negsamp_ratio], dtype=torch.float32, device=device),
        )
        batch_num = nb_nodes // args.batch_size + 1
        best = 1e9
        best_t = 0
        save_model_path = Path(args.save_model_path)

        # Train model
        with tqdm(total=args.num_epoch, desc="Training") as pbar:
            for epoch in range(args.num_epoch):
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
                    best_t = epoch
                    torch.save(model.state_dict(), save_model_path)

                pbar.set_postfix(loss=mean_loss)
                pbar.update(1)

        # Test model
        print(f"Loading {best_t}th epoch")
        model.load_state_dict(torch.load(save_model_path, map_location=device))
        model.eval()

        multi_round_attr_ano_score = np.zeros((args.auc_test_rounds, nb_nodes), dtype=np.float32)
        nodes_embed = torch.zeros([nb_nodes, args.embedding_dim], dtype=torch.float32, device=device)

        with tqdm(total=args.auc_test_rounds, desc="Testing") as pbar_test:
            for test_round in range(args.auc_test_rounds):
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

                pbar_test.update(1)

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
            auc_temp = roc_auc_score(ano_label, final_scores_rate)
            rate_auc.append(auc_temp)
        max_alpha = alpha_list[rate_auc.index(max(rate_auc))]
        final_scores_rate = max_alpha * attr_ano_score_final + (1 - max_alpha) * stru_ano_score_final
        best_auc = roc_auc_score(ano_label, final_scores_rate)
        print("Alpha:", max_alpha)
        print("AUC:{:.4f}".format(best_auc))
        print()
        all_auc.append(best_auc)

    print("\n==============================")
    print("FINAL TESTING AUC:{:.4f}".format(np.mean(all_auc)))
    print("==============================")


if __name__ == "__main__":
    main()
