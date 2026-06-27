# ARISE 低硬件消耗优化说明

本版本针对 `run.py`、`utils.py`、`optuna_tune.py` 做了资源消耗优化，默认仍保持原有训练/测试流程和 CSV/NPZ 结果输出。

## 主要优化

1. **取消全图 dense 邻接矩阵上 GPU**
   - 原代码会执行 `adj.todense()` 并把 `N*N` 邻接矩阵放到 GPU。
   - 新代码默认保留稀疏邻接，仅在每个 mini-batch 中按子图节点构造很小的 `(subgraph_size+1)^2` 邻接张量。
   - 显存从约 `O(N^2)` 降为 `O(batch_size * subgraph_size^2)`。

2. **特征矩阵默认保存在 CPU**
   - 只把当前 batch 的子图特征搬到 GPU，降低 GPU 常驻显存。

3. **多 trial 共享一次数据加载/预处理结果**
   - `run.py` 和 `optuna_tune.py` 都会先加载一次数据集。
   - 后续 trial 复用邻居表、稀疏邻接索引和特征缓存，减少重复 IO 与 CPU 预处理。

4. **测试阶段不再保存 `[test_rounds, nodes]` 大矩阵**
   - 直接累加每轮 anomaly score 后求均值。
   - 内存从 `O(test_rounds * N)` 降为 `O(N)`。

5. **结构异常分数不再构造 `N*N` embedding similarity**
   - 原代码使用 `features_similarity = Z @ Z.T`。
   - 新代码使用等价公式 `sum_{i != j} z_i^T z_j = ||sum_i z_i||^2 - sum_i ||z_i||^2`。
   - 内存从 `O(N^2)` 降为 `O(N*d)`。

6. **默认不频繁写 checkpoint**
   - 最优模型权重默认保存在内存中，减少磁盘 IO。
   - 如需保存模型，添加 `--save_best_model`。

7. **提供可选降耗参数**
   - `--auc_test_rounds` 默认由 256 降为 64；论文最终结果如需更稳可手动设回 256。
   - `--subgraph_resample_interval N`：每 N 个 epoch/test round 重采样一次 RWR 子图，N 越大 CPU 消耗越低；默认 1 保持旧行为。
   - `--amp`：CUDA 环境下启用混合精度，进一步降低显存。

8. **Optuna 默认搜索空间更省资源**
   - 默认不搜索 `embedding_dim=256` 和 `batch_size=512`。
   - 如需恢复大搜索空间，添加 `--allow_large_search`。

## 示例命令

省资源普通训练：

```bash
python run.py --dataset twitter --num_epoch 100 --runs 5 --batch_size 256 --auc_test_rounds 64 --subgraph_resample_interval 2
```

如果显存仍紧张，可进一步降低：

```bash
python run.py --dataset twitter --num_epoch 100 --runs 5 --embedding_dim 32 --batch_size 128 --auc_test_rounds 32 --subgraph_resample_interval 5
```

启用混合精度：

```bash
python run.py --dataset twitter --num_epoch 100 --runs 5 --amp
```

Optuna 调优：

```bash
python optuna_tune.py --dataset twitter --n_trials 30 --auc_test_rounds 32 --subgraph_resample_interval 2
```

恢复较大的 Optuna 搜索空间：

```bash
python optuna_tune.py --dataset twitter --n_trials 30 --allow_large_search
```
