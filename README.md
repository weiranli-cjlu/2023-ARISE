# ARISE: Graph Anomaly Detection on Attributed Networks via Substructure Awareness

This forked code is based on the paper "ARISE: Graph Anomaly Detection on Attributed Networks via Substructure Awareness", accepted by IEEE TNNLS.

This version removes the DGL dependency and uses PyG-style `edge_index` plus PyTorch/Python sampling for RWR, so it is easier to run on newer CUDA/GPU environments such as RTX 50 series.

## Requirements

```bash
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric scikit-learn --torch-backend=cu128
```

## Dataset

The default dataset directory is:

```text
~/datasets/GAD/mat
```

The dataset file should be placed as:

```text
~/datasets/GAD/mat/<dataset>.mat
```

You can also specify another directory with `--data_dir`.

## Quick Start

```bash
python run.py --dataset cora --data_dir ~/datasets/GAD/mat
```

## Multi-trial evaluation and CSV output

The script now supports resumable multi-trial experiments. If the target number of trials is larger than the number of completed trials with the same configuration, only the remaining trials are run.

```bash
python run.py \
  --dataset cora \
  --data_dir ~/datasets/GAD/mat \
  --runs 5 \
  --num_epoch 100 \
  --results_dir results
```

Outputs:

```text
results/trial_metrics.csv                 # per-trial auc/auprc and score file path
results/summary.csv                       # final summary csv
results/<dataset>_scores/*.npz            # y_true and y_score for every trial
```

The final `summary.csv` contains:

```text
datetime, config_key, dataset, trials, auc, auprc
```

`auc` and `auprc` are formatted as:

```text
90.21±2.33(91.00)
```

meaning `mean ± std (max)`, expressed as percentages.

AUPRC is computed by scikit-learn:

```python
precision, recall, _ = precision_recall_curve(y_true, y_score)
auprc = auc(recall, precision)
```

Each trial score file is a compressed `.npz` file with at least:

```text
y_true, y_score, attr_score, stru_score, alpha, auc, auprc
```

To force rerunning all trials instead of resuming, add:

```bash
--rerun_completed
```

## Citation

If you find this project useful for your research, please cite the original paper with the following BibTeX entry.

```bibtex
@article{ARISE,
  title={ARISE: Graph Anomaly Detection on Attributed Networks via Substructure Awareness},
  author={Duan, Jingcan and Xiao, Bin and Wang, Siwei and Zhou, Haifang and Liu, Xinwang},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```
