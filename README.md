# ARISE: Graph Anomaly Detection on Attributed Networks via Substructure Awareness

An forked code for paper "ARISE: Graph Anomaly Detection on Attributed Networks via Substructure Awareness", accepted by IEEE TNNLS. Any communications or issues are welcomed. Please contact jingcan_duan@163.com. If you find this repository useful to your research or work, it is really appreciate to star this repository. :heart:

### Requirements

```bash
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric scikit-learn --torch-backend=cu128
```

### Quick Start

python run.py --dataset cora

The default dataset directory has been changed to `~/datasets/GAD/mat`, so the file should be placed as `~/datasets/GAD/mat/<dataset>.mat`. You can also specify another directory, for example:

```bash
python run.py --dataset cora --data_dir ~/datasets/GAD/mat
```

### Citation

If you find this project useful for your research, please cite your paper with the following BibTeX entry.

```
@article{ARISE,
  title={ARISE: Graph Anomaly Detection on Attributed Networks via Substructure Awareness},
  author={Duan, Jingcan and Xiao, Bin and Wang, Siwei and Zhou, Haifang and Liu, Xinwang},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```
