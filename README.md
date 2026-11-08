# Reproducing and Stress-Testing "SGC ≈ GCN" (Midway Check)
**Team:** Vladislav Osetrov, Danis Abdulov

## Repository Structure
* `models.py` — Concise `GCN`, `SGC`, and `MLP` models using PyTorch Geometric (`GCNConv`, `SGConv`).
* `reproduce_gcn_vs_sgc.ipynb` — Main notebook reproducing public split results, depth/degree ablations, and the 100-split paired significance study.
* `midway_report.md` — Concise Midway Check report (includes *Our Contribution*, *Discrepancy Log*, *Preliminary Results*, and *Blocker List*).
* `figures/` — Generated plots (`fig1_reproduction.png`, `fig2_ablation.png`, `fig3_split_variance.png`).
* `requirements.txt` — Pinned Python dependencies.

## Quickstart
```bash
pip install -r requirements.txt
jupyter notebook reproduce_gcn_vs_sgc.ipynb
```

## External Code Citations
* **PyTorch Geometric (`pyg-team/pytorch_geometric`, release tag `v2.6.0`):** Used `Planetoid`, `NormalizeFeatures`, `RandomNodeSplit`, `GCNConv`, and `SGConv` (`examples/gcn.py`).
* **GCN (`tkipf/gcn`, commit `39a4089`, `gcn/models.py` & `gcn/train.py`; `tkipf/pygcn` `master`):** First-layer L2 penalty ($5\times 10^{-4}$ on $W_1$) and 10-epoch rolling validation loss early stopping.
* **SGC (`Tiiiger/SGC`, commit `2c7a272`, `citation.py` & `tuning.py`):** Precomputed feature propagation (`cached=True`), Adam $\text{lr}=0.2$, and tuned weight decay per dataset.
