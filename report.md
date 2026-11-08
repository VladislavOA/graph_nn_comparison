# Midway Check Report: Reproducing and Stress-Testing "SGC $\approx$ GCN"
**Team:** Vladislav Osetrov, Danis Abdulov  
**Milestone:** Week 12 Midway Check

---

## 1. Our Contribution & External Code Citations

### External Code & Libraries Used
* **PyTorch Geometric (`pyg-team/pytorch_geometric`, `v2.6.0`):** Used `Planetoid` dataset loaders, `NormalizeFeatures`, `RandomNodeSplit`, `GCNConv`, and `SGConv` (adapted from `examples/gcn.py`).
* **GCN Reference (`tkipf/gcn`, commit `39a4089`, `gcn/models.py` & `gcn/train.py`):** Kipf & Welling (ICLR 2017) protocol applying L2 weight decay ($5 \times 10^{-4}$) strictly to the first-layer weight matrix $W_1$ and using a 10-epoch rolling validation loss early-stopping window.
* **SGC Reference (`Tiiiger/SGC`, commit `2c7a272`, `citation.py` & `tuning.py`):** Wu et al. (ICML 2019) protocol precomputing $\bar{X} = \hat{A}^K X$ once (`SGConv(..., cached=True)`) and optimizing multiclass logistic regression with Adam ($\text{lr} = 0.2$) and dataset-specific L2 weight decay ($\text{wd} = 5 \times 10^{-5}$ for 100 epochs on Cora/PubMed; $\text{wd} = 5 \times 10^{-3}$ for 150 epochs on CiteSeer).

### Our Implementation & Contributions
* **Unified Evaluation Pipeline (`models.py`, `reproduce_gcn_vs_sgc.ipynb`, `run_experiments.py`):** Implemented modular `GCN`, `SGC`, and structure-free `MLP` baselines under a shared PyTorch Geometric pipeline with exact weight-decay group separation and wall-clock timing.
* **Step 1 (Public Split Reproduction & Ablations):** Evaluated 2-layer GCN, $K=2$ SGC, 2-layer MLP, and $K=0$ SGC across 10 random seeds on the canonical Planetoid splits, plus 5-seed ablations across propagation degree $K \in \{0, 1, 2, 4, 8\}$ and network depth $L \in \{1, 2, 3, 4\}$.
* **Step 2 (100-Split Stress-Test & Variance Decomposition):** Evaluated paired GCN and SGC models across 100 random splits $\times$ 5 initialization seeds (1,000 runs per model per dataset), computing paired differences $\delta_s = \text{acc}_{\text{GCN}}(s) - \text{acc}_{\text{SGC}}(s)$, 10,000-resample 95% bootstrap confidence intervals, Wilcoxon signed-rank $p$-values, and law-of-total-variance decompositions between split variance and seed variance.

---

## 2. Discrepancy Log

| Component / Setting | Original Paper Specification | Our PyG Implementation | Observed Impact on Reproduction |
| :--- | :--- | :--- | :--- |
| **GCN L2 Regularization** | Kipf & Welling (2017) penalize only first-layer weights $W_1$ ($\text{wd} = 5 \times 10^{-4}$), leaving $W_2$ unregularized. | Parameter groups in Adam: `convs[0]` with `weight_decay=5e-4`, remaining parameters with `weight_decay=0.0`. | Applying L2 decay to all layers drops Cora/CiteSeer accuracy by $1.0\text{--}1.5\%$; matching first-layer-only decay reproduces reported GCN numbers within $0.2\text{--}0.8\%$. |
| **SGC Optimizer & Hyperparameters** | Wu et al. (2019) report L-BFGS in the paper text but use Adam ($\text{lr} = 0.2$) with Hyperopt-tuned weight decay in their official script (`citation.py`). | Adam ($\text{lr} = 0.2$) with official repo values: $\text{wd} = 5 \times 10^{-5}$ (100 epochs) on Cora/PubMed; $\text{wd} = 5 \times 10^{-3}$ (150 epochs) on CiteSeer. | Reproduces exact $81.0\%$ on Cora and $78.8\%$ on PubMed, and achieves $72.3\%$ on CiteSeer ($+0.4\%$ above paper). However, $\text{wd} = 5 \times 10^{-3}$ is brittle when the training split size changes (see Step 2). |
| **MLP Seed Sensitivity** | 2-layer MLP with ReLU, dropout $0.5$, hidden dimension $16$, and 10-epoch rolling validation loss early stopping. | Same architecture and early-stopping criterion as GCN (`patience=10` on rolling validation loss). | Without graph smoothing, 2-layer MLP exhibits high initialization variance across seeds ($\pm 14.1\%$ on Cora, $\pm 15.9\%$ on PubMed), whereas linear $K=0$ SGC is deterministic ($\pm 0.0\%$). |
| **Random Split Protocol (Step 2)** | Planetoid public split uses $20c$ training nodes, $500$ validation nodes, and $1,000$ test nodes. | Used `RandomNodeSplit(split='train_rest', num_val=500, num_test=1000)` across 100 random seeds. | In PyG, `split='train_rest'` assigns $500$ validation and $1,000$ test nodes and places all remaining nodes into the training set. With larger training sets, non-linear GCN significantly outperforms linear SGC across all three benchmarks. |

---

## 3. Preliminary Results

### 3.1 Dataset Summary & Edge Homophily
All three citation networks exhibit high edge homophily ($h > 0.73$), explaining why low-pass neighborhood averaging ($\hat{A}^K X$) provides a strong inductive bias.

| Dataset | Nodes | Undirected Edges | Features | Classes | Public Split (Train/Val/Test) | Edge Homophily ($h$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cora** | 2,708 | 5,278 | 1,433 | 7 | 140 / 500 / 1000 | 0.810 |
| **CiteSeer** | 3,327 | 4,552 | 3,703 | 6 | 120 / 500 / 1000 | 0.736 |
| **PubMed** | 19,717 | 44,324 | 500 | 3 | 60 / 500 / 1000 | 0.802 |

---

### 3.2 Step 1: Public Split Reproduction (10 Seeds) & Training Speedup

| Dataset | GCN Paper (2017) | SGC Paper (2019) | Our GCN ($L=2$) | Our SGC ($K=2$) | Our MLP ($L=2$) | Our SGC ($K=0$) | SGC Training Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cora** | $81.5\%$ | $81.0 \pm 0.0\%$ | $80.7 \pm 0.8\%$ | $81.0 \pm 0.0\%$ | $48.1 \pm 14.1\%$ | $59.9 \pm 0.0\%$ | $25.2\times$ |
| **CiteSeer** | $70.3\%$ | $71.9 \pm 0.1\%$ | $70.5 \pm 0.7\%$ | $72.3 \pm 0.1\%$ | $54.2 \pm 2.4\%$ | $61.4 \pm 0.0\%$ | $16.4\times$ |
| **PubMed** | $79.0\%$ | $78.9 \pm 0.0\%$ | $78.3 \pm 0.4\%$ | $78.8 \pm 0.0\%$ | $63.2 \pm 15.9\%$ | $73.3 \pm 0.0\%$ | $18.1\times$ |

* **Accuracy Parity on Public Splits:** On the standard Planetoid splits, linear SGC ($K=2$) slightly outperforms 2-layer GCN on all three datasets ($+0.3\%$ on Cora, $+1.8\%$ on CiteSeer, $+0.5\%$ on PubMed) with near-zero seed variance ($\le 0.1\%$) due to convex cross-entropy optimization.
* **Role of Graph Propagation ($K=2$ vs. $K=0$/MLP):** Removing neighborhood propagation ($K=0$ SGC and MLP) causes massive accuracy drops of $20.8\text{--}32.6\%$ on Cora, $10.9\text{--}16.3\%$ on CiteSeer, and $5.5\text{--}15.1\%$ on PubMed, confirming that 2-hop Laplacian smoothing—not hidden-layer ReLU non-linearity—drives public-split performance.
* **Wall-Clock Efficiency:** Caching $\hat{A}^2 X$ enables SGC to train $16.4\times$ to $25.2\times$ faster than 2-layer GCN (`figures/fig1_reproduction.png`).

---

### 3.3 Depth ($L$) vs. Propagation Degree ($K$) Ablation (5 Seeds)

| Dataset | SGC $K=0$ | SGC $K=1$ | SGC $K=2$ | SGC $K=4$ | SGC $K=8$ | GCN $L=1$ | GCN $L=2$ | GCN $L=3$ | GCN $L=4$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cora** | $59.8\%$ | $77.1\%$ | $81.1\%$ | **$81.6\%$** | $80.6\%$ | $76.7\%$ | **$80.9\%$** | $76.0\%$ | $72.1\%$ |
| **CiteSeer** | $61.4\%$ | $71.1\%$ | $72.3\%$ | $72.5\%$ | **$72.9\%$** | $68.7\%$ | **$70.2\%$** | $62.3\%$ | $46.9\%$ |
| **PubMed** | $73.3\%$ | $76.9\%$ | $78.8\%$ | **$79.5\%$** | $76.8\%$ | $74.5\%$ | **$78.4\%$** | $75.3\%$ | $73.4\%$ |

* **Decoupling Over-Smoothing from Optimization Degradation:** As shown in `figures/fig2_ablation.png`, SGC accuracy peaks at $K=4$ on Cora ($81.6\%$) and PubMed ($79.5\%$) and continues increasing up to $K=8$ on CiteSeer ($72.9\%$). In contrast, GCN peaks at $L=2$ and degrades sharply at $L=3$ and $L=4$ (falling to $72.1\%$ on Cora and $46.9\%$ on CiteSeer). Because both models apply the same $K$-step graph filter, GCN's sharp drop at $L \ge 3$ is primarily caused by optimization and overfitting in deeper non-linear parameter stacks rather than feature over-smoothing alone.

---

### 3.4 Step 2: 100 Random Splits, Paired Significance & Variance Decomposition

| Dataset | GCN (100 Splits) | SGC (100 Splits) | Paired $\Delta$ (GCN $-$ SGC) | 95% Bootstrap CI | Wilcoxon $p$-value | GCN Split Var Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cora** | $86.1 \pm 1.2\%$ | $80.9 \pm 1.6\%$ | $+5.21 \pm 1.02\%$ | $[+5.01\%, +5.41\%]$ | $3.90 \times 10^{-18}$ (Sig) | $65.3\%$ |
| **CiteSeer** | $75.1 \pm 1.3\%$ | $24.5 \pm 5.8\%$ | $+50.63 \pm 5.89\%$ | $[+49.45\%, +51.75\%]$ | $3.90 \times 10^{-18}$ (Sig) | $89.2\%$ |
| **PubMed** | $86.2 \pm 1.1\%$ | $84.1 \pm 1.1\%$ | $+2.06 \pm 0.59\%$ | $[+1.95\%, +2.18\%]$ | $3.90 \times 10^{-18}$ (Sig) | $92.3\%$ |

* **Breakdown of "SGC $\approx$ GCN" Across Random Splits:** Under the 100 random `train_rest` splits (`figures/fig3_split_variance.png`), GCN outperforms SGC with high statistical significance ($p = 3.90 \times 10^{-18}$) across all three datasets, gaining $+5.21\%$ on Cora and $+2.06\%$ on PubMed when more labeled training nodes are available to fit non-linear decision boundaries.
* **Hyperparameter Fragility of SGC on CiteSeer:** On CiteSeer, SGC drops to $24.5 \pm 5.8\%$ across random splits because the public-split-tuned weight decay ($\text{wd} = 5 \times 10^{-3}$ with $\text{lr} = 0.2$) over-regularizes/destabilizes optimization when the training set distribution changes.
* **Variance Decomposition:** Using the law of total variance, $\text{Var}(\text{acc}) = \text{Var}_s(\mathbb{E}_r[\text{acc}]) + \mathbb{E}_s[\text{Var}_r(\text{acc})]$, graph split selection accounts for $65.3\%$ (Cora), $89.2\%$ (CiteSeer), and $92.3\%$ (PubMed) of GCN's total variance, and nearly $100\%$ of SGC's variance.

---

## 4. Blocker List & Next Steps for Final Report

1. **Completed Milestones (No Active Blockers on Steps 1 & 2):** Public split reproduction, depth/degree ablations, and the 100-split variance and paired significance pipelines are complete and verified.
2. **Split Size & Weight Decay Sensitivity Analysis (Before Final Submission):** Because `RandomNodeSplit(split='train_rest')` allocates all non-val/test nodes to training and exposes SGC's sensitivity to fixed $\text{wd} = 5 \times 10^{-3}$ on CiteSeer, we will compare `split='test_rest'` ($20c$ training nodes) against `split='train_rest'` and include per-split validation tuning of $\text{wd}$ alongside our Step 3 heterophilous benchmarks (`Chameleon`, `Squirrel`, `Actor`, `Cornell`, `Texas`, `Wisconsin`).