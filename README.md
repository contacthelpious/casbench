# CausalRec-Bench

**A Semi-Synthetic Benchmark for Evaluating Causal Cold-Start Recommendation Under Exposure Bias and Concept Drift**

Ali Hassan — RecSys 2026 CONSEQUENCES Workshop  
Collaborating with Dr. Yan Zhang, Charles Darwin University

---

## Quick Start — 3 Commands

```bash
git clone https://github.com/contacthelpious/casbench.git
cd casbench
pip install -r requirements.txt
python benchmark/run_evaluation.py
```

All data and pre-trained models are included. No downloads required.

---

## What Is CausalRec-Bench?

The first semi-synthetic benchmark for causal recommendation evaluation with ground-truth causal labels on every interaction. Covers seven evaluation dimensions simultaneously: cold-start, warm users, difficulty levels, seasonal concept drift, cross-domain generalisation, position bias, and causal metric comparison.

| Statistic | Value |
|-----------|-------|
| Users | 50,000 |
| Items | 4,000 (2 domains) |
| Interactions | 2,468,985 |
| Domains | E-commerce + Streaming |
| Confounders | 5 |
| Cold-start users | 9,896 (19.8%) |
| Non-genuine clicks | 70.0% |
| Evaluation splits | 18 |

---

## Key Findings

| Finding | Scenario | Result |
|---------|----------|--------|
| Causal MF vs Standard MF | Cold-Start | **+46.1%** CP@10 |
| Causal LightGCN vs Standard LightGCN | Level 3 Hard (warm users) | **+31.8%** CP@10 |
| Non-genuine clicks | Entire benchmark | **70.0%** confounder-driven |
| Position bias ratio | All interactions | **1.94x** position 1 vs 10 |
| Promotion bias ratio | All interactions | **1.55x** |
| Popularity domain collapse | E-commerce cold-start | **0.0000** CP@10 |
| Graph methods cold-start | Cold-start | **Identical** regardless of causal training |

---

## Five Confounders

| Confounder | Effect | Measured Ratio |
|------------|--------|----------------|
| Promotion bias | +40% exposure, +15% click | 1.55x |
| Popularity bias | +30% exposure, +12% click | — |
| Position bias (novel) | +25% at position 1, decays to +1% at position 10 | 1.94x |
| Seasonal concept drift | Winter +15% books, Summer +15% outdoor | Validated |
| New item penalty | -20% exposure for new items | Validated |

---

## Causal Ground-Truth Labels

Every interaction includes a `click_cause` label — unavailable in any existing public recommendation dataset:

| Label | Meaning |
|-------|---------|
| `genuine_preference` | Click reflects true user preference |
| `promotion_bias` | Click caused by promotional placement |
| `popularity_bias` | Click caused by social proof |
| `position_bias` | Click caused by display position |
| `mixed` | Multiple confounder effects |
| `no_click` | Item exposed but not clicked |

---

## Complete Results — All 6 Models, All Scenarios

### Category Precision@10

| Model | Cold-Start | Level 3 Hard | Level 1 Simple | Winter CS | Summer CS | E-com CS | Stream CS |
|-------|-----------|--------------|----------------|-----------|-----------|----------|----------|
| Popularity | 0.2462 | 0.2463 | 0.1058 | 0.2230 | 0.2732 | 0.0000 | 0.3865 |
| Standard MF | 0.2835 | 0.2900 | 0.1183 | 0.2869 | 0.2853 | 0.1696 | 0.2679 |
| **Causal MF** | **0.4140** | **0.4241** | **0.1852** | **0.4156** | **0.4146** | **0.3853** | 0.3037 |
| Standard LightGCN | 0.5480 | 0.5474 | 0.2393 | 0.5574 | 0.5528 | 0.5044 | 0.3816 |
| **Causal LightGCN** | 0.5480 | **0.7216** | **0.3584** | 0.5574 | 0.5528 | 0.5044 | 0.3816 |
| Causal Upper Bound | 0.5480 | 0.5508 | 0.2386 | 0.5574 | 0.5528 | 0.5044 | 0.3816 |

CS = Cold-Start

### Key Observations From Results Table

**Causal MF consistently beats Standard MF across ALL scenarios** — cold-start, warm users, seasonal splits, and domain splits. This is not limited to one scenario.

**Causal LightGCN dramatically improves warm user recommendation** — +31.8% on Level 3 Hard while producing identical cold-start results, revealing a structural limitation of graph methods for unseen users.

**Popularity completely fails on e-commerce cold-start** (0.0000) because the globally popular items are all streaming items — demonstrating the cross-domain generalisation problem.

**Standard metrics mislead** — on Level 3 Hard, Standard MF achieves P@10=0.0281 and NDCG@10=0.0299 which appears competitive with Causal LightGCN (P@10=0.0195). But on Category Precision the gap is massive: 0.2900 vs 0.7216.

---

## 18 Evaluation Splits

| Category | Splits | Purpose |
|----------|--------|---------|
| Standard | train / val / test | Baseline evaluation |
| Cold-start | cold_start | Zero-history users |
| Difficulty | level1 / level2 / level3 | Progressive confounder complexity |
| Seasonal | winter / summer / autumn / spring | Concept drift evaluation |
| Domain | ecom_cold / stream_cold | Cross-domain generalisation |
| Position | high_position / low_position | Position bias isolation |

---

## Reproduce From Scratch

```bash
# Step 1 - Generate full benchmark dataset (~2 minutes)
python benchmark/generate_benchmark.py

# Step 2 - Train all models (~30 minutes on CPU)
python benchmark/train_models.py

# Step 3 - Run full evaluation (~15 minutes)
python benchmark/run_evaluation.py

# Step 4 - Generate publication charts
python benchmark/generate_charts.py
```

---

## Evaluate Your Own Model

```python
from evaluation.metrics import evaluate_model
import pandas as pd

# Load any evaluation split
cold_start = pd.read_csv('data/cold_start.csv')
users = pd.read_csv('data/users.csv')
items = pd.read_csv('data/items.csv')

def my_recommender(user_id, user_info, items_df, k=10):
    # Your recommendation logic here
    return [item_id_1, item_id_2, ...]

results = evaluate_model(
    model_name='MyModel',
    model_func=my_recommender,
    test_data=cold_start,
    users_df=users,
    items_df=items,
    k=10
)

# Returns: precision@10, recall@10, ndcg@10,
#          genuine_p@10, category_p@10
print(results)
```

---

## Pre-trained Models
pretrained_models/
fmf_std_U.npy    Standard MF user embeddings  (35,000 x 32)
fmf_std_V.npy    Standard MF item embeddings  (4,000 x 32)
fmf_caus_U.npy   Causal MF user embeddings    (35,000 x 32)
fmf_caus_V.npy   Causal MF item embeddings    (4,000 x 32)
lgcn_std.pt      Standard LightGCN weights    (2.5M parameters)
lgcn_caus.pt     Causal LightGCN weights      (2.5M parameters)

Training details:
- Standard models: 575,553 training clicks
- Causal models: 172,870 genuine clicks (402,683 biased removed)
- LightGCN: 50 epochs, embedding dim 64, 3 propagation layers
- Item embedding divergence between standard and causal: 0.5126

---

## Project Structure
casbench/
├── README.md
├── requirements.txt
├── data/                      18 evaluation splits (all included)
├── pretrained_models/         6 trained model files (all included)
├── results/                   Evaluation results CSV
├── figures/                   Publication quality charts
├── benchmark/
│   ├── generate_benchmark.py  Regenerate data from scratch
│   ├── train_models.py        Train all models from scratch
│   ├── run_evaluation.py      Run full evaluation
│   └── generate_charts.py     Generate publication figures
├── models/
│   └── fast_mf.py             Vectorised FastMF implementation
└── evaluation/
└── metrics.py             Standard + causal evaluation metrics

---

## Citation

```bibtex
@inproceedings{hassan2026causalrecbench,
  title     = {CausalRec-Bench: A Semi-Synthetic Benchmark for Evaluating
               Causal Cold-Start Recommendation Under Exposure Bias
               and Concept Drift},
  author    = {Hassan, Ali},
  booktitle = {Proceedings of the CONSEQUENCES Workshop at ACM RecSys 2026},
  year      = {2026},
  note      = {Collaborating with Dr. Yan Zhang,
               Charles Darwin University}
}
```

---

## License

Code: MIT License | Dataset: CC BY 4.0 | Pre-trained Models: CC BY 4.0
