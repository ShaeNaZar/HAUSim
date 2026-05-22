# HAUSim — Human-Adaptive Unified Simulator

A research framework for simulating language-learner behaviour and training sequence models to predict exercise outcomes. The project combines a **FSRS-based human simulator**, three **exercise-selector strategies**, and two **Transformer architectures** evaluated on both real (Duolingo SLAM) and synthetic data.

---

## Overview

```
HAUSim/
├── src/
│   ├── fsrs_simulator.py    # FSRS memory model + synthetic human factory
│   ├── algorithm.py         # Exercise selectors (Random / Queue / GreedyFSRS)
│   ├── synthetic_dataset.py # Synthetic event stream generator
│   ├── slam_loader.py       # Duolingo SLAM dataset parser → canonical schema
│   ├── sequence_dataset.py  # Vocab building, sliding-window PyTorch dataset
│   ├── models.py            # HSTU / ARGUS model definitions + training (CPU)
│   └── models_gpu.py        # GPU-optimised variants (BF16, Flash Attention, MoE)
├── words/
│   └── ENGLISH_CERF_WORDS.csv   # CEFR-tagged English word list
├── data/                    # SLAM .train file + auto-generated .parquet cache
├── main.ipynb               # End-to-end experiment notebook
└── sim.ipynb                # FSRS session analysis and selector comparison
```

---

## Core Components

### 1. FSRS Human Simulator (`fsrs_simulator.py`)

Implements the **Free Spaced Repetition Scheduler (FSRS-5)** extended with psychologically motivated learner parameters.

**`Word`** — tracks per-word memory state:
| Field | Meaning |
|---|---|
| `stability` (S) | How long the word persists in memory (days) |
| `difficulty` (D) | Intrinsic word difficulty, \[1, 10\] |
| `last_review_ts` | Timestamp of last review |
| `reps` / `lapses` | Consecutive successes / cumulative failures |

**`Human`** — a synthetic learner with individual parameters:
| Parameter | Description |
|---|---|
| `ability` | Language talent multiplier, \[0.5, 1.8\] |
| `base_error_rate` | Background error probability |
| `fatigue_rate` | Recall decay per exercise within a session |
| `interference_sensitivity` | Penalty from semantically similar recent words |
| `typo_rate` | Extra failure probability for typing exercises |
| `native_language_distance` | Controls cognate bonus \[0 = close, 1 = distant\] |

Five exercise types modulate recall probability via logistic difficulty scaling:

| Exercise | Difficulty multiplier |
|---|---|
| `MULTIPLE_CHOICE` | 0.70 |
| `TRANSLATE_EN_RU` | 0.85 |
| `TRANSLATE_RU_EN` | 1.00 (baseline) |
| `TYPING` | 1.30 |
| `LISTENING` | 1.40 |

**`HumanFactory`** samples diverse learner populations from realistic distributions (log-normal ability, beta error rates, gamma fatigue). FSRS weights can be individually perturbed per learner.

**`estimate_erg`** computes **Expected Recognition Growth** — the expected increase in retrievability at a future horizon if a given exercise is attempted now, used by the greedy selector.

---

### 2. Exercise Selectors (`algorithm.py`)

All selectors share the `Selector` base class and track word familiarity (0 → 1 per word). Words are sourced from `words/ENGLISH_CERF_WORDS.csv` (CEFR A1–C2).

| Selector | Strategy |
|---|---|
| `RandomSelector` | Uniform random word + exercise type |
| `QueueSelector` | Round-robin over words; skips mastered words mid-cycle |
| `GreedyFSRSSelector` | Picks the word × exercise pair with maximum ERG |

`QueueGenerator` samples a session vocabulary from the CEFR word list and wraps any selector in a `Queue` with a unified `progress` / `is_finished` interface.

Human ability presets by CEFR level (`A1`–`C2`) govern `ability`, `base_error_rate`, `fatigue_rate`, and `interference_sensitivity`.

---

### 3. Synthetic Dataset Generator (`synthetic_dataset.py`)

`generate_synthetic_events()` drives a population of synthetic users through multi-session learning runs and returns a canonical event `DataFrame`:

```
user_id | word | context: list[str] | action: bool | action_prob: float | timestamp: int
```

`action_prob` is the FSRS recall probability computed at the moment of each exercise — the ground-truth soft target used for the MSE component of the training loss. It is always present for synthetic events and will be `NaN` for Duolingo rows after dataset merging.

Key design choices:
- Users are distributed equally across A1–C2 CEFR levels.
- Session counts follow a **log-normal distribution** (realistic engagement heterogeneity).
- Selectors rotate across sessions (Random → Queue → GreedyFSRS → …).
- A configurable fraction of users (`context_fraction`, default 0.7) receive rich context tokens `[exercise_class, cefr_level]`; the rest receive only `[exercise_class]`.

---

### 4. Duolingo SLAM Loader (`slam_loader.py`)

Parses the [Duolingo SLAM dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO) into the same canonical schema. All SLAM tokens are treated as `PICK_DEFINITION` exercises. Context is the full sentence token list. `action_prob` is not available for SLAM events and will be `NaN`.

A `.parquet` cache is written on first parse (zstd-compressed) so subsequent loads are instant:

```python
from src.slam_loader import load_duolingo_slam
df = load_duolingo_slam("data/en_es.slam.20190204.train")
```

---

### 5. Sequence Dataset (`sequence_dataset.py`)

**`build_dataset(slam_df, synthetic_df, data_cfg, model_cfg)`** merges both event sources and runs the full preprocessing pipeline:

1. Filter users by event count; optionally subsample.
2. Build a shared token vocabulary (words + context tokens).
3. Compute log-bucketed inter-event time deltas.
4. Ensure `action_prob` column exists for all rows — `NaN` for SLAM rows, FSRS probability for synthetic rows.
5. Slice per-user sequences into train / val splits.

Returns `(train_ds, val_ds, vocab)` where each dataset is a **`SequenceWindowDataset`** — a sliding-window PyTorch `Dataset` that pads histories to `max_seq_len` and samples random target positions during training. Each batch item includes `target_prob` (float32, may be NaN) alongside the binary `target_label`.

---

### 6. Models (`models.py` / `models_gpu.py`)

Two causal Transformer architectures share a unified config, embedding scheme, training loop, and save/load API.

**Shared embedding:** every event is represented as  
`word_emb + context_emb (mean-pooled) + action_emb + delta_emb`

**Shared loss (both architectures, both CPU and GPU variants):**
- **BCE** — binary cross-entropy on `action` (did the user recall correctly?)
- **MSE** — regression of `sigmoid(logit)` onto `action_prob` where not `NaN`; skipped automatically for Duolingo rows

#### HSTU
*Hierarchical Sequential Transduction Unit.* Replaces softmax attention with a **SiLU-gated linear attention** variant and adds a **relative attention bias** (log-scaled position buckets). Designed for high-throughput recommendation.

GPU variant (`hstu_gpu`) adds: RoPE, SwiGLU FFN, sparse MoE layers, stochastic depth, gradient checkpointing, RMSNorm.

#### ARGUS
*Two-tower architecture* with independent **user tower** (causal Transformer) and **item tower** (lightweight MLP). Trained with a triple loss:
- **FP loss** — binary cross-entropy on `P(correct)`.
- **MSE** — soft regression on `action_prob` where available.
- **NIP loss** — in-batch contrastive (softmax over cosine similarities).

GPU variant (`argus_gpu`) adds: Flash GQA, hard-negative NIP re-weighting, deeper item tower, sparse MoE, stochastic depth.

```python
from src.models import build_model, ModelConfig, TrainConfig, train, save_model, load_model

cfg   = ModelConfig(arch='argus', d_model=64, n_heads=2, n_layers=2)
model = build_model(cfg, vocab)
history = train(model, train_ds, val_ds, TrainConfig())
save_model(model, 'model.pt')
```

**Inference:**

```python
from src.models import load_model, HSTUPredictor, HistoryEvent

model     = load_model('model.pt')
predictor = HSTUPredictor(model)

p = predictor.predict_action(history, context=["TYPING", "B1"], new_word="ubiquitous")
scores = predictor.predict_action_batch(history, [("word_a", ctx_a), ("word_b", ctx_b)])
```

---

## Quick Start

### 1. Download the SLAM dataset

```bash
python -m src.download
```

This fetches the Duolingo SLAM archive from Harvard Dataverse, decompresses it into `data/`, and writes a fast `.parquet` cache on first load.

### 2. Run the notebook

Open `main.ipynb` for an end-to-end experiment: data loading → synthetic generation → recognition analytics → model training → evaluation.

The notebook auto-detects CUDA and dispatches to the appropriate model family:
- **CPU / no CUDA** → `ModelConfig` + `train` (lightweight, runs anywhere)
- **CUDA** → `ModelConfigGPU` + `train_gpu` (BF16 AMP, fused AdamW, gradient accumulation)

To override manually, set `USE_GPU = True/False` before the config cell.

---

## License

[Apache 2.0](LICENSE)
