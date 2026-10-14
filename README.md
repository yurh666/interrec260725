# ChoiceRec

ChoiceRec is a framework for interactive recommendation through multiple-choice preference elicitation. It maintains a Bayesian preference state, identifies uncertain directions near the recommendation frontier, and selects clarification questions according to their expected value of information. Choice feedback and newly observed behavior update the preference state across turns, which is then used to produce personalized Top-K recommendations.

This repository provides the implementation, experiment configurations, processed datasets, precomputed item embeddings, and supporting documentation for offline experiments on LastFM, MovieLens, and Amazon Book.

**Anonymous review artifact:** <https://anonymous.4open.science/r/interrec260725-43BB/>

## Method overview

ChoiceRec connects recommendation, question selection, and preference updating in one interaction loop:

1. **Maintain Bayesian preference memory.** Represent the user's latent preference by a Gaussian state with mean `mu` and covariance `Sigma`, initialized from the visible interaction history.
2. **Rank items and construct a recommendation frontier.** Combine preference alignment with uncertainty-aware exploration, then identify candidate items that are informative for the current ranking decision.
3. **Build transient intent slots.** Cluster frontier items and construct three complementary hypothesis roles: **Preserve**, **Boundary**, and **Explore**. These slots are regenerated from the current state at each turn.
4. **Decide whether to ask.** Evaluate the information value of the joint multiple-choice question against the configured asking cost. Ordinary choices and a NONE option support preference elicitation.
5. **Update from choices and behavior.** Apply the choice-based posterior update and incorporate behavioral observations as they become available. The behavioral update preserves choice-derived information while updating the preference mean.
6. **Produce recommendations.** Use the updated state to rank unseen candidate items and evaluate the terminal Top-K list.

The implementation retains the `interrec` name in several modules and command-line options. The final ChoiceRec experiment protocol is selected by `simulation.protocol: choicerec` in `configs/choicerec/`.

### Interaction and evaluation protocol

The released ChoiceRec configurations use a maximum of **15 turns**, **Top-K = 10**, and **seed 42**.

- The initial visible behavioral history is `observed_history + future_train`, concatenated in chronological order.
- `future_valid` interactions are revealed in chunks at the end of successive turns and incorporated into the behavioral history.
- Choice feedback updates the Bayesian preference state; behavioral reveals update its mean while preserving its covariance.
- The reference user simulator constructs its current latent target from the remaining unrevealed validation interactions and the test interactions. It generates responses using deterministic argmax.
- Recommendation metrics are evaluated against **`future_test` only**.
- After the final turn's behavioral reveal, the system produces the terminal recommendation list.

The policy-visible history and simulator-only target are documented in [the protocol audit](CHOICEREC_PROTOCOL_AUDIT_CN.md). The mathematical background and module-level explanation are available in [METHOD_CN.md](METHOD_CN.md) and [PIPELINE_CN.md](PIPELINE_CN.md).

## Repository structure

| Path | Contents |
|---|---|
| `src/choicerec/` | Interaction protocol, behavioral reveals, simulator target, and experiment helpers |
| `src/belief/` and `src/memory/` | Preference state, initialization, posterior updates, and memory utilities |
| `src/ask/` | Frontier construction, hypothesis selection, information gain, and asking decisions |
| `src/agents/` | Semantic hypothesis construction and representative-item integration |
| `src/recommendation/` | Item scoring, ranking, and recommendation metrics |
| `src/simulation/` | Session execution and experiment orchestration |
| `src/logging/` | Run logs, aggregate metrics, and reports |
| `configs/choicerec/` | Final ChoiceRec full-run and smoke-run configurations |
| `scripts/` | Experiment entry points and utilities |
| `data/processed/` | Processed interactions, sessions, item metadata, and embeddings |
| `outputs/intent_teacher_v2/` | Optional learned surface-writer adapter and setup instructions |
| `docs/` | Method, configuration, results, and model provenance notes |
| `experiments/runs/` | Output directory created by experiment commands |

## Installation

Use **Python 3.10 or newer**. Run all commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. On systems where Python is named `python3`, use that command in place of `python`.

The default ChoiceRec runs reuse the supplied item embeddings and use `llm.mode: mock` with `surface_writer.mode: mock_template`. This execution path does not require an external LLM API key or the optional local language-model weights.

### Data and embeddings

| Dataset | Processed directory | Embedding file | Dimension |
|---|---|---|---:|
| LastFM | `data/processed/lastfm_bge_m3_pca128/` | `item_embeddings.npy` | 128 |
| MovieLens | `data/processed/movielens/` | `item_embeddings_pca256.npy` | 256 |
| Amazon Book | `data/processed/amazon_book/` | `item_embeddings_pca128.npy` | 128 |

The data locations are recorded in [data/DATA_MANIFEST.json](data/DATA_MANIFEST.json). Keep the processed data and configuration directories together when extracting or copying the artifact. Use the released embeddings and sessions to reproduce the configured experiments.

## Quick start

### Run a three-session smoke experiment

The LastFM smoke configuration selects three fixed sessions and exercises the ChoiceRec protocol:

```bash
python scripts/run_main_experiment.py \
  --config configs/choicerec/lastfm_smoke3_seed42.yaml \
  --seed 42 \
  --run-id choicerec_lastfm_smoke3_seed42
```

Outputs are written to `experiments/runs/choicerec_lastfm_smoke3_seed42/`. Use this short run to inspect the environment, interaction records, and output format. Full-dataset scores require the full configurations below.

### Run the full experiments

Run each dataset directly:

```bash
python scripts/run_main_experiment.py \
  --config configs/choicerec/lastfm_full_seed42.yaml \
  --seed 42 \
  --run-id choicerec_lastfm_full_seed42

python scripts/run_main_experiment.py \
  --config configs/choicerec/movielens_full_seed42.yaml \
  --seed 42 \
  --run-id choicerec_movielens_full_seed42

python scripts/run_main_experiment.py \
  --config configs/choicerec/book_full_seed42.yaml \
  --seed 42 \
  --run-id choicerec_book_full_seed42
```

Alternatively, the Bash wrapper runs LastFM, MovieLens, and Book sequentially. It resumes runs with existing logs and skips runs with a completion marker:

```bash
bash scripts/run_choicerec_full_serial.sh
```

The wrapper requires a Bash environment. For PowerShell, place each direct Python command on one line.

### Resume an interrupted run

Use the same configuration and existing run ID with `--resume`:

```bash
python scripts/run_main_experiment.py \
  --config configs/choicerec/book_full_seed42.yaml \
  --seed 42 \
  --run-id choicerec_book_full_seed42 \
  --resume
```

## Experiment outputs

Each run writes files under `experiments/runs/<run-id>/`:

| File | Purpose |
|---|---|
| `config.yaml` | Resolved configuration for the run |
| `run_summary.json` | Run summary and aggregate metrics |
| `metrics.csv` | Aggregate recommendation metrics |
| `metrics_by_turn.csv` | Logged metrics across interaction turns |
| `full_log.jsonl` | Detailed session and interaction records |
| `report.md` | Generated experiment report |

Metric availability depends on the logging and evaluation configuration. The results note documents a twelve-metric analysis reconstructed from full interaction logs; its definitions and aggregation should be followed when comparing those reported values.

### Recorded results

The accompanying [ChoiceRec results note](docs/CHOICEREC_OUTCOME_CN.md) records the following terminal metrics for full runs with seed 42:

| Dataset | Hit@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| LastFM | 0.6201 | 0.1525 | 0.3054 |
| MovieLens | 0.2363 | 0.0343 | 0.0800 |
| Amazon Book | 0.2402 | 0.0426 | 0.0911 |

These are documented results from the recorded runs. The repository's `experiments/runs/` starts as an output directory; historical full logs and external result archives mentioned in older notes are outside this release. The commands above generate new run outputs for inspection and analysis.

## Optional learned question rendering

The release also contains an Intent Teacher v2 surface-writer adapter under `outputs/intent_teacher_v2/`. Local model rendering uses additional dependencies and a separately obtained Qwen2.5-1.5B-Instruct base model.

See [the base-model setup guide](outputs/intent_teacher_v2/BASE_MODEL_SETUP.md), [adapter provenance](docs/SURFACE_WRITER_LORA_PROVENANCE_CN.md), and `requirements-surface-distill.txt` for setup details. The final configurations in `configs/choicerec/` use template rendering. In the documented pipeline, learned surface text is produced after choice-based updating and is logged separately from the vector-based simulated choice and asking decision.

## Further documentation

The detailed notes are primarily in Chinese:

- [Method derivation](METHOD_CN.md)
- [Implementation pipeline](PIPELINE_CN.md)
- [ChoiceRec interaction protocol and data boundary](CHOICEREC_PROTOCOL_AUDIT_CN.md)
- [Recorded ChoiceRec results](docs/CHOICEREC_OUTCOME_CN.md)
- [Release contents and known boundaries](docs/RELEASE_REVIEW_CN.md)
- [Learned surface-writer provenance](docs/SURFACE_WRITER_LORA_PROVENANCE_CN.md)

This artifact supports offline experiments with a reference user simulator. Its reported metrics should be interpreted under the released simulation protocol and experiment configurations.
