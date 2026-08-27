# ChoiceRec

ChoiceRec extends the InterRec **260725 full** stack (integrated + γ=0.05 + Intent Teacher v2 surface LoRA) with an **online valid-reveal interaction protocol**.

Built on [interrec260725](https://github.com/yurh666/interrec260725); the recommender / Frontier / VOI / belief / memory pipeline is unchanged — ChoiceRec adds protocol hooks and configs.

## Docs

- **ChoiceRec protocol audit**：`CHOICEREC_PROTOCOL_AUDIT_CN.md`
- **InterRec method (full paper text)**：`METHOD_CN.md`
- **Pipeline codebook**：`PIPELINE_CN.md`
- **260725 path addendum**：`docs/INTERREC260725_PATH_ADDENDUM_CN.md`
- **Config index (InterRec full)**：`configs/final_unified/manifest_yrh.yaml`

## Quick start (ChoiceRec full, seed 42)

```bash
pip install -r requirements.txt

# LastFM → MovieLens → Book (serial)
bash scripts/run_choicerec_full_serial.sh

# Or single dataset
python3 scripts/run_main_experiment.py \
  --config configs/choicerec/book_full_seed42.yaml \
  --run-id choicerec_book_full_seed42
```

Configs inherit InterRec full mock-surface settings and set `simulation.protocol: choicerec`, `max_turns: 15`.

## What ChoiceRec adds

| Component | Path |
|-----------|------|
| Protocol state machine | `src/choicerec/protocol.py` |
| Dynamic θ* | `src/choicerec/theta_builder.py` |
| Subexperiment helpers | `src/choicerec/subexperiment_runtime.py` |
| Session integration | `src/simulation/session_simulator.py` |
| Full configs | `configs/choicerec/*_full_seed42.yaml` |

## InterRec full (unchanged)

```bash
python3 scripts/run_final_interrec.py --dataset book
```
