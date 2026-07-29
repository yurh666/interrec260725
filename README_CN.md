# interrec260725

InterRec **最终版**（interrec260725）：integrated + γ=0.05 + **Intent Teacher v2** surface LoRA。

- **Method（论文方法全文）**：`METHOD_CN.md`
- **Pipeline（核心代码对照手册）**：`PIPELINE_CN.md`
- **发布复核**：`docs/RELEASE_REVIEW_CN.md`
- **260725 跑数路径补充**：`docs/INTERREC260725_PATH_ADDENDUM_CN.md`
- **配置索引**：`configs/final_unified/manifest_yrh.yaml`
- **Full 结果（外链）**：`../interrec-yrh-outcome/`（`docs/OUTCOME_FILES_CN.md`）

## 快速开始

```bash
pip install -r requirements.txt
# LoRA 推理需基座：见 outputs/intent_teacher_v2/BASE_MODEL_SETUP.md

python3 scripts/run_final_interrec.py --dataset book
# 或
python3 scripts/run_main_experiment.py \
  --config configs/runs/book_full_integrated_distill_surface_seed42.yaml --seed 42
```

## 数据与 LoRA

- 预处理数据：`data/processed/`（见 `data/DATA_MANIFEST.json`）
- Canonical LoRA：`outputs/intent_teacher_v2/lora_adapter/`
