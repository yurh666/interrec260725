# ChoiceRec Full 结果说明（最终版）

**代码仓库：** [interrec260725](https://github.com/yurh666/interrec260725)（已含 ChoiceRec 协议层）  
**跑数真源（本地）：** `/home/yrh666/interrec-final-github/experiments/runs/`

| 数据集 | run 目录 | 状态 |
|--------|----------|------|
| LastFM | `choicerec_lastfm_full_seed42/` | 完成 |
| MovieLens | `choicerec_movielens_full_seed42/` | 完成 |
| Book | `choicerec_book_full_seed42/` | 完成 |

**无需重跑。** 下列十二指标已从 `full_log.jsonl` 用 `session_metrics_from_turns` 重算并归档。

---

## 主表十二指标（macro 平均，seed 42）

| 数据集 | Hit@5 | Hit@10 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | MRR@5 | MRR@10 | AR@5 | AR@10 | PR@5 | PR@10 |
|--------|------:|-------:|---------:|----------:|-------:|--------:|------:|-------:|-----:|------:|-----:|------:|
| **LastFM** | 0.4697 | **0.6201** | 0.1025 | 0.1591 | 0.1336 | 0.1525 | 0.2852 | 0.3054 | 7.03 | 4.86 | 0.6566 | 0.7828 |
| **MovieLens** | 0.1363 | **0.2363** | 0.0145 | 0.0284 | 0.0316 | 0.0343 | 0.0671 | 0.0800 | 12.68 | 10.89 | 0.2397 | 0.3636 |
| **Book** | 0.1552 | **0.2402** | 0.0285 | 0.0460 | 0.0363 | 0.0426 | 0.0801 | 0.0911 | 12.75 | 10.90 | 0.3004 | 0.4542 |

机器可读：

- `/home/yrh666/interrec-yrh-outcome/metrics_main12_choicerec.csv`
- `/home/yrh666/interrec-yrh-outcome/metrics_main12_choicerec.json`
- 详表：`../interrec-yrh-outcome/MAIN_TABLE_CHOICEREC_CN.md`

---

## 归档目录（摘要，不含 full_log）

`/home/yrh666/interrec-yrh-outcome/choicerec/{lastfm,movielens,book}/`

含 `run_summary.json`、`metrics.csv`、`report.md`、`config.yaml`、`OUTCOME_PROVENANCE.json`。  
`full_log.jsonl` 体积过大（~7GB 合计），保留在 `interrec-final-github/experiments/runs/` 原路径。

---

## 与 InterRec 260725 的关系

- **InterRec full 主表：** `interrec-yrh-outcome/metrics_main12_interrec260725.csv`
- **ChoiceRec full 主表（本文件）：** 同上口径，协议改为 online valid reveal + 15 turns
- 指标定义：`interrec-yrh-outcome/MAIN_TABLE_METRICS_DEFINITION_CN.md`

---

## 注意

- `run_summary.json` / `metrics.csv` 仅含 HitRate@10、NDCG@10、MRR@10 等少量字段；**十二指标必须以 main12 CSV 为准**。
- PR@10 **不可**从 `metrics_by_turn.csv` 聚合误读（与 InterRec 主表相同约束，见 `PR_METRICS_AUDIT_CN.md`）。
