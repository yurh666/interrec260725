# interrec260725 Full 结果文件说明

主表数字与 **逐 session / 逐 turn 是否命中推荐** 的真源如下（与 surface 用 legacy 或 v2 **无关**——已归档 run 的 hypothesis 链已是最终版；surface 字段可能为 `mock_template`）。

## 归档目录（推荐对外引用）

| 数据集 | 目录 | 状态 |
|--------|------|------|
| Book | `/home/yrh666/interrec-yrh-outcome/book/` | 完整（2107 sessions） |
| LastFM | `/home/yrh666/interrec-yrh-outcome/lastfm/` | 完整（1648 sessions） |
| MovieLens | `/home/yrh666/interrec-yrh-outcome/movielens/` | 跑完后写入；进行中见下方 live run |

## 每个数据集目录内关键文件

| 文件 | 用途 |
|------|------|
| **`run_summary.json`** | **主表 aggregate**：`aggregate_metrics.HitRate@10`、`NDCG@10`、`MRR@10`、`avg_ask_count`、`n_sessions` |
| **`metrics.csv`** | 与 summary 一致的汇总行 |
| **`metrics_by_turn.csv`** | 按 turn 聚合的指标（非逐 session） |
| **`full_log.jsonl`** | **逐 session × turn 全量日志**（体积大） |
| `config.yaml` | 该 run 的 resolved 配置 |
| `report.md` | 人类可读摘要 |
| `OUTCOME_PROVENANCE.json` | 从哪次 `experiments/runs/...` 拷贝 |

## `full_log.jsonl` 里「有没有推荐中」看什么

每行一条 **turn**（JSON），常用字段：

- `episode_id` / `user_id`：session 标识  
- `turn`：1…15  
- **`recommendation`**：该轮 **提问前** Top-10 推荐  
  - `recommended_items`：item id 列表  
  - **`HitRate@10` / `HitRate@5`**：该 turn 是否在 Top-K 命中 θ*（0/1）  
- **`final_recommendation_corrected`**：belief 更新后 **该轮最终** 推荐的 Hit 指标（同样含 `HitRate@10` 等）  
- `ask_decision`：`do_ask`、`voi`、`ig`  
- `user_choice`：模拟用户选的 option  
- `implementation_modes.surface_writer`：surface 实际模式（`mock_template` / `local_small_model` 等）

**按 session 汇总 Hit@10**：需对同一 `episode_id` 的 turn 行聚合（通常取各 turn 的 `final_recommendation_corrected.HitRate@10` 或按论文脚本定义）；仓库 aggregate 已在 `run_summary.json` 算好。

## Live / 源 run 路径（拷贝前）

| 数据集 | 源目录（interrec-final-github） |
|--------|----------------------------------|
| Book | `experiments/runs/book_full_integrated_distill_surface_seed42/` |
| LastFM | `experiments/runs/lastfm_full_integrated_distill_surface_seed42/` |
| MovieLens（进行中） | `experiments/runs/movielens_full_integrated_distill_surface_seed42/`（完成后应有 `run_summary.json`） |

Book 归档来源见 `book/OUTCOME_PROVENANCE.json`。

## 当前主表数字（aggregate，来自 outcome）

- **Book** Hit@10 约 **0.2074**（2107 users）  
- **LastFM** Hit@10 约 **0.5661**（1648 users）  
- **MovieLens**：待 full 完成并拷入 `interrec-yrh-outcome/movielens/`
