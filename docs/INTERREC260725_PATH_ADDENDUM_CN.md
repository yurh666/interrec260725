# interrec260725 相对 Pipeline 手册的路径更新

本包 **Method** 见根目录 `METHOD_CN.md`（`BAYESIAN_MEMORY_TRANSIENT_SLOTS_METHOD_CN` 全文）。  
**Pipeline（含核心代码文件/函数对照）** 见根目录 `PIPELINE_CN.md`（`FINAL_INTERREC_PIPELINE_CODEBOOK_CN` 全文）。

跑数请以本包为准（非文中旧的 `final_results_seed42` / `final_book_full_noapi`）：

| 数据集 | Full 配置 |
|--------|-----------|
| Book | `configs/runs/book_full_integrated_distill_surface_seed42.yaml` |
| MovieLens | `configs/runs/movielens_full_integrated_distill_surface_seed42.yaml` |
| LastFM | `configs/runs/lastfm_full_integrated_distill_surface_seed42.yaml` |

**260725 唯一 intentional 代码差异**：`surface_writer.mode = local_small_model` + **`outputs/intent_teacher_v2/lora_adapter`（Intent Teacher v2）**（见 `configs/INTERREC_CANONICAL_SPEC_20260725.md`）。Pipeline 各阶段 **文件与函数名不变**。

代码根目录请读：`/home/yrh666/interrec-yrh`（或你解压后的 `interrec-yrh/`）。
