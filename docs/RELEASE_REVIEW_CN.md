# interrec260725 发布复核（interrec-yrh → GitHub）

**复核日期：** 2026-07-29  
**对照文档：** `METHOD_CN.md`、`PIPELINE_CN.md`

## 结论

**代码主链与 Method / Pipeline 一致，可作为 [interrec260725](https://github.com/yurh666/interrec260725) 发布包。** 下列为已核对项与已知边界（不阻塞发布）。

## 已核对（与 PIPELINE §0–§5、§7 一致）

| 项 | 结果 |
|----|------|
| CLI / 配置 | `run_main_experiment.py`、`load_config` 继承链；三数据集 `*_full_integrated_distill_surface_seed42.yaml` 可 resolve |
| `semantic_backend.implementation_mode` | 三数据集均为 `frontier_clustered_v32_integrated_representative_item` |
| Hypothesis 路径 | `session_simulator` → `run_frontier_clustered_v32_hypotheses_turn` |
| Integrated 三角色 | `integrated_representative_backend.py`，γ=0.05 |
| VOI / ask | Book `when_to_ask=full_voi`；ML/LF 未设 entropy-only → full MC VOI；`voi_gate.enabled=false`；`use_mock_llm=true` |
| `include_none` | 三数据集 canonical 均为 **true** |
| Surface 顺序 | `apply_choice_belief_update` **之后** `SurfaceWriter.write_question`（offline post-hoc surface） |
| LoRA canonical | `outputs/intent_teacher_v2/lora_adapter`（见 `docs/SURFACE_WRITER_LORA_PROVENANCE_CN.md`） |
| 数据 manifest | `data/DATA_MANIFEST.json` 与三数据集 `processed_dir` / PCA 嵌入路径一致 |

## 已知边界（文档已说明或非主链）

| 项 | 说明 |
|----|------|
| `scripts/verify_smoke10_vs_canon.py` | 依赖 `experiments.in_depth`（完整 in-depth 仓才有）；**本 release 包未含**，脚本在本目录下会 `ModuleNotFoundError`；不影响 full 跑法 |
| `interrec_variants/`（PIPELINE §11） | 效率加速包装层未打入 yrh 精简包；主表 full 不依赖 |
| `models/Qwen2.5-1.5B-Instruct` | 本地 symlink 至大模型，**不随仓库推送**；推理需自行下载（`outputs/intent_teacher_v2/BASE_MODEL_SETUP.md`） |
| `outputs/surface_writer_qwen/` | Legacy LoRA，非 canonical v2；保留作对照 |
| Full 主表数字 | 归档在 `../interrec-yrh-outcome/`（见 `docs/OUTCOME_FILES_CN.md`），不在本仓库 mandatory |

## 与 METHOD 叙事对齐（抽查）

- §5–§8 Frontier pool → cluster → contrast → Preserve/Boundary/Explore：实现于 `frontier_clustered_v32.py` + `integrated_representative_backend.py`  
- §9–§12 向量化、IG/VOI、Laplace update：与 `session_simulator` 调用顺序一致  
- §13.4.1：260725 主表 HRV gate **关闭**，与 resolved yaml 一致  
- §9 离线 surface：Method 已写明 post-hoc surface，与代码顺序一致  

## 发布物清单（Git）

- 源码 `src/`、`scripts/`、`configs/`  
- `data/processed/`（三数据集预处理 + PCA/BGE 嵌入）  
- `outputs/intent_teacher_v2/`（canonical LoRA + inference_config）  
- `outputs/surface_writer_qwen/`（legacy，可选对照）  
- 文档 `METHOD_CN.md`、`PIPELINE_CN.md`、`docs/`、`README_CN.md`  
