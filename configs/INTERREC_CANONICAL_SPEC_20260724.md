# InterRec 三数据集 canonical 规范（2026-07-24 最终版）

> **代码锚点**：`git rev-parse HEAD` → 文档撰写时仓库 `897c01caa2e222e44f2404fe52f39b7527e39531`（以实际跑数目录内 `git_commit.txt` 为准）。  
> **范围**：除 **surface 选择题文案**（`surface_writer` / mock template）外，假设生成、cluster-bound、integrated 代表 item、belief、推荐与 memory 兼容层。  
> **ML/LF 已验路径**：`scripts/verify_smoke10_vs_canon.py` 对前 10 session **逐 session 与主表一致**（integrated 路径）。

---

## 1. 权威路径一览（跑数 / 主表 / 配置）

| 角色 | MovieLens | LastFM | Book |
|------|-----------|--------|------|
| **论文主表（当前有效 · interrec260725）** | `experiments/runs/movielens_full_integrated_distill_surface_seed42/`（跑完后） | 同左 | `experiments/runs/book_full_integrated_distill_surface_seed42/` |
| **过渡 / 废弃（勿作子实验对照）** | — | — | `book_full_integrated_mock_surface_seed42`（260724 mock）；旧 `final_results_seed42/book`（Hit@10≈0.259） |
| **主表配置快照** | `…/movielens/config.yaml` | `…/lastfm/config.yaml` | 跑完以 run 目录 `config.yaml` 为准 |
| **复现用 run yaml（hit@0.25 + distill 链）** | `configs/runs/movielens_hit025_surface_distill_only_seed42.yaml` | `configs/runs/lastfm_hit025_surface_distill_only_seed42.yaml` | `configs/runs/book_hit025_surface_distill_only_seed42.yaml` |
| **10-session smoke（与主表对齐验证）** | `configs/runs/smoke_movielens_canonical_mock10_seed42.yaml` | `configs/runs/smoke_lastfm_canonical_mock10_seed42.yaml` | （Book 全量即 FINAL；无单独 smoke 文件） |
| **Book 全量 FINAL（integrated，无 lazy cap）** | — | — | `configs/runs/book_full_integrated_mock_surface_seed42.yaml` |
| **统一入口 manifest** | `configs/final_unified/manifest.yaml` | 同左 | 同左 |
| **入口脚本** | `scripts/run_main_experiment.py` | 同左 | 续跑：`scripts/run_book_full_integrated_mock_surface.sh`（tmux：`scripts/run_book_full_integrated_mock_surface_tmux.sh`） |

**废弃 / 仅历史参照（勿与当前 Method 混用）**

- Book 旧主表 `experiments/final_results_seed42/book/`：`preserve_reference_route: true` + `frontier_clustered_v32_cluster_bound_v2_mock`，Hit@10≈0.259，**不是** integrated 链。
- **作废**：勿再对照 Hit@10≈0.2553 的 γ 污染 mock 主表口径；子实验以 **interrec260725 distill full** 为准（Book 已与 integrated mock 指标一致，Hit@10≈0.2074）。

---

## 2. 运行时硬约束（三数据集 MUST，log 可审计）

| 约束项 | 要求 | 审计字段 / 说明 |
|--------|------|-----------------|
| 假设生成模式 | `ask.hypothesis_generation.mode = frontier_clustered_v32` | — |
| Integrated 开关 | `frontier_clustered_v32.integrated = true` | — |
| Mock 假设 LLM | `frontier_clustered_v32.use_mock_llm = true` | `implementation_modes.llm = mock_llm` |
| **禁止** contrastive mock 主链 | `semantic_backend.implementation_mode = frontier_clustered_v32_integrated_representative_item` | `hypothesis_generation_mode_resolved` 同上 |
| **禁止** cb-v2 mock fallback | `semantic_backend.allow_cluster_bound_v2_mock_fallback = false` | 不得长期 `mock_contrastive_v2` 占主导 |
| Integrated 路由 | `semantic_backend.preserve_reference_route = false`（Book FINAL 显式；ML/LF 等效为未启用 reference mock 链） | — |
| Cluster-bound v2 | `cluster_bound_generation_v2_enabled = true` | log 中 cb-v2 成功、prebound 向量 |
| 代表 item 检索（fallback 路径） | `ask.representative_retrieval.mode = legacy` | prebound 成功时与 ML/LF 行为一致 |
| Integrated 代表打分 | `semantic_backend.type = integrated_representative_item`，`variant = frontier_aware` | 代码：`_semantic_backend_cfg` 合并 root+nested |
| α/β/γ（合并后） | **α=0.7, β=0.3, γ=0.05** | nested 覆盖 root；Book 已在 yaml 修正 nested γ |
| VOI 主门 | `ask.voi_gate.enabled = false` | 与主表一致 |
| 无 API | `network_guard.expected_api_calls = 0`，`api_backend.enabled = false` | — |
| 随机种子 | `seed: 42` | — |
| 仿真 | `simulation.split = test`，`max_turns = 15`，`user_choice_mode = deterministic_argmax` | — |

**Smoke 验证命令（ML/LF）**

```bash
python3 scripts/verify_smoke10_vs_canon.py --dataset movielens --run-dir experiments/runs/<smoke_run_id>
python3 scripts/verify_smoke10_vs_canon.py --dataset lastfm --run-dir experiments/runs/<smoke_run_id>
```

---

## 3. 严格对照表：Method 对齐 vs 允许差异

图例：**U** = 三数据集一致（或 Book 缺键时代码默认与 ML 相同）；**D** = 仅数据/候选域；**P** = 协议刻意不同（须在论文/Method 写明）；**S** = 仅 surface/mock 文案（不计入推荐指标链）。

### 3.1 核心方法链（应对齐 → 当前状态）

| # | 配置键 / 行为 | ML canon | LF canon | Book FINAL | 状态 |
|---|----------------|----------|----------|------------|------|
| 1 | `semantic_backend.implementation_mode` | integrated | integrated | integrated | **U** |
| 2 | `use_mock_llm` | true | true | true | **U** |
| 3 | `allow_cluster_bound_v2_mock_fallback` | false | false | false | **U** |
| 4 | `representative_retrieval.mode` | legacy | legacy | legacy | **U** |
| 5 | 合并后 `semantic_backend` γ | 0.05 | 0.05 | 0.05（2026-07-24 修正 nested） | **U** |
| 6 | `c_ask` / `none_threshold` / `mc_samples` | 0.35 / 0.25 / 128 | 同左 | 同左 | **U** |
| 7 | `K_directions` / `M_hypotheses` / `n_options` | 5 / 5 / 3 | 同左 | 同左 | **U** |
| 8 | `voi_gate.enabled` | false | false | false | **U** |
| 9 | `require_*` 特征约束（缺省） | true（yaml 显式） | 同左 | 缺省 → 代码默认 true | **U** |
| 10 | `memory.mode` + `effect_strength` | compatibility / 0 | 同左 | 同左 | **U** |
| 11 | `belief.init_mode` | weighted_average | 同左 | 同左 | **U** |
| 12 | `recommendation.top_k` | 10 | 10 | 10 | **U** |

### 3.2 数据集固有（允许不同 **D**）

| 项 | MovieLens | LastFM | Book |
|----|-----------|--------|------|
| `dataset.name` / `processed_dir` | movielens pack | lastfm bge_m3_pca128 | amazon_book pack |
| `embedding.dim` / `precomputed_path` | 256 / `…/item_embeddings_pca256.npy` | 128 / manifest 链 | 128 / `…/item_embeddings_pca128.npy` |
| `cluster_bound_domain` | `movielens_movie` | `lastfm_music` | `amazon_book` |
| Session 数 | 4731 | 1648 | 2107 |

### 3.3 协议 / Gate（刻意不同 **P** — 与 ML/LF 主表并存）

| 项 | ML / LF（`final_results_seed42`） | Book FINAL | 说明 |
|----|-----------------------------------|------------|------|
| `lazy_generation_enabled` | **true** | **false** | Book 全量 VoI 生成，无 lazy cap |
| `pre_gate_enabled` / `pre_gate` | **true** | **false** | Book 不做 pre-gate 截断 |
| `enable_cluster_bound_v2_entropy_ask_gate` | **true** | **false** | Book 关闭 entropy ask gate |
| `when_to_ask` | 未显式（默认链） | **`full_voi`** | Book 15-turn 全 VoI 协议 |
| `max_v32_asks_per_session` 等 | 5（lazy  sanity） | 无同等 cap | 影响 **ask 次数**，不是假设公式 |
| `frontier_top_k` / `boundary_rank_end` | 500 / 100（ML）；LF frontier 100 | 1500 / 1500 | **候选/frontier 规模 D+P** |
| `ask.neutrality.mode` | llm | **mock** | 选项中立性检查；假设仍 mock LLM |

### 3.4 明确排除（**S** — 当前三数据集 mock surface 跑法）

| 项 | 当前 FINAL mock 跑法 | 正式 distill 目标 |
|----|----------------------|-------------------|
| `surface_writer.mode` | `mock_template` | `local_small_model` + LoRA（hit025 基座） |
| 对 Hit/NDCG 主链影响 | **无**（仅 MCQ 表面句） | 换 surface 后需再验一遍 |

---

## 4. Book 是否与 Method + ML/LF smoke 对齐？（结论）

**是 — 在下列前提下：**

1. **假设 / 推荐主链**：Book FINAL yaml 与 ML/LF 在 §3.1 一致；运行时 log 为 `frontier_clustered_v32_integrated_representative_item`，cb-v2 prebound，**不是**旧 book mock 主表路由。  
2. **刻意不对齐部分**仅 §3.3（ask 预算、frontier 规模、entropy gate）与 §3.2（embedding / 域），应在 Method 中单独一句，**不应**称为「代码路径不一致」。  
3. **Surface** §3.4 为 mock，三数据集当前均为 mock_template 快跑；与 smoke 设计一致。  
4. **2026-07-24 修正**：Book `book_hit025` 内 nested `semantic_backend.gamma` 由 `0.5` 改为 `0.05`，与 ML/LF 合并语义一致（否则 `_semantic_backend_cfg` 会错误覆盖）。

**否 / 勿混淆：**

- 用 `experiments/final_results_seed42/book` 代表当前 InterRec integrated 方法。  
- `use_mock_llm: false` 或 `preserve_reference_route: true` 跑 Book 全量。

---

## 5. 配置加载与续跑规则（Book）

1. **唯一 FINAL 全量配置**：`configs/runs/book_full_integrated_mock_surface_seed42.yaml`（继承 `book_hit025_*`，并 override integrated + 关 lazy/pre_gate + mock surface）。  
2. **勿**对同一 run 使用裸 `--resume` 且无 `episode_ids` 过滤（会重复 session）。  
3. **续跑**：`scripts/run_book_full_integrated_mock_surface.sh` → 已完成 15-turn 的 session 写入 `episode_ids` 排除 + `--resume` 重算全量 metrics。  
4. **跑完 promote**：将 `experiments/runs/book_full_integrated_mock_surface_seed42/` 定为 Book canonical，并更新 `configs/final_unified/manifest.yaml` 中 `canonical_results`（若与 unified 目录同步）。

---

## 6. γ 参数未对齐根因（2026-07-24 核查）

**结论：确曾未与 ML/LF 对齐，不是误报。**

| 层级 | ML/LF `*_hit025` | Book（FULL 开跑前） | 运行时 `_semantic_backend_cfg` |
|------|------------------|---------------------|--------------------------------|
| 顶层 `semantic_backend.gamma` | 0.05 | `book_full` override → **0.05** | 先加载 root |
| nested `fc32.semantic_backend.gamma` | **0.05**（含 α/β） | `book_hit025` 长期为 **0.5**（Book 旧 benchmark / frontier 习惯） | **nested 覆盖 root** → 实际 **0.5** |
| 跑数目录快照 | — | `experiments/runs/book_full_integrated_mock_surface_seed42/config.yaml` L71 `gamma: 0.5` | 与上表一致 |

代码：`src/agents/frontier_clustered_hypothesis_integration.py` 中 `out = {**root, **nested}`，故只改顶层 0.05 **无效**，必须改 nested（或删 nested 的 γ）。

**历史原因**：Book 多条 yaml（`book_benchmark_overlay`、`frontier_aware_book_full` 等）沿用 **γ=0.5**；ML/LF hit025 在 integrated 化时显式写成 **0.05**。`book_full_integrated_*` 合并时只 override 了顶层 `implementation_mode` / αβγ，**漏改 nested**，导致全量 run 仍按 0.5 选代表 item。

**修正**：`book_hit025` nested + 顶层 γ 均改为 0.05；A/B 见 `scripts/compare_book_gamma_first100.py` 与 backup log `…_pre_gamma005_rerun_20260724/`。

---

## 8. 实时监控脚本（260724 Book full）

| 用途 | 命令 |
|------|------|
| **推荐：刷新进度 + 部分 Hit/NDCG/PR/AR** | `cd /home/yrh666/interrec-final-github && bash scripts/watch_canonical_full_mock_surface_progress.sh` |
| 单次快照（不循环） | `python3 scripts/_canonical_full_mock_surface_metrics.py --dataset book --run-dir experiments/runs/book_full_integrated_mock_surface_seed42` |
| 交叉审计 ML/LF/Book | `python3 scripts/audit_interrec_260724_cross_dataset.py` |
| γ 前 100 对比报告 | `experiments/runs/book_full_integrated_mock_surface_seed42/reports/book_gamma_ab_first100_20260724.md` |

底层：`watch_*` 每 20s 调 `_canonical_full_mock_surface_metrics.py`；主表参照行来自 `final_results_seed42`（Book 行为 **旧 mock 主表**，仅历史对照）。

---

## 9. 260724 终审结论（除 mock / gate / 域参数外）

**与 ML/LF `final_results_seed42` 同版（integrated + mock LLM 假设链）**：`implementation_mode`、`use_mock_llm`、`repr=legacy`、合并后 **γ=0.05**、memory compatibility、c_ask/voi_gate 等已对齐；当前 run `config.yaml` nested **γ=0.05**；log 100% `frontier_clustered_v32_integrated_representative_item`。

**已修复的隐藏不对齐**：nested γ 0.5 覆盖顶层（§6）；fresh full 自 2026-07-24 15:54 起为正确配置。

**yaml 少写、运行等价 ML**：`require_*` 约束（代码默认 True）；`enable_oos_contrastive_v2_skip`（Book 缺省 False，仅 lazy∧pre_gate 时生效，Book 皆关）；**neutrality** 在 `llm.mode=mock` 下均走 `OptionBalancer`。

**刻意不同**：lazy/pre_gate/entropy gate、`when_to_ask=full_voi`、frontier/candidate 规模、embedding/域。

---

## 10. 变更记录

| 日期 | 变更 |
|------|------|
| 2026-07-24 | 三数据集 `representative_retrieval.mode = legacy`；Book FINAL integrated 无 lazy cap；本文档首版；Book nested γ 对齐 0.05；审计脚本 `audit_interrec_260724_cross_dataset.py` |

---

*本文档为 2026-07-24 InterRec canonical 约束参照；后续改 yaml 或主表路径时请同步更新本节与 §1 表格。*
