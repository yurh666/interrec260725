# InterRec 最终版 Pipeline 代码对照手册（interrec260725 · integrated distill surface · seed42）

**复核日期：2026-07-25（interrec-yrh release）**  
**代码根目录：** `/home/yrh666/interrec-yrh`  
**Canonical Full 配置：**

- Book：`configs/runs/book_full_integrated_distill_surface_seed42.yaml`
- MovieLens：`configs/runs/movielens_full_integrated_distill_surface_seed42.yaml`
- LastFM：`configs/runs/lastfm_full_integrated_distill_surface_seed42.yaml`

**Unified CLI：** `python3 scripts/run_final_interrec.py --dataset {book|movielens|lastfm}` → `configs/final_unified/runs/{dataset}_seed42.yaml`（继承上述 distill full）。

**方法论文档：** `METHOD_CN.md`（`BAYESIAN_MEMORY_TRANSIENT_SLOTS_METHOD_CN`）

本文按**真实执行顺序**逐文件说明：每个阶段在哪个文件、调用什么函数、读哪些配置键。可配合 IDE 跳转逐行核对。

**Surface Writer 在 turn 内的位置：** 用户模拟选择与 belief update **之后**写 log 文案（见 `session_simulator.py` 中 `apply_choice_belief_update` 之后 `SurfaceWriter.write_question`），不改变 choice vector。

---

## 0. 快速索引

| 阶段 | 主文件 | 入口函数 |
|------|--------|----------|
| CLI 启动 | `scripts/run_main_experiment.py` | `main()` L101 |
| 配置合并 | `src/utils/config.py` | `load_config()` L20 |
| 数据集 | `src/data/dataset.py` | `InterRecDataset.load()` |
| 嵌入索引 | `src/embedding/index.py` | `EmbeddingIndex.load()` |
| 实验循环 | `src/simulation/experiment_runner.py` | `run_interrec_experiment()` L100 |
| Session 循环 | `src/simulation/session_simulator.py` | `run_session()` L211 |
| 推荐 | `src/recommendation/recommender.py` | `BeliefRecommender.recommend()` |
| 假设/选择题 | `src/agents/frontier_clustered_hypothesis_integration.py` | `run_frontier_clustered_v32_hypotheses_turn()` |
| 本地 v2 选项 | `src/ask/integrated_representative_backend.py` | `generate_cluster_bound_options()` L120 |
| VOI 选题 | `src/ask/question_selector.py` | `select_best_question()` L13 |
| Ask 决策 | `src/ask/ask_policy.py` | `should_ask()` L6 |
| 信念更新 | `src/belief/laplace_update.py` | `laplace_choice_update()` |
| 日志 | `src/logging/run_logger.py` | `RunLogger.log_turn()` |

---

## 1. 如何启动一次完整运行

```bash
cd /home/yrh666/interrec-yrh

python3 scripts/run_main_experiment.py \
  --config configs/runs/book_full_integrated_distill_surface_seed42.yaml \
  --run-id-suffix interrec260725 \
  --seed 42
```

或：

```bash
python3 scripts/run_final_interrec.py --dataset book
```

输出目录：`experiments/runs/book_full_integrated_distill_surface_seed42/`（或带 suffix 的 run_id）  
关键产物：`config.yaml`（resolved 配置）、`full_log.jsonl`、`run_summary.json`、`metrics.csv`

---

## 2. 配置继承链（interrec260725 distill full）

`load_config()`（`src/utils/config.py`）递归读 `inherits:` 并 `deep_update` 合并。**子层覆盖父层**。

`configs/final_unified/runs/{dataset}_seed42.yaml` **仅**继承 `configs/runs/{dataset}_full_integrated_distill_surface_seed42.yaml`，**不**加载 `configs/final_unified/shared/protocol_seed42_ARCHIVED_20260722.yaml`（归档设计稿，见 `configs/final_unified/shared/README_CN.md`）。

**Book 典型链：**

```
book_full_integrated_distill_surface_seed42.yaml   ← Canonical（Intent Teacher v2 surface LoRA）
  └─ book_full_integrated_mock_surface_seed42.yaml
       integrated_representative；Book：when_to_ask=full_voi，entropy supplemental gate 关
  └─ book_hit025_surface_distill_only_seed42.yaml
       γ=0.05、frontier_clustered_v32 继承
```

MovieLens / LastFM：`{ds}_full_integrated_distill_surface_seed42` → `{ds}_full_integrated_mock_surface_seed42` → `{ds}_hit025_surface_distill_only_seed42`（hypothesis 链与 260724 mock full 相同，仅 surface 改为 `local_small_model`）。

**Surface LoRA：** canonical 为 **Intent Teacher v2**（`outputs/intent_teacher_v2/lora_adapter`，SHA256 `7efee94c…`）。见 `docs/SURFACE_WRITER_LORA_PROVENANCE_CN.md`。

### 2.1 三数据集 resolved 关键参数（distill full）

| 配置键 | Book | MovieLens | LastFM |
|--------|------|-----------|--------|
| `simulation.max_turns` | 15 | 15 | 15 |
| `ask.c_ask` / `mc_samples` | 0.35 / 128 | 同左 | 同左 |
| `ask.include_none` | **true** | **true** | **true** |
| `...when_to_ask` | **full_voi** | （未设，非 `entropy_only` → 走 MC VOI） | 同 ML |
| `...enable_cluster_bound_v2_entropy_ask_gate` | **false** | **true** | **true** |
| `...use_mock_llm` | **true** | **true** | **true** |
| `ask.voi_gate.enabled` | **false** | **false** | **false** |
| `semantic_backend.implementation_mode` | integrated_representative_item | 同左 | 同左 |
| `semantic_backend.gamma` | 0.05 | 0.05 | 0.05 |
| `surface_writer.mode` | `local_small_model` + LoRA | 同左 | 同左 |
| `llm.mode` | mock | mock | mock |
| `dataset.processed_dir` | `data/processed/amazon_book` | `data/processed/movielens` | `data/processed/lastfm_bge_m3_pca128` |
| 嵌入 | PCA128 | PCA256 | BGE-M3 PCA128 |

验证：

```bash
python3 -c "
from src.utils.config import load_config
c = load_config('configs/runs/book_full_integrated_distill_surface_seed42.yaml')
fc = c['ask']['hypothesis_generation']['frontier_clustered_v32']
print('impl', c['semantic_backend']['implementation_mode'])
print('when_to_ask', fc.get('when_to_ask'))
print('use_mock_llm', fc.get('use_mock_llm'))
print('voi_gate', c['ask']['voi_gate'])
print('surface', c['surface_writer']['local_model_path'])
"
```

---

## 3. 顶层执行流（`run_main_experiment.py`）

### 3.1 `parse_args()` L60–98

CLI 参数：`--config`, `--method`, `--split`, `--max-users`, `--user-start/--user-end`, `--run-id`, `--resume`, `--seed`

### 3.2 `main()` L101–276 逐步

| 步骤 | 代码位置 | 做什么 |
|------|----------|--------|
| 1 | L103 | `cfg = load_config(args.config)` |
| 2 | L105–120 | CLI 覆盖 split / user slice / max_users / seed |
| 3 | L122–123 | `set_seed(seed)` → `src/utils/seed.py` |
| 4 | L144–145 | `RunLogger(run_id)` → 创建 `experiments/runs/<run_id>/` |
| 5 | L151–161 | 设置 `INTERREC_LATENCY_PROFILE_PATH`、`INTERREC_DIAGNOSTICS_JSONL` 环境变量 |
| 6 | L163–174 | 检查 `interactions.csv`, `items.csv`, `sessions.json`, `item_embeddings.npy` |
| 7 | L205–207 | `logger.write_config(cfg)` 写初始配置 |
| 8 | L213 | **`InterRecDataset.load(processed_dir)`** → `src/data/dataset.py` |
| 9 | L226–234 | 读 `embedding.precomputed_path`，Book 用 `item_embeddings_pca128.npy` |
| 10 | L234 | **`EmbeddingIndex.load(processed_dir, embedding_filename=...)`** |
| 11 | L237 | **`BM25Index.build(dataset.items)`** → 代表 item 检索 / BM25 |
| 12 | L244 | **`run_interrec_experiment(cfg, run_dir, dataset, index, logger, bm25)`** |
| 13 | L245–247 | 若 `logging.generate_report=true` → `generate_run_report()` |
| 14 | L268 | `save_config(cfg, run_dir / "config.yaml")` 写 resolved 配置 |

**数据路径（Book）：**  
`data/processed/amazon_book/`  
- `sessions.json`：test split sessions  
- `items.csv`  
- `item_embeddings_pca128.npy`：预计算 BGE→PCA128

---

## 4. 实验层（`experiment_runner.py`）

### 4.1 `run_interrec_experiment()` L100–206

```
configure_engineering_guards(cfg)          # src/utils/session_guard_state.py
[可选] install_network_guard()             # network_guard.enabled=true 时
components = build_components(cfg, dataset)
recommender = BeliefRecommender(index, lambda_explore)
sessions = _sessions_for_simulation(dataset, sim_cfg)
for session in tqdm(sessions):
    theta_info = build_theta_star(session, index, sim_cfg)   # theta_builder.py
    turns = run_session(session, theta_info, index, recommender, ...)
    logger.log_turn(...)  # 每个 turn 一行 full_log.jsonl
agg = _aggregate_metrics(episodes, top_k)
logger.write_metrics(agg, episodes)
```

### 4.2 `build_components()` L53–97

实例化四个运行时组件（**Book final 多数走 mock/本地路径，不调 API**）：

| 组件 | 类 | 文件 |
|------|-----|------|
| LLM | `LLMClient` | `src/agents/llm_client.py` |
| 方向翻译 | `DirectionTranslator` | `src/agents/direction_translator.py` |
| 假设生成 | `HypothesisGenerator` | `src/agents/hypothesis_generator.py` |
| 选项文案 | `OptionWriter` | `src/agents/option_writer.py` |

`item_lookup = dataset.item_lookup()` 把 item ID 映射为书名/标题，供本地 backend 生成可读选项。

### 4.3 `build_theta_star()` — `src/simulation/theta_builder.py` L11–76

- 模拟用户 ground truth：对 `future_test`（或 valid/train）物品 embedding 加权平均  
- `user_choice_mode: deterministic_argmax` → `UserSimulator` 用 θ* 选答案  
- 日志字段：`theta_star_items`, `theta_star_source`

### 4.4 `_aggregate_metrics()` L252–276

聚合 `HitRate@10`, `NDCG@10`, `MRR@10`, `avg_ask_count` 等到 `run_summary.json`

---

## 5. Session 层：单用户 15 轮循环（`session_simulator.py`）

### 5.1 Session 初始化（turn 0，L252–346）

```python
# 读配置
K_dir=5, M_hyp=5, n_opt=3, mc_samples=128, c_ask=0.35, max_turns=15

observed_history = session["observed_history"]
weights = dataset.weights_for_items(user_id, observed_history)

# 信念初始化
belief, _, memory_init_trace = initialize_belief_with_memory(...)
# → src/memory/session_memory_integration.py
# Book final: memory.mode=compatibility, effect_strength=0
# 实际等价 initialize_belief() weighted_average → BeliefState(μ, Σ=σ₀²I)

theta_star = theta_info["theta_star"]
simulator = UserSimulator(theta_star, choice_mode=deterministic_argmax, tau=0.2)

voi_gate.enabled = False   # Book final 不走 cheap gate
```

**`initialize_belief()`**（`src/belief/initialization.py` L10–32）：
- `init_mode=weighted_average`：历史 item embedding 加权平均 → L2 归一化得 μ₀  
- `Sigma = sigma0² * I`，默认 `sigma0=1.0`, `jitter=1e-6`

### 5.2 每轮 Turn 流水线（L348–2143，`for turn in range(1, max_turns+1)`）

下面按**实际调用顺序**列出（Book final 路径）：

---

#### 阶段 A：推荐（L360–382）

```python
recs = recommender.recommend(belief, top_k=10, exclude=seen_items)
```

**`BeliefRecommender.recommend()`** — `src/recommendation/recommender.py` L36–38  
**打分** — `src/recommendation/scorer.py` L8–11：

```
score(i) = μ·e_i + λ * sqrt(e_iᵀ Σ e_i)
```

- `λ = recommendation.lambda_explore`（默认 0.1）  
- 输出 Top-10 → `rec_log`，并算 `HitRate@5/10`, `NDCG`, `MRR`

---

#### 阶段 B：不确定性方向 / 锚点（L398–417）

```python
anchor_dirs = build_anchor_directions(belief, index, K=5, anchor_top_k=5, exclude=seen_items)
```

**文件：** `src/ask/anchor_selector.py` L63+  
- 对 belief 协方差做特征分解，取 top-K 不确定方向  
- 每个方向配 positive/negative anchor items

---

#### 阶段 C：方向语义化（L419–438）

```python
translated = translator.translate(anchor_dirs, session_id=ep_id, turn_id=turn)
```

**文件：** `src/agents/direction_translator.py`  
- Book final：`llm.mode=mock` → mock 模板生成 direction_name（不调 API）  
- 结果写入 `anchor_dirs_log[].semantic_description`

---

#### 阶段 D：Frontier 假设生成（L469–667 区域）

**触发条件：**  
`hypothesis_generation.mode == "frontier_clustered_v32"` 且  
`frontier_clustered_v32.{enabled|simulator_integration_enabled|integrated}` 为 true  
（见 `frontier_cluster_session_armed()` — `frontier_clustered_hypothesis_integration.py` L55–59）

**Book final：** `pre_gate_enabled=false` → **每轮必跑** v32 生成（无 lazy pre-gate 跳过）

```python
frontier_gen_block = run_frontier_clustered_v32_hypotheses_turn(
    belief, index, rec_ids, rec_scores, observed_history,
    translated, hyp_generator, cfg, fc32_cfg, ...
)
```

**主文件：** `src/agents/frontier_clustered_hypothesis_integration.py`  
**入口：** `run_frontier_clustered_v32_hypotheses_turn()` ≈ L230

##### D.1 构建 Frontier Pool 并聚类

| 子步骤 | 函数 | 文件 |
|--------|------|------|
| 候选池 | `build_frontier_pool()` | `src/ask/frontier_clustered_v32.py` |
| 聚类 (~5 clusters) | `cluster_frontier_pool()` | 同上 |
| 对比对 | `select_frontier_cluster_contrasts()` | 同上 |

配置：`candidate_scope=1500`, `frontier_top_k=1500`, `cluster_count=5`

##### D.2 Cluster-bound v2 本地生成（Book final 关键路径）

**路由判断** `_representative_backend_enabled()` L72–81：

```python
# 以下任一则为 False → 走 mock：
preserve_reference_route == True
implementation_mode == cluster_bound_v2_mock

# Book final（改后）：
implementation_mode == integrated_representative_item  → True
```

**当 `cluster_bound_generation_v2_enabled=true` 且 representative backend 启用：**

```python
local_out = representative_backend.generate_cluster_bound_options(state_context)
# IntegratedRepresentativeSemanticBackend — integrated_representative_backend.py L120
```

**`generate_cluster_bound_options()` 逻辑（L120–184）：**

1. `_pick_three_candidate_clusters(clusters, contrasts)` → 选 3 个 cluster  
2. 固定角色：`roles = ["preserve", "boundary", "explore"]`  
3. 每个 cluster：`_select_representatives()` → 绑定 local representative item IDs  
4. 生成 `option_text`（Boundary 含 preserve/shift 双侧语义）  
5. 输出 JSON schema → `validate_cluster_bound_v2_schema()` + `validate_cluster_bound_v2_self_check()`  
6. 成功 → `cluster_bound_v2_to_hypotheses()` 转为 hypothesis 列表 + choice_question bundle

**与旧 mock 路径的区别：**

| | mock (`cluster_bound_v2_mock`) | integrated（Book final 新） |
|--|-------------------------------|----------------------------|
| 入口 | `run_cluster_bound_generation_v2()` mock 分支 | `IntegratedRepresentativeSemanticBackend.generate_cluster_bound_options()` |
| 选项文本 | 模板/mock LLM | 本地 representative items + 规则拼装 |
| API | 0 | 0 |
| 与 ML/LF | 不同 | **相同 backend** |

##### D.3 假设向量化（L670+）

```python
hyp_vectorized = vectorize_hypotheses(hyps_raw, index, ...)
```

**文件：** `src/ask/hypothesis_vectorizer.py`  
- 每个 hypothesis 的 `option_vector` / representative items → 归一化向量  
- 供 VOI 的 IG 计算使用

---

#### 阶段 E：Full VOI 选题（L878–1180）

**Book final 分支：**

```python
when_to_ask = "full_voi"   # ≠ entropy_only
voi_gate.enabled = False
→ run_full_voi = True
→ 直接 select_best_question(...)
```

**`select_best_question()`** — `src/ask/question_selector.py` L13–110：

1. 预计算 `inv(Sigma)`, Cholesky, MC samples（`fast_inv_and_logdet`, `fast_sample`）  
2. 从 M 个 hypothesis 中选 **n=3** 的子集，最大化 IG  
3. M≤8：枚举 `combinations(M, n)`；M>8：贪心增选  
4. 每个子集调用 `expected_posterior_entropy()` — `src/ask/information_gain.py` L12–74  
5. 返回 `best_ids`, `ig`, `option_probs`, profiling

**IG 定义：**

```
IG(Q) = H(prior) - E_θ[ H(posterior | answer, θ) ]
```

含 NONE option 的 softmax choice model（`tau`, `none_bias`, `alpha_none`）

---

#### 阶段 F：Ask / No-Ask 决策（L1182–1230+）

```python
do_ask, voi = should_ask(ig, c_ask)   # ask_policy.py L6-9
# voi = ig - 0.35; do_ask = (voi > 0)
```

**Book final 不走：**
- `entropy_only_policy`（ML/LF legacy 单轮用过）  
- `enable_cluster_bound_v2_entropy_ask_gate` 补充 gate  
- `midask_guard`（默认 disabled）  
- `voi_gate` cheap 预筛（`enabled=false`）

若 `do_ask=false`：本轮不展示问题，belief 不变，仍记录 final recommendation metrics。

---

#### 阶段 G：问题呈现与用户模拟（L1646–1757）

**Book：** `use_frontier_choice_question_for_ask=true`  
→ 复用 frontier 生成的 MCQ 文本，**跳过 OptionWriter LLM**

```python
_frontier_choice_options_for_selected_hypotheses(...)
user_choice = simulator.choose(option_vectors, ...)  # user_simulator.py
```

**`UserSimulator.choose()`** — `src/simulation/user_simulator.py`  
- `deterministic_argmax`：选与 θ* 内积最大的选项（含 NONE 逻辑）

---

#### 阶段 H：信念更新（L1761–1791）

```python
belief = apply_choice_belief_update(belief, selected_hypotheses, user_choice, ...)
# → session_memory_integration.py → laplace_choice_update()
```

**`laplace_choice_update()`** — `src/belief/laplace_update.py` L56+  
- Multinomial-softmax likelihood + 高斯 prior → Laplace 近似后验  
- 更新 μ, Σ（即 Method 文档中的 Memory M_{t+1}）  
- NONE 选择：扩大 Σ，μ 不变

Book final：`memory.effect_strength=0` → 无 memory projection 混入

---

#### 阶段 I：最终推荐 + 日志（L1948–2113）

```python
final_recs = recommender.recommend(belief, top_k=10, exclude=seen_items)
# ranking_metrics → HitRate@10 等写入 turn log
turns_log.append({..., "recommendation": rec_log, "final_recommendation_corrected": ...})
```

**`RunLogger.log_turn()`** — `src/logging/run_logger.py`  
每 turn 一行 JSON 写入 `full_log.jsonl`

---

## 6. 三数据集 Final 配置差异对照（interrec260725）

| 项 | Book | MovieLens | LastFM |
|----|------|-----------|--------|
| Canonical config | `book_full_integrated_distill_surface_seed42.yaml` | `movielens_full_integrated_distill_surface_seed42.yaml` | `lastfm_full_integrated_distill_surface_seed42.yaml` |
| when_to_ask / supplemental | **full_voi**；entropy gate **关** | MC VOI + **entropy supplemental gate 开** | 同 ML |
| hypothesis LLM | `use_mock_llm=true`（三数据集） | 同左 | 同左 |
| voi_gate (HRV) | **off** | **off** | **off** |
| surface_writer | `local_small_model` + Intent Teacher v2 | 同左 | 同左 |
| embedding | PCA128 | PCA256 | BGE-M3 PCA128 |

---

## 7. 与 `METHOD_CN.md` 的对齐说明

- **Belief → recommend → frontier → 三角色 integrated backend → joint VOI → Laplace** 与 Method 主叙事一致。  
- **Preserve / Explore 公式**在 Method §8 已按 `integrated_representative_backend.py` 写成 \(\arg\max o\)、\(\arg\max \bar u\)（无 tie-break 项）。  
- **include_none=true** 为三数据集 canonical，非仅 LastFM。  
- **HRV gate**：Method §13.4.1 与 resolved 配置一致（260725 **关闭**）。  
- **Surface 文案**：蒸馏 LoRA 在 turn 末 post-hoc 写 log；加载失败时 `mock_template` 不影响 VOI 向量。Intent v2 Test-300 报告 **不** 证明当前 LoRA 指标。

---

## 8. 关键源码文件清单（按目录）

```
scripts/
  run_main_experiment.py          # CLI 入口
  run_dataset.py                  # subprocess 包装

src/utils/
  config.py                       # YAML 继承加载
  seed.py                         # 随机种子

src/data/
  dataset.py                      # InterRecDataset, sessions, weights

src/embedding/
  index.py                        # EmbeddingIndex
  bm25_index.py                   # BM25 检索

src/belief/
  belief_state.py                 # BeliefState, entropy, eigendecompose
  initialization.py               # initialize_belief
  laplace_update.py               # laplace_choice_update

src/recommendation/
  recommender.py                  # BeliefRecommender
  scorer.py                       # belief_scores
  metrics.py                      # HitRate, NDCG, MRR

src/ask/
  anchor_selector.py              # build_anchor_directions
  frontier_clustered_v32.py       # pool, cluster, mock fallback
  cluster_bound_generation_v2.py  # v2 schema, mock/API 路径
  integrated_representative_backend.py  # ★ 本地 v2 选项生成
  hypothesis_vectorizer.py        # 假设 → 向量
  question_selector.py            # select_best_question (Full VOI)
  information_gain.py             # expected_posterior_entropy
  ask_policy.py                   # should_ask
  choice_model.py                 # softmax 选项概率
  voi_gate.py                     # cheap gate（Book final 关闭）

src/agents/
  frontier_clustered_hypothesis_integration.py  # ★ v32 集成主入口
  direction_translator.py
  hypothesis_generator.py
  option_writer.py
  llm_client.py

src/memory/
  session_memory_integration.py   # initialize/apply with memory wrapper

src/simulation/
  experiment_runner.py            # run_interrec_experiment
  session_simulator.py            # ★ run_session 主循环
  user_simulator.py               # 模拟用户选题
  theta_builder.py                # build_theta_star

src/logging/
  run_logger.py                   # full_log.jsonl
  report_generator.py             # report.md

configs/runs/
  book_full_integrated_distill_surface_seed42.yaml   # ★ canonical（Book）
  movielens_full_integrated_distill_surface_seed42.yaml
  lastfm_full_integrated_distill_surface_seed42.yaml
  final_unified/runs/{book,movielens,lastfm}_seed42.yaml   # CLI 薄封装
```

---

## 9. full_log.jsonl 字段对照（抽查用）

每个 turn 的典型顶层键：

| 字段 | 含义 |
|------|------|
| `turn` | 1..15 |
| `recommendation` | 本轮初始 Top-10 + HitRate@10 |
| `belief_before` / `belief_after` | μ, Σ, entropy |
| `anchor_directions` | 不确定方向 + 翻译 |
| `frontier_clustered_v32_generation` | pool/cluster/假设生成全块 |
| `...cluster_bound_generation_v2.backend_type` | 应为 `integrated_representative_item` |
| `hypotheses_vectorized` | VOI 输入向量 |
| `question_selection` | IG, best_ids, profiling |
| `ask_decision` | do_ask, voi, ig |
| `user_choice` | 模拟用户选择 |
| `final_recommendation_corrected` | 信念更新后最终推荐 + HitRate@10 |

**改 integrated 后验收：** 在 `full_log.jsonl` 中 grep `"backend_type": "integrated_representative_item"` 应覆盖所有 turn；不应再出现 `frontier_clustered_v32_cluster_bound_v2_mock`。

---

## 10. 跑通 Checklist（distill full）

- [ ] `python3 scripts/run_main_experiment.py --config configs/runs/book_full_integrated_distill_surface_seed42.yaml --seed 42`
- [ ] 确认 `experiments/runs/<run_id>/config.yaml` 中 `implementation_mode=frontier_clustered_v32_integrated_representative_item`
- [ ] 确认 `ask.voi_gate.enabled=false`、`use_mock_llm=true`（三数据集 hypothesis 路径）
- [ ] `full_log.jsonl` 中 `"backend_type": "integrated_representative_item"`
- [ ] Surface：若需证明 LoRA 生效，检查 turn log 中 surface writer 非纯 mock（或对照 `docs/SURFACE_WRITER_LORA_PROVENANCE_CN.md`）

---

## 11. 加速变体（非 canonical full run）

效率子实验用的 strict equivalence 加速：

| 变体 | 路径 |
|------|------|
| session_exact_accel_v2 | `interrec_variants/session_exact_accel_v2/session_runtime.py` |
| 固定 20 session benchmark | `interrec_variants/session_exact_accel/fixed_benchmark.py` |

加速层**包装** `run_session()`，不改变算法语义；Book 效率数字来自 V2 包装 + 同一 canonical config。

---

*文档版本：2026-07-22 · 对应 Book integrated cluster_bound v2 配置变更*
