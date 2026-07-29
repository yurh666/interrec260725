# InterRec Unified Final — 统一 Pipeline 代码对照手册（三数据集）

**版本：** interrec260725 distill full（`final_unified` CLI 薄封装）  
**复核日期：** 2026-07-25  
**代码根目录：** `/home/yrh666/interrec-yrh`  
**Method 对照文档：** `METHOD_CN.md`

| 项 | 当前 release | 说明 |
|----|--------------|------|
| 统一入口 | `scripts/run_final_interrec.py` | `--dataset {book\|movielens\|lastfm}` |
| 实际 resolved 配置 | `configs/runs/{dataset}_full_integrated_distill_surface_seed42.yaml` | Unified yaml **只继承 distill full** |
| `shared/protocol_seed42_ARCHIVED_20260722.yaml` | **未接入执行链** | 见 `configs/final_unified/shared/README_CN.md` |
| 结果目录 | `experiments/runs/<run_id>/` | 非 `experiments/final_unified_seed42/` 硬编码 |

**勿**再写「Unified 与 legacy resolved 相同且 symlink 字节相同」——260725 canonical 是 **integrated + distill surface** 单链，HRV gate **关闭**，hypothesis 路径 **`use_mock_llm=true`**（与 archived protocol 中 `use_mock_llm: false` 不同）。

---

## 0. 一句话

三数据集共用 **同一条 Python pipeline**；差别 **只在 YAML**（数据、嵌入、surface、ask 策略）。  
正式用户模拟为 **向量 argmax**（`UserSimulator`），与 LLM 模拟用户子实验无关。

---

## 1. 怎么跑

```bash
cd /home/yrh666/interrec-yrh
python3 scripts/run_final_interrec.py --list-datasets
python3 scripts/run_final_interrec.py --dataset book
```

底层等价于：

```bash
python3 scripts/run_main_experiment.py --config configs/runs/book_full_integrated_distill_surface_seed42.yaml
# 或 configs/final_unified/runs/book_seed42.yaml（inherits 上述文件）
```

---

## 2. 核心配置速查（三数据集 + 共享）

### 2.1 三数据集 **有差别且重要** 的配置

| 配置键 | Book | MovieLens | LastFM | 影响什么 |
|--------|------|-----------|--------|----------|
| `dataset.processed_dir` / `data_root` | `.../amazon_book` | `.../movielens` | `.../lastfm_bge_m3_pca128` | 数据与 session 数 |
| `embedding.dim` + `precomputed_path` | 128 / `pca128.npy` | 256 / `pca256.npy` | 128 / BGE-M3 manifest | 向量维度、算力 |
| `...enable_cluster_bound_v2_entropy_ask_gate` | **false** | **true** | **true** | **可选 supplemental gate**：自然 VOI 拒绝后，高 entropy 时仍可补问 |
| `...cluster_bound_domain` | `amazon_book` | `movielens_movie` | `lastfm_music` | cluster_bound **schema 域**（结构化 hypothesis 校验），与 surface 文案无关 |
| `surface_writer` | `local_small_model` + Intent Teacher v2 LoRA，失败则 `mock_template` | 同左 | 同左 | **仅 post-hoc 文案**；`user_choice_mode=deterministic_argmax` 看 hypothesis 向量 |
| `recommendation.lambda_explore` | 0.1（继承） | 0.1 | 0.1 | explore 项权重 λ |
| `protocol.candidate_scope` 等 | 1500（Book 协议） | ML frontier 继承链 | LF 继承链 | frontier 池大小 |

YAML 位置：`configs/final_unified/profiles/{book,movielens,lastfm}_seed42.yaml`

### 2.2 interrec260725 **共享** 核心配置（distill full resolved）

| 配置键 | 值 | 代码消费位置 |
|--------|-----|--------------|
| `seed` | 42 | 全局随机性 |
| `simulation.max_turns` | 15 | `run_session` |
| `simulation.user_choice_mode` | `deterministic_argmax` | `UserSimulator` |
| `ask.c_ask` | 0.35 | `should_ask()` |
| `ask.mc_samples` | 128 | `select_best_question()` |
| `ask.n_options` | 3 | VOI 子集大小 |
| `ask.include_none` | **true**（三数据集） | 选项空间含 NONE |
| `ask.voi_gate.enabled` | **false** | 不做 HRV cheap 预筛；full MC VOI |
| `...use_mock_llm` | **true** | hypothesis / cluster_bound **不走**在线 LLM |
| `semantic_backend.implementation_mode` | `frontier_clustered_v32_integrated_representative_item` | `integrated_representative_backend.py` |
| `semantic_backend.gamma` | 0.05 | representative 检索 |
| `memory.mode` | `compatibility` | belief 等价 original Laplace |
| `llm.mode` | mock | 无在线 API |

**Archived（未继承）：** `configs/final_unified/shared/protocol_seed42_ARCHIVED_20260722.yaml` 曾声明 `voi_gate.enabled: true` 与 `use_mock_llm: false` — **不是** 当前跑法。

---

## 3. 已有 Final 结果（Unified = Legacy 同一批 log）

| 数据集 | Hit@10 | PR@10 | AR@10 | avg_ask |
|--------|-------:|------:|------:|--------:|
| Book | 0.2591 | 0.3218 | 5.544 | 6.12 |
| MovieLens | 0.2114 | 0.3088 | 2.179 | 2.61 |
| LastFM | 0.5661 | 0.7015 | 2.152 | 4.55 |

PR/AR 三数据集同一算法，详见 release-staging `PR_AR_METRICS_CN.md`。

---

## 4. Pipeline 总览（统一代码，无数据集 fork）

```
scripts/run_final_interrec.py
  └─ scripts/run_main_experiment.py :: main()
       ├─ load_config()
       ├─ InterRecDataset.load()
       ├─ EmbeddingIndex.load(precomputed_path)   # 唯一数据集相关：嵌入文件
       ├─ BM25Index.build()
       └─ run_interrec_experiment()
            └─ for session: run_session()       # 所有逻辑在此；分支读 cfg
```

---

## 5. Pipeline 逐步：核心代码 + 关键配置

以下每一节：**Method 在讲什么 → 代码在哪 → 读哪些 config**。

---

### 5.0 配置加载

**作用：** 递归合并 YAML（`inherits`），得到 resolved config；三数据集仅入口 yaml 不同。

**代码：**

```20:28:interrec-final-github/src/utils/config.py
def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent_name = cfg.pop("inherits", None)
    if parent_name:
        parent = load_config(path.parent / parent_name)
        cfg = deep_update(parent, cfg)
    return cfg
```

**入口：**

```101:103:interrec-final-github/scripts/run_main_experiment.py
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
```

**嵌入路径（数据集差异点）：**

```226:234:interrec-final-github/scripts/run_main_experiment.py
    _emb_cfg = cfg.get("embedding", {})
    _precomputed_path = _emb_cfg.get("precomputed_path", None)
    if _precomputed_path:
        ...
        _emb_filename = _os.path.basename(str(_precomputed_path))
    else:
        _emb_filename = "item_embeddings.npy"
    index = EmbeddingIndex.load(processed_dir, embedding_filename=_emb_filename)
```

---

### 5.1 实验循环（Session 级）

**作用：** 遍历 test sessions，每个 session 调 `run_session`，写 `full_log.jsonl`。

**代码：**

```138:148:interrec-final-github/src/simulation/experiment_runner.py
        for session in tqdm(sessions, desc="Running InterRec episodes"):
            ...
                theta_info = build_theta_star(session, index, split=split, dataset=dataset)
                episode = run_session(
                    session=session,
                    theta_info=theta_info,
```

**关键 config：** `simulation.split`, `simulation.max_users`, `simulation.episode_ids`

---

### 5.2 Ground Truth θ*（仅模拟环境，不进策略）

**作用：** 用 `future_test` 物品 embedding 加权平均得到 θ*；**不参与**推荐、VOI、选项生成（Method §1.3 数据边界）。

**代码：**

```11:38:interrec-final-github/src/simulation/theta_builder.py
def build_theta_star(
    session: dict[str, Any],
    index: EmbeddingIndex,
    split: str = "test",
    mode: str = "future_average",
    dataset: Any = None,
) -> dict[str, Any]:
    """Construct the ground-truth preference vector theta* for a user simulation.
    ...
    split:
        One of 'train', 'valid', 'test'. Determines which future split is used to
        build theta*. This is recorded in the run log as theta_star_source.
```

**关键 config：** `simulation.theta_star_split`, `simulation.theta_star_mode`

---

### 5.3 Memory 初始化 M₀ = (μ₀, Σ₀)

**作用：** 可见历史加权平均 → μ₀；Σ₀ = σ₀²I。Final 三数据集 `memory.mode=compatibility` 且 `effect_strength=0`，数值上等价 `initialize_belief()`。

**代码：**

```10:32:interrec-final-github/src/belief/initialization.py
def initialize_belief(
    observed_history: list[str],
    index: EmbeddingIndex,
    weights: list[float] | None = None,
    init_mode: str = "weighted_average",
    sigma0: float = 1.0,
    jitter: float = 1e-6,
) -> BeliefState:
    ...
            mu = weighted_average(vectors, usable_weights)
            mu = l2_normalize(mu.reshape(1, -1))[0]
    ...
    Sigma = (sigma0 ** 2) * np.eye(dim, dtype=float)
    return BeliefState(mu=mu, Sigma=Sigma, jitter=jitter)
```

**Session 内调用：** `initialize_belief_with_memory()` → `session_simulator.py` L260 附近

**关键 config：** `belief.init_mode`, `belief.sigma0`, `memory.mode`, `memory.effect_strength`

---

### 5.4 推荐：Exploitation + Exploration

**作用：** Method §4：score(i) = μ·eᵢ + λ√(eᵢᵀΣeᵢ)；取 Top-K 构成当前推荐与 frontier 输入。

**代码：**

```8:11:interrec-final-github/src/recommendation/scorer.py
def belief_scores(belief: BeliefState, item_embeddings: np.ndarray, lambda_explore: float = 0.1) -> np.ndarray:
    mean_score = item_embeddings @ belief.mu
    uncertainty = np.einsum("ij,jk,ik->i", item_embeddings, belief.Sigma, item_embeddings)
    return mean_score + lambda_explore * np.sqrt(np.maximum(uncertainty, 0.0))
```

```36:38:interrec-final-github/src/recommendation/recommender.py
    def recommend(self, belief: BeliefState, top_k: int = 10, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        ranked = self.rank_all(belief, exclude=exclude)
        return ranked[:top_k]
```

**关键 config：** `recommendation.top_k`, `recommendation.lambda_explore`

---

### 5.5 Frontier Pool → 聚类 → Transient Slots

**作用：** 从 Top 排序截断 frontier（candidate_scope / frontier_top_k），聚成 ~5 个 cluster；每个 cluster 是 Method 里的 **Transient Frontier Intent Slot**（仅当轮有效）。

**代码入口：** `run_frontier_clustered_v32_hypotheses_turn()` — `src/agents/frontier_clustered_hypothesis_integration.py` L230+

内部调用链：
- `build_frontier_pool()` → `cluster_frontier_pool()` — `src/ask/frontier_clustered_v32.py`
- `select_frontier_cluster_contrasts()` — 构造 cluster 对比对

**关键 config：** `ask.hypothesis_generation.frontier_clustered_v32.candidate_scope`, `frontier_top_k`, `cluster_count`, `boundary_rank_end`

---

### 5.6 Preserve / Boundary / Explore 三选项（cluster_bound v2 本地）

**作用：** 从 cluster 中选 3 个，固定角色 preserve/boundary/explore，绑定 local representative items，生成 **option_vector**（Method §6–§7）。

**路由（config 决定 mock vs integrated）：**

```72:81:interrec-final-github/src/agents/frontier_clustered_hypothesis_integration.py
def _representative_backend_enabled(cfg: dict[str, Any], fc32_cfg: dict[str, Any]) -> bool:
    sb = _semantic_backend_cfg(cfg, fc32_cfg)
    if bool(sb.get("preserve_reference_route", False)):
        return False
    impl = str(sb.get("implementation_mode") or "").strip()
    if impl == "frontier_clustered_v32_cluster_bound_v2_mock":
        return False
    ...
    return bool(sb.get("enabled")) and str(sb.get("type") or "") == "integrated_representative_item"
```

**核心生成（Final 三数据集均走此路径）：**

```120:127:interrec-final-github/src/ask/integrated_representative_backend.py
    def generate_cluster_bound_options(self, state_context: dict[str, Any]) -> dict[str, Any]:
        ...
        picks = self._pick_three_candidate_clusters(clusters, contrasts)
        letters = ["A", "B", "C"]
        roles = ["preserve", "boundary", "explore"]
```

**关键 config：** `semantic_backend.implementation_mode`, `...cluster_bound_domain`, `semantic_backend.alpha/beta/gamma`

**注意：** `option_text` 字符串仅影响日志/展示；**正式模拟读 `option_vector`**（见 §5.10）。

---

### 5.7 假设向量化

**作用：** 每个 hypothesis 的 representative items / option_vector 归一化为 VOI 输入向量。

**代码：** `vectorize_hypotheses()` — `src/ask/hypothesis_vectorizer.py`

**关键 config：** `ask.representative_top_k`, `ask.hypothesis_generation.frontier_clustered_v32.max_representatives_per_cluster`

---

### 5.8 Full VOI：联合 IG 选题 + Ask 决策

**作用：** Method §9–§11：对**整道三选一**（含 NONE 预演）选 hypothesis 子集最大化 IG；自然策略 VOI = IG − c_ask。

**自然 VOI 决策（三数据集 Final 相同）：** 先 `select_best_question()`，再 `should_ask(ig, c_ask)`。代码**仅当** `when_to_ask=entropy_only` 时跳过 MC VOI；Final 三数据集均未设该项（或设为 `full_voi`），**行为等价：一律走 MC VOI**，无需在配置表里单独对比 Book「写了 full_voi」与 ML/LF「未写」。

```904:908:interrec-final-github/src/simulation/session_simulator.py
        fc_cfg_policy = hyp_gen_cfg.get("frontier_clustered_v32") or {}
        entropy_only_policy_enabled = (
            str(fc_cfg_policy.get("when_to_ask") or "").strip().lower() == "entropy_only"
        )
        run_full_voi = not entropy_only_policy_enabled
```

**Ask 决策栈（同一 `run_session`，按顺序叠加 optional gate）：**

| 顺序 | Gate / 机制 | Final 三数据集 | 配置键 |
|------|-------------|----------------|--------|
| 1 | **自然 VOI**（核心） | 全开 | `should_ask(ig, c_ask)` |
| 2 | OOS skip | 可挡 ask | frontier OOS 审计 |
| 3 | **Entropy supplemental gate** | Book **关**；ML/LF **开** | `enable_cluster_bound_v2_entropy_ask_gate` |
| 4 | Session ask budget / lazy cap | Final 默认关 | `max_total_asks_per_session_*` |
| 5 | Cooldown / 连问路由修复 | pre_gate 关时不走 | `ask_cooldown_turns`, `consecutive_ask_rank_unchanged` |
| 6 | Cheap VOI pre-gate (HRV) | **关**（260725：`voi_gate.enabled=false`） | `ask.voi_gate.*` |
| 7 | Mid-ask force | **关** | `ask.midask_guard.enabled` |

**Entropy supplemental gate 核心逻辑**（与 cooldown、budget 同属 optional overlay）：

```1090:1107:interrec-final-github/src/ask/cluster_bound_generation_v2.py
def try_cb_v2_entropy_ask_gate(
    ...
) -> tuple[bool, dict[str, Any] | None]:
    """Supplemental ask when v2 succeeded but VOI gate declined and entropy is high."""
    ...
    if not entropy_ask_gate_enabled(fc32_cfg):
        return False, None
    if policy_natural_should_ask or oos_skip_applied:
        return False, None
```

仅在 **自然 VOI 已拒绝**（`policy_natural_should_ask=False`）且 belief entropy 超阈值时，把 `do_ask` 改回 `True`——与 Method §13.5「配置化 gate 可覆盖自然 VOI 决策」同族。

**IG / 选题核心：**

```13:22:interrec-final-github/src/ask/question_selector.py
def select_best_question(
    belief: BeliefState,
    hypotheses_vectorized: list[dict[str, Any]],
    n_options: int = 3,
    ...
    mc_samples: int = 64,
) -> dict[str, Any]:
    """Select the hypothesis subset that maximises expected IG."""
```

```12:23:interrec-final-github/src/ask/information_gain.py
def expected_posterior_entropy(
    belief: BeliefState,
    option_vectors: np.ndarray,
    ...
    mc_samples: int = 64,
    ...
) -> dict[str, Any]:
    """Compute expected posterior entropy (and IG) for a given question."""
```

**Ask 决策：**

```6:9:interrec-final-github/src/ask/ask_policy.py
def should_ask(ig: float, c_ask: float) -> tuple[bool, float]:
    """VOI = IG - c_ask. Ask iff VOI > 0."""
    voi = ig - c_ask
    return voi > 0, voi
```

**HRV pre-gate（260725 主表关闭）**

1. **VOI 不能随意跳过**：每轮 ask 由 **full MC VOI**（`select_best_question`, `mc_samples=128`）给出 IG，再经 `should_ask(ig, c_ask)` 决定。  
2. **当前 release** `ask.voi_gate.enabled=false` — 不跑 cheap HRV 预筛；`src/ask/voi_gate.py` 仅作可选工程路径。  
3. Archived `protocol_seed42_ARCHIVED_20260722.yaml` 中的 `preserve_ask_decision=true` **未接入** distill full 继承链。

**关键 config：** `ask.c_ask`, `ask.mc_samples`, `ask.n_options`, `enable_cluster_bound_v2_entropy_ask_gate`, `ask.voi_gate.enabled`（260725 为 false）

**Memory / Slot 叙事 wrapper（数值等效）：** `src/memory/method_views.py`

- `BayesianLatentIntentMemoryView(mean, covariance)` ← `BeliefState`
- `TransientIntentSlotView(slot_id, role, ...)` ← `FrontierCluster`（integrated backend 绑选项时使用）

**关键 config：** `ask.c_ask`, `ask.mc_samples`, `ask.n_options`, `enable_cluster_bound_v2_entropy_ask_gate`, `ask.voi_gate.enabled`, `ask.voi_gate.preserve_ask_decision`（`when_to_ask` 仅 `entropy_only` 时特殊；Final 未启用）

---

### 5.9 Surface 文案（与主方法向量路径分离）

**作用：** 把结构化选项渲染成用户可见问句；**不改变** hypothesis / option 向量，**不改变** VOI、Laplace 更新或 `UserSimulator` 的 argmax 输入（见 `SurfaceWriter` 类文档字符串）。

| 数据集 | legacy 主表 log | 说明 |
|--------|-----------------|------|
| ML | `hybrid_surface_cache_strict` + `require_cache_hit: true` | **工程加速**：离线把蒸馏文案写入 sqlite，线上只 lookup，避免每轮跑小模型 |
| Book | `local_small_model` → mock fallback | 未挂 LoRA 时回退模板 |
| LF | `mock_template` | 模板文案 |

**`require_cache_hit`：** 仅约束 **surface 文案来源**（miss 则报错或禁止在线生成），**不是**推荐算法开关；与 belief / frontier / VOI 无关。ML 用 strict cache 是为了 **MovieLens 全量跑通时的吞吐**，不是改指标的设计。

**统一重跑（蒸馏 surface）：** `configs/final_unified/runs/*_seed42_surface_distill.yaml` + tmux `scripts/run_final_surface_distill_unified_tmux.sh` → 三数据集均 `local_small_model` + `outputs/surface_writer_qwen/lora_adapter`。**预期 Hit/PR/AR 与 mock/cache 主表一致**（仅 log 里 question_text 变化）。若机器上有 `local_v2_model` 适配器，可改 `mode: local_v2_model` + `v2_adapter_path`（同样只改文案）。

**关键 config：** `surface_writer.mode`, `local_model_path`, `require_cache_hit`（仅 cache 模式）

---

### 5.10 用户模拟（正式版 = 向量，非 LLM）

**作用：** Method §16：模拟用户按 softmax 在选项向量上选答案；Final 用 `deterministic_argmax`。

**代码：**

```38:54:interrec-final-github/src/simulation/user_simulator.py
    def choose(
        self,
        option_vectors: np.ndarray,
        option_ids: list[str],
        include_none: bool = True,
    ) -> dict[str, Any]:
        ...
        for i, oid in enumerate(option_ids):
            sim = float(np.dot(self.theta_star, option_vectors[i]))
            sims[oid] = sim
            logits[oid] = sim / max(self.tau, 1e-8)
```

**关键 config：** `simulation.user_choice_mode`, `simulation.tau_user_choice`, `ask.tau_choice`, `ask.alpha_none`

**LLM 模拟用户** 仅在子实验 `05_api_vs_vector`；**不在此 pipeline 主路径**。

---

### 5.11 Bayesian Memory 写回 M_{t+1}

**作用：** Method §15：普通选项 → Laplace 更新 μ,Σ；NONE → 扩大 Σ。

**代码：**

```56:86:interrec-final-github/src/belief/laplace_update.py
def laplace_choice_update(
    belief: BeliefState,
    option_vectors: np.ndarray,
    selected_index: int | None,
    ...
) -> "BeliefState | float":
    ...
    if selected_index is None:
        Sigma_new = belief.Sigma * float(alpha_none)
        ...
        return new_belief
```

**Session 包装：** `apply_choice_belief_update()` — `src/memory/session_memory_integration.py`

**关键 config：** `memory.belief_update_mode` (= `original_laplace`), `ask.alpha_none`

---

### 5.12 日志与指标

**代码：** `RunLogger.log_turn()` — `src/logging/run_logger.py`  
**产物：** `full_log.jsonl`, `run_summary.json`, `metrics.csv`

---

## 6. 配置文件地图

```
configs/final_unified/
  runs/{book,movielens,lastfm}_seed42.yaml   # → inherits *_full_integrated_distill_surface_seed42.yaml
  shared/protocol_seed42_ARCHIVED_20260722.yaml   # 未接入；见 shared/README_CN.md

configs/runs/
  {book,movielens,lastfm}_full_integrated_distill_surface_seed42.yaml   # ★ interrec260725 canonical

outputs/intent_teacher_v2/lora_adapter/   # Intent Teacher v2（canonical surface Student）
```

## 7. 等效性验证（历史）

2026-07-22 的 `verify_final_unified_equivalence.py` 针对 **旧** legacy/unified 对照；**不**适用于 interrec260725 distill full。当前以 `load_config('configs/runs/*_full_integrated_distill_surface_seed42.yaml')` 与 `experiments/runs/<run_id>/config.yaml` 为准。

Config resolved 字段一致 + 结果 symlink + `full_log.jsonl` 等 checksum 一致 → **book / movielens / lastfm 全部 PASS**。

---

## 8. 与 Method 文档的「讲故事」差异（非数值版本差异）

对照 `BAYESIAN_MEMORY_TRANSIENT_SLOTS_METHOD_CN.md`。下面只谈 **论文叙事该怎么讲**，不谈具体跑分版本。

### 8.1 Method 主故事（Ideal Narrative）

Method 想讲一条干净闭环：

1. 用户有 Bayesian Latent Intent Memory \(M_t=(\mu_t,\Sigma_t)\)  
2. 每轮从 \(M_t\) 推荐 → 抽 frontier → 聚类得 **Transient Slots**  
3. 固定构造 **Preserve / Boundary / Explore** 三选项  
4. 对**整道题**算联合 VOI，自然策略 **IG − c_ask > 0** 才问  
5. 向量回答 Laplace 写回 \(M_{t+1}\)  
6. 重复直到预算结束  

这是 **单一、可推导** 的方法论故事。

### 8.2 Unified Final 实现故事（Implementation Narrative）

**与 Method 一致、可以当主线的部分：**

| Method 概念 | 实现 | 论文怎么说 |
|-------------|------|------------|
| \(M_t=(\mu_t,\Sigma_t)\) | `BeliefState` | 直接对应，无需改口 |
| Exploit–explore 推荐 | `belief_scores` + λ | 公式一致 |
| Transient slots | frontier cluster（~5 个） | 「当轮有效、非跨轮兴趣槽」 |
| 三角色选项 | `_pick_three_candidate_clusters` + roles | Preserve/Boundary/Explore |
| 联合 VOI | `select_best_question` max IG | 不是逐 slot 取 max VOI |
| 向量用户 + Laplace 写回 | `UserSimulator` + `laplace_choice_update` | 正式实验口径 |
| θ* 仅模拟用 | `build_theta_star` + future split | Method §1.3 边界一致 |

**需要在论文里主动说明的差异（讲故事层面）：**

| 主题 | Method 写法 | Unified Final 实际 | 建议叙事 |
|------|-------------|-------------------|----------|
| **Ask：核心 vs 可选 gate** | 自然策略 IG−c>0；§13.5 允许工程 gate 覆盖 | **三数据集均先算 MC VOI + `should_ask`**；Book 关 supplemental gate，ML/LF **开 entropy supplemental gate** | **可以且应该**写成：「自然 VOI 是主策略；entropy gate、cooldown、session budget、OOS skip 等是同一类的 **optional ask overlay**」——与 Method §13.5 完全一致，不是第二套算法 |
| **Bayesian Memory 强度** | 强调跨轮读写 memory | Final：`compatibility` + effect=0，数值=original Laplace | 「框架支持 memory-active 扩展；主表为 compatibility 基线」 |
| **Surface 文案层** | 结构化选项 + 可读 text | ML 蒸馏 cache；Book/LF mock | 「展示层与推理层分离；模拟读 **option_vector**」 |
| **Cheap VOI pre-gate (HRV)** | §13 流水线内 cheap 预筛 | Unified **开** + preserve；Legacy **关** | **不是跳过 VOI**；Final full VOI avg **≪0.5s**（Method §13.4.2），支持实时交互 |

#### 8.2.1 论文里 Ask 策略推荐写法（optional gate 框架）

**主策略（Main Policy，Method 正文）：**

> 每轮生成 hypothesis 后，对整道三选一问题计算联合 IG；若 VOI = IG − c_ask > 0 则提问，否则跳过（natural ask）。

**可选覆盖层（Optional Ask Overlays，Implementation / Appendix）：**

与代码中已有机制 **同一抽象**，不必为 Book / ML / LF 各讲一套：

| Overlay | 作用 | Final 配置 |
|---------|------|------------|
| Entropy supplemental gate | 自然 VOI 拒绝，但 belief entropy 高 → **仍问** | ML/LF **on**；Book **off** |
| Session ask budget | 限制总提问次数 | Final 默认 off |
| Ask cooldown / routing repair | 限制连续无效提问、路由修复 | pre_gate off 时不生效 |
| OOS skip | 选项不合格 → 不问 | 按 frontier 审计 |
| HRV cheap gate | VOI 流水线 profiling / 预筛 | Unified **on** + preserve（full VOI 仍执行）；Legacy off |
| Mid-ask force | 低置信区间强制问 | Final **off** |

**Book vs ML/LF 一句话（讲故事用）：**

> 三数据集共用同一 VOI 主策略；Book Final 关闭全部 supplemental overlay，得到 **pure natural VOI**；ML/LF 额外开启 **entropy supplemental gate**，在「仍不确定但 VOI 未过线」时允许少量补问——与 cooldown、budget 同类，而非替换 VOI 的另一套方法。

**不宜再说：**「Book 用 VOI、ML/LF 不用 VOI」——代码上 **三者都调用** `select_best_question` + `should_ask`。

### 8.3 不应再讲的故事（已过时或易误导）

- ~~Book 单轮 entropy-only、不用 VOI~~ → Final Book 是 **15 轮 + MC VOI**  
- ~~ML/LF 不用 VOI、只用 entropy~~ → **同样先算 VOI**；entropy 只是 **optional supplemental gate**  
- ~~三数据集 ask 是两套算法~~ → 同一 `run_session` + 同一 gate 栈，仅 **overlay 开关** 不同  
- ~~选项文案影响主实验 Hit~~ → 正式模拟 **只读向量**；LLM user 是子实验  
- ~~PR/AR 各数据集算法不同~~ → 三数据集 **同一 postprocess 口径**

### 8.4 一句话总结（写 Related / Implementation 用）

> **Method 给出统一的 Bayesian-Memory + Transient-Slot + Joint-VOI 方法论；Unified Final 用同一套 `src/` 实例化。Ask 方面：自然 VOI 是主策略；entropy supplemental gate 与 cooldown、budget 等是同一类的 optional overlay——Book 关、ML/LF 开 entropy overlay。论文主叙事锚定核心公式；overlay 写 Implementation，与 Method §13.5 一致。**

---

## 9. 与旧版文档关系

- 旧版（Book 细节多）：`docs/FINAL_INTERREC_PIPELINE_CODEBOOK_CN.md` — 未修改  
- **本文件**：Unified Final 三数据集 + 逐步核心代码 + Method 叙事对照  

---

## 10. 复现 PR/AR

```bash
cd /home/yrh666/interrec-final-github
python3 -c "
from pathlib import Path
from experiments.in_depth.scripts.postprocess_subexperiment_6 import _aggregate_run
for ds in ['book','movielens','lastfm']:
    _, a = _aggregate_run(Path('experiments/final_unified_seed42')/ds, ds)
    print(ds, 'PR@10', round(a['PR@10'],4), 'AR@10', round(a['AR@10'],4))
"
```

---

*Unified Final · 三数据集统一 pipeline · 2026-07-22*
