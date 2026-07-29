# InterRec：基于 Bayesian Latent Intent Memory 与 Transient Frontier Slots 的交互式推荐方法

**复核日期：2026-07-22。** 本文以 `src/` 当前实现为数值事实来源，以
`InterRec_Bayesian_Memory_Transient_Slots_Method.pdf` 为叙事依据。第 1–18 节可作为论文
Method 初稿主体；第 19–25 节为伪代码、复杂度、实现映射和声明边界，投稿时可移入附录。

## 摘要

本文给出 InterRec 的完整方法定义。系统将用户在第 \(t\) 轮的高斯偏好后验



\[
q_t(\theta)=\mathcal{N}(\mu_t,\Sigma_t)
\]



解释为可跨轮读写的 **Bayesian Latent Intent Memory**



\[
M_t=(\mu_t,\Sigma_t),
\]



其中 \(\mu_t\) 表示当前最可能的连续偏好方向，\(\Sigma_t\) 表示方向相关的不确定性。系统首先从可见历史初始化 \(M_0\)，随后在每一轮利用 \(M_t\) 对候选物品进行 exploitation–exploration 联合评分，从当前排序中提取决策敏感的 recommendation frontier，并对该 frontier 进行聚类。每个 frontier cluster 被解释为一个仅在当前轮有效的 **Transient Frontier Intent Slot**。系统进一步从这些 slots 中构造固定的 Preserve、Boundary 和 Explore 三类 hypothesis，分别表示保持当前推荐方向、测试可能改变 Top-\(K\) 的邻近分叉，以及探索当前高不确定方向。

三个 hypothesis 对应同一道三选一澄清问题的三个普通选项；当前正式 LastFM 配置还展示 NONE。系统针对整道问题联合计算期望信息增益，而不是分别计算三个 slot 的 VOI 后取最大值。自然策略在预期信息收益超过询问成本时提问；最终 ask/no-ask 还可能被配置化 gate、budget 或 guard 覆盖。普通回答通过原有的 Laplace posterior update 写回 \(M_t\)，形成 \(M_{t+1}\)；NONE 则保持偏好均值并扩大协方差，以表示当前 slots 未覆盖真实意图。该闭环持续执行，直至达到交互预算或停止条件，最终输出终止 Memory 下的 Top-\(K\) 推荐。

本文所述 Bayesian Memory 与 transient slot 是对项目主推理路径的功能性重述，不引入新的 memory network，也不将 frontier cluster 改造成跨轮持久的用户兴趣槽。

---

## 1. 问题定义

### 1.1 输入

设用户 \(u\) 的按时间排序可见历史为



\[
H^{\mathrm{obs}}_u=(i_1,i_2,\ldots,i_m),
\]



其中每个物品 \(i\) 具有一个 \(d\) 维语义表示



\[
e_i\in\mathbb{R}^d.
\]



所有候选物品组成目录 \(\mathcal I\)，其 embedding 矩阵为



\[
E=[e_1,\ldots,e_N]^\top\in\mathbb{R}^{N\times d}.
\]



系统的运行时输入包括：

1. 可见历史 \(H^{\mathrm{obs}}_u\)；
2. 历史物品权重 \(w_1,\ldots,w_m\)；
3. 候选物品目录 \(\mathcal I\)；
4. 物品 embedding \(E\)；
5. 最大交互轮数 \(T_{\max}\)；
6. 推荐、frontier、聚类、choice model、VOI 和停止策略的配置。

### 1.2 潜在偏好

系统假设用户当前存在一个不可直接观测的连续偏好方向



\[
\theta\in\mathbb{R}^d.
\]



系统不会将 \(\theta\) 表示为单个确定向量，而是在第 \(t\) 轮维护一个高斯近似后验：



\[
q_t(\theta)
=q(\theta\mid H^{\mathrm{obs}}_u,Q_{<t},a_{<t})
=\mathcal{N}(\mu_t,\Sigma_t).
\]



其中：

- \(\mu_t\in\mathbb{R}^d\) 是当前偏好中心；
- \(\Sigma_t\in\mathbb{R}^{d\times d}\) 是偏好方向的不确定性；
- \(Q_{<t}\) 是此前展示过的问题；
- \(a_{<t}\) 是此前获得的用户回答。

### 1.3 运行时与评估数据边界

离线实验可以使用 `future_train`、`future_valid` 或 `future_test` 构造模拟用户的真实偏好 \(\theta^\star\)，并据此产生模拟回答或计算推荐指标。但是这些未来数据不得参与：

- Memory 初始化；
- 推荐打分；
- frontier 构造；
- cluster/slot 生成；
- hypothesis 构造；
- VOI 或 ask/no-ask 判断。

因此，\(\theta^\star\) 是离线 user simulator 和 evaluator 的环境变量，不是推荐策略的运行时输入。

---

## 2. Bayesian Latent Intent Memory

### 2.1 定义

定义第 \(t\) 轮的 Bayesian Latent Intent Memory 为



\[
M_t=(\mu_t,\Sigma_t).
\]



它与代码中的 `BeliefState(mu, Sigma)` 是同一数值对象，不存在额外的 belief-to-memory 神经转换：



\[
\phi(q_t)=\phi(\mu_t,\Sigma_t)=M_t.
\]



**主表 Final 读入路径：** 三数据集（Book / MovieLens / LastFM）均通过 `memory.mode: compatibility` 正常初始化与跨轮读写 \(M_t\)。运行时会把 `compatibility` 规范化为 `memory_compatible`：初始化时 `memory_projection_alpha=0`、`memory_belief_blend=0`，回答后 belief 仍由 `original_laplace` 更新；IntentMemory 扩展仅作 trace，**不改变** \(M_t=(\mu_t,\Sigma_t)\) 的数值路径。代码叙事层提供只读 wrapper `BayesianLatentIntentMemoryView(mean, covariance)`，分别 alias `belief.mu` 与 `belief.Sigma`，保证与 `BeliefState` 完全等效。

这种命名强调该状态具备五个记忆属性：

1. **历史依赖性**：\(M_0\) 由可见历史产生；
2. **跨轮持久性**：\(M_{t+1}\) 由 \(M_t\) 递推得到；
3. **可读性**：排序、frontier、hypothesis 和 VOI 都读取 \(M_t\)；
4. **可写性**：用户回答会更新 \(\mu_t\) 和 \(\Sigma_t\)；
5. **概率性**：系统不仅记录当前偏好中心，也显式保留未确认方向。

### 2.2 Memory mean

\(\mu_t\) 可以理解为系统当前的“偏好指南针”。对于归一化物品向量 \(e_i\)，内积



\[
e_i^\top\mu_t
\]



越大，表示物品方向与当前已知偏好越一致。

### 2.3 Memory uncertainty

\(\Sigma_t\) 可以理解为一团有方向的“不确定性椭球”。物品方向 \(e_i\) 上的预测方差为



\[
v_t(i)=e_i^\top\Sigma_t e_i,
\]



对应的预测标准差为



\[
u_t(i)=\sqrt{\max(v_t(i),0)}.
\]



与单一置信度不同，\(\Sigma_t\) 能表达“系统对民谣方向较确定、但对电子方向仍不确定”这类方向相关状态。

---

## 3. Memory 初始化

### 3.1 历史加权聚合

对于非空可见历史，项目主路径取历史物品 embedding 的加权平均：



\[
\bar e_u=
\frac{\sum_{j=1}^{m}w_j e_{i_j}}
{\sum_{j=1}^{m}w_j},
\]



并进行 L2 归一化：



\[
\mu_0=\frac{\bar e_u}{\|\bar e_u\|_2}.
\]



若为 cold-start 或没有可用历史，则



\[
\mu_0=\mathbf 0.
\]



### 3.2 初始协方差

初始不确定性采用各向同性协方差：



\[
\Sigma_0=\sigma_0^2 I_d,
\]



其中 \(\sigma_0>0\) 控制初始不确定程度，\(I_d\) 为 \(d\) 维单位矩阵。

### 3.3 主表 Final 的 Memory 读入（compatibility，非 memory_active）

三数据集 Final 配置（`configs/final_unified/` 与 legacy `configs/runs/final_*_seed42.yaml`）统一采用：

- `memory.mode: compatibility`；
- `effect_strength: 0.0`；
- `memory_projection_enabled: false`；
- `memory_projection_alpha: 0.0`；
- `memory_belief_blend: 0.0`；
- `belief_update_mode: original_laplace`。

因此，\(M_0\) 由可见历史加权平均 + 各向同性 \(\Sigma_0\) 初始化（§3.1–§3.2），每轮由 Laplace update 写回 \(M_{t+1}\)。**这就是本文所称的正常 Memory 读入/读写路径**——不是子实验中的 `memory_active` 变体，也不是 `IntentMemoryUnit` 投影回 belief 的路径。

### 3.4 超参数对比实验（Subexperiment 6）与 Final 参数选择

子实验 6 在相同 embedding 与 Final 协议下对比三类配置（详见 `experiments/in_depth/subexperiment_6_hparam_sensitivity_20260713_000602/reports/SUBEXPERIMENT_6_RESULTS_CN.md`）：

| 变体 | 含义 |
|------|------|
| **Final** | 本文主表：compatibility Memory + \(c_{\mathrm{ask}}=0.35\) + 完整 MC VOI |
| **Memory Variant** | 激活 `memory_active` 整体路径（非仅调 \(\eta\) 的单因素实验） |
| **Ask-Threshold Variant** | 调整 ask 阈值族 |

**结论（三数据集一致）：Final 配置在 Hit@10、PR@10、AR@10 上均优于或持平 Memory Variant 与 Ask-Threshold Variant。** 例如 Book：Final Hit@10=0.2591、PR@10=0.3218；Memory Variant Hit@10=0.2046；Ask-Threshold Hit@10=0.2216。因此论文主表应锚定 **compatibility Memory + 当前 Final 超参**，而非 memory_active 或单独调阈值变体。

### 3.5 可选历史 Memory 扩展（不属于本文核心 Memory）

项目中还存在 `src/memory/intent_memory_state.py` 定义的 `IntentMemoryState` /
`IntentMemoryUnit`。它们可将 observed history 分解为多个历史意图单元，并在
`memory_active` 模式下选择性投影或混合回 belief。它们不是 \(M_t\)，也不是本文的 transient
frontier slots。

在 LastFM 配置快照
`experiments/in_depth/interrec_in_depth_20260712_main/config_snapshots/lastfm_final.yaml`
及对应 final-full 配置中，上述扩展参数与 §3.3 相同（projection/blend 均为 0）。该扩展可记录
memory trace，但不改变主表数值路径。本文后续的 \(M_t\) 始终专指
`BeliefState(mu, Sigma)`；若实验切换到 `memory_active` 或非零 projection/blend，则不再属于
本文所称的“原实现等价”主路径。

### 3.6 Book Final 配置实例（相对 legacy 旧版的更新）

**Canonical 入口：** `configs/runs/final_book_full_noapi_seed42.yaml`  
**Unified 入口：** `configs/final_unified/runs/book_seed42.yaml`

| 项 | Book Final（当前） | legacy 旧版（不再作为主表） |
|----|-------------------|---------------------------|
| `simulation.max_turns` | **15** | turn=1 单轮 ablation |
| `ask.c_ask` | **0.35** | 0（单轮实验） |
| `when_to_ask` | **`full_voi`**（每轮 MC VOI） | `entropy_only` 等 |
| `semantic_backend` | **`frontier_clustered_v32_integrated_representative_item`** | `cluster_bound_v2_mock` |
| `embedding` | **PCA128**（`item_embeddings_pca128.npy`） | 同族 |
| `enable_cluster_bound_v2_entropy_ask_gate` | **false**（pure natural VOI） | — |
| test sessions | **2107** | 同 |

Book 与 ML/LF 共用同一 Python pipeline；差异仅在 YAML（嵌入维度、surface writer、entropy supplemental gate 等）。**不应再叙述** Book 为单轮 entropy-only 或 mock cluster_bound 主路径。

---

## 4. Bayesian Memory Read：推荐评分

### 4.1 实际代码公式

给定当前 Memory \(M_t=(\mu_t,\Sigma_t)\)，项目实际代码对每个候选物品 \(i\) 计算：



\[
s_t(i)
=e_i^\top\mu_t
+\lambda_{\mathrm{exp}}
\sqrt{\max(e_i^\top\Sigma_t e_i,0)}.
\]



其中：

- \(e_i^\top\mu_t\) 是 exploitation 项；
- \(\sqrt{e_i^\top\Sigma_t e_i}\) 是 uncertainty bonus；
- \(\lambda_{\mathrm{exp}}\ge 0\) 控制探索强度。

### 4.2 直观解释

第一项回答：

> 这个物品与当前已经推断出的偏好有多一致？

第二项回答：

> 这个物品是否位于系统还没有弄清楚的方向上？

因此，高分物品可能来自两类：

1. 系统已经较确定用户会喜欢的物品；
2. 当前不确定、但值得探索的物品。

### 4.3 排序

过滤 observed history 和其他已见物品后，对剩余候选按 \(s_t(i)\) 降序排列：



\[
\pi_t=\operatorname{argsort}_{i\in\mathcal I\setminus\mathcal I_{\mathrm{seen}}}
[-s_t(i)].
\]



当前推荐结果为：



\[
R_t^K=\pi_t[1:K].
\]



### 4.4 PDF 与代码的公式差异

现有 PDF 将探索项写为



\[
\lambda_{\mathrm{exp}}e_i^\top\Sigma_t e_i,
\]



即使用方差；实际 `src/recommendation/scorer.py` 使用



\[
\lambda_{\mathrm{exp}}\sqrt{e_i^\top\Sigma_t e_i},
\]



即使用标准差。本文以运行代码为准。若论文要求严格声称“叙事重述不改变数值计算”，正式版本应将 PDF 评分公式修正为带平方根的代码公式，或者明确说明文中公式仅为概念化表示。

---

## 5. Decision-Sensitive Recommendation Frontier

### 5.1 设计动机

系统没有必要对整个候选目录构造澄清问题。大量低分、低不确定且远离 Top-\(K\) 边界的物品，即使被用户确认，也未必改变当前推荐。因此，系统先构造一个较小的决策敏感工作集：



\[
F_t=\operatorname{Frontier}(\pi_t,M_t).
\]



### 5.2 Frontier 的三个来源

项目实际 frontier pool 是以下三类候选的并集：



\[
F_t=
F_t^{\mathrm{top}}
\cup F_t^{\mathrm{boundary}}
\cup F_t^{\mathrm{uncertain}}.
\]



#### 5.2.1 高排名候选



\[
F_t^{\mathrm{top}}
=\{\pi_t(1),\ldots,\pi_t(K_F)\},
\]



其中 \(K_F\) 由 `frontier_top_k` 控制。

它们表示当前 Memory 最支持的推荐方向。

#### 5.2.2 排名边界候选



\[
F_t^{\mathrm{boundary}}
=\{\pi_t(b_0),\ldots,\pi_t(b_1)\},
\]



其中 \(b_0\) 和 \(b_1\) 分别对应 `boundary_rank_start` 与 `boundary_rank_end`。

这些物品接近当前决策边界，其排名在 Memory 轻微变化后可能显著改变。

#### 5.2.3 高不确定候选

系统在排序头部大小为



\[
W_U=\min\left(|\pi_t|,\max(200,4K_U)\right)
\]



的候选窗口内计算



\[
u_t(i)=\sqrt{\max(e_i^\top\Sigma_t e_i,0)},
\]



并选取不确定性最高的 \(K_U\) 个候选：



\[
F_t^{\mathrm{uncertain}}
=\operatorname{Top}_{K_U}\{u_t(i)\}.
\]



其中 \(K_U\) 由 `frontier_uncertain_k` 控制。

### 5.3 去重与字段

三个来源合并后按 item ID 去重。每个 frontier item 保留：

- `item_id`；
- 当前 rank；
- belief score \(s_t(i)\)；
- uncertainty \(u_t(i)\)。

因此，frontier 中的物品只能称为“潜在有决策价值的候选”，不能直接称为“已经值得询问的选项”。是否真正提问仍由后续 hypothesis 构造和 VOI 决定。

---

## 6. Frontier Clustering 与 Transient Intent Slots

### 6.1 标准化

设 frontier 中共有 \(P_t\) 个有效物品，其 embedding 矩阵为



\[
E_t^F\in\mathbb{R}^{P_t\times d}.
\]



对每个 embedding 做 L2 归一化：



\[
\tilde e_i=\frac{e_i}{\|e_i\|_2}.
\]



### 6.2 KMeans 聚类

系统对归一化后的 frontier embeddings 执行 KMeans：



\[
\{C_{t,1},\ldots,C_{t,R_t}\}
=\operatorname{KMeans}(E_t^F).
\]



有效 cluster 数根据配置、frontier 大小和 `min_cluster_size` 启发式共同确定。代码中的有效
\(K\) 精确为（令 \(n_{\min}=\max(3,\texttt{min\_cluster\_size})\)）：



\[
R_t=
\min\left(
K_{\mathrm{requested}},
\max\left(2,
\left\lfloor
\frac{P_t}{\max(3,\lfloor n_{\min}/3\rfloor)}
\right\rfloor
\right),
P_t
\right),
\]



并至少尝试构造两个 cluster；KMeans 使用 `n_init=5`、`max_iter=40` 和由会话 RNG 产生的
`random_state`。需要注意，`min_cluster_size` **不**会在 KMeans 后删除小 cluster，它只用于
frontier pool 的最小可聚类条件和有效 \(K\) 的启发式计算。若 sklearn 不可用或聚类条件不足，
代码退化为 single-cluster fallback；而三角色构造要求至少两个 cluster，因此该 fallback
通常不能形成正常的三角色问题。

### 6.3 Cluster 统计量

对每个 cluster \(C_{t,r}\)，系统记录：

1. 成员集合 \(C_{t,r}\)；
2. cluster size；
3. 平均排名
   


\[
\overline{\operatorname{rank}}_{t,r};
\]



4. 平均 belief score
   


\[
\bar s_{t,r};
\]



5. 平均 uncertainty
   


\[
\bar u_{t,r};
\]



6. 与当前 Top-10 的重合比例
   


\[
o_{t,r}
   =\frac{|C_{t,r}\cap R_t^{10}|}{|C_{t,r}|};
\]



7. 与 observed-history centroid 的平均相似度；
8. 代表物品与语义摘要；
9. 平均 embedding 方向。

cluster prototype 为：



\[
c_{t,r}
=\operatorname{Normalize}
\left(
\frac{1}{|C_{t,r}|}
\sum_{i\in C_{t,r}}\tilde e_i
\right).
\]



### 6.4 Cluster 到 transient slot 的映射

本文将当前轮的 cluster 解释为 Transient Frontier Intent Slot：



\[
S_{t,r}
=
\left(
C_{t,r},
c_{t,r},
\bar u_{t,r},
\overline{\operatorname{rank}}_{t,r},
o_{t,r},
\mathcal E_{t,r}
\right),
\]



其中 \(\mathcal E_{t,r}\) 表示代表物品和语义证据。

该映射不改变数值：



\[
\phi(C_{t,r})=S_{t,r}.
\]



代码叙事层提供只读 wrapper `TransientIntentSlotView`（`src/memory/method_views.py`）：`slot_id` alias `cluster_id`，`centroid` alias cluster prototype，`role` 为 preserve/boundary/explore。Integrated backend 绑定选项时使用该 view，底层仍操作 `FrontierCluster` 对象；schema 字段 `bound_cluster_id` 保留兼容，数值等于 `slot_id`。

### 6.5 为什么是 transient

slot 只在当前轮有效。用户回答后，\(\mu_t\) 和 \(\Sigma_t\) 改变，导致：

1. 候选分数改变；
2. 排名改变；
3. frontier 成员改变；
4. KMeans 输入改变；
5. cluster 数量、成员和 ID 都可能改变。

因此：

- `cluster_id` 只能解释为当前轮局部 `slot_id`；
- 不保证 \(S_{t,r}\) 与 \(S_{t+1,r}\) 是同一个兴趣；
- 不得声称 slot identity 跨轮稳定；
- 不得声称 cluster 被写入长期用户 Memory。

---

## 7. Cluster Contrasts

### 7.1 作用

普通聚类只说明“哪些候选彼此相似”。为了形成有区分意义的问题，系统还需构造 cluster 之间的对比关系：



\[
\mathcal C_t
=\{(r_{\mathrm{ref}},r_{\mathrm{alt}})\}.
\]



每个 contrast 包含：

- reference cluster；
- contrasting cluster；
- 对比理由。

### 7.2 初始安全方向

系统优先寻找当前排名和历史较支持的 cluster。代码以 Top-10 overlap、history similarity 和 mean rank 的组合启发式选择 reference cluster，而不是直接计算
\(\mu_t^\top c_{t,r}\)。

### 7.3 高不确定竞争方向

contrast 构造先按 mean uncertainty 降序选择竞争方向；同 uncertainty 下，代码的排序键为
\(-|\overline{\operatorname{rank}}-25|\)，在升序排序中实际上优先选择离 rank 25 **更远**
的 cluster。因而这里不能把实现精确描述成“离典型边界最近”或显式最大化 Top-\(K\) rank
impact；更稳妥的解释是：代码用 uncertainty 与 rank-position 启发式构造竞争方向，Boundary
再把该 contrast 语言化为可能引起推荐转向的邻近分叉。

### 7.4 额外 contrasts

当 cluster 数量允许时，系统还会构造其他 cluster pair，以便语义生成或 fallback 路径具有更多可用对比。

### 7.5 与最终三个选项的关系

cluster contrasts 是 hypothesis 构造的中间层。准确链路是：



\[
\text{frontier items}
\rightarrow
\text{clusters/slots}
\rightarrow
\text{cluster contrasts}
\rightarrow
\text{hypotheses}
\rightarrow
\text{choice vectors}.
\]



因此，最终一个选项可以绑定一个主要 cluster，同时携带另一个 contrast cluster 的语义信息；它不一定是“一个原始 cluster 原封不动变成一个选项”。

---

## 8. Preserve、Boundary 与 Explore Hypotheses

### 8.1 固定三角色结构

在启用 `cluster_bound_generation_v2` 的正常 v32 路径中，系统要求生成且仅生成三个普通选项：



\[
\mathcal H_t
=
\{
h_t^{\mathrm{pre}},
h_t^{\mathrm{bdry}},
h_t^{\mathrm{exp}}
\}.
\]



schema 硬约束要求：

- option 数量必须等于 3；
- role 集合必须严格等于
  


\[
\{\text{preserve},\text{boundary},\text{explore}\};
\]



- 每种 role 恰好出现一次；
- 每个 option 必须绑定 cluster ID 和代表物品。

因此，在该主路径中：

- Preserve：1 个；
- Boundary：1 个；
- Explore：1 个。

### 8.2 Preserve hypothesis

Preserve 表示：

> 若用户希望继续保持当前推荐已经支持的方向，应选择哪个选项？

代码选择：



\[
r_{\mathrm{pre}}
=\arg\max_r\, o_{t,r}.
\]



即 **Integrated representative 主路径**（`integrated_representative_backend._pick_three_candidate_clusters`）中，Preserve 候选按 **Top-10 overlap 降序** 取首个 cluster，**不使用** mean-rank 次级键。（`cluster_bound_generation_v2` 等其它 legacy 路径仍可有不同 tie-break；**interrec260725 Final integrated full 以 representative backend 为准**。）

这与概念公式



\[
\arg\max_r\mu_t^\top c_{t,r}
\]



含义接近，但并非同一实际计算。正式 Method 若声称实现等价，应报告代码启发式，而不能把概念点积写成精确实现。

### 8.3 Explore hypothesis

Explore 表示：

> 哪个当前覆盖较弱或尚不确定的方向最值得确认？

代码选择：



\[
r_{\mathrm{exp}}
=\arg\max_r\, \bar u_{t,r}.
\]



即 **Integrated representative 主路径**中 Explore 候选按 **mean uncertainty 降序** 取首个 cluster，**不使用** low-overlap 次级键。（其它 backend 路径可有不同 tie-break。）

### 8.4 Boundary hypothesis

Boundary 表示：

> 哪个竞争方向可能让当前推荐从“保持原方向”转向另一个方向？

系统首先遍历已经构造的 cluster contrasts，寻找可用的 contrasting cluster；若没有合适 contrast，则退化为按 Top-10 overlap 排序后的中间 cluster。

Boundary option 额外包含：

- `boundary_preserve_feature`；
- `boundary_shift_feature`；
- `boundary_relation`；
- 可选 contrast cluster ID。

因此 Boundary 的语义不是简单的“第二个兴趣”，而是“保留一部分当前偏好，同时朝竞争方向发生变化”。

### 8.5 角色属于 hypothesis，而不是 cluster 的永久属性

Preserve、Boundary、Explore 是当前问题中 hypothesis/option 的角色。一个 cluster 在不同轮、不同上下文或 fallback 路径中可能承担不同角色。角色不应写入跨轮 Memory，也不应被解释为 cluster 的永久类型。

### 8.6 三角色 cluster 绑定（integrated 主路径）

代码在 `_pick_three_candidate_clusters` 中依次尝试绑定 safe / boundary / uncertain，并通过 `if cl not in out` **跳过已选 cluster**；若可用 cluster ≥3，会再从剩余列表补齐至 3 个 **互异** cluster。因此：

- **正常成功路径**（frontier 上至少 3 个有效 cluster）下，三个选项倾向于绑定 **三个不同的** `bound_cluster_id`；
- 仅当有效 cluster 不足 3 或 contrast 退化时，才可能出现角色共用同一 cluster。

准确说法：

> 实现目标是三个语义角色不同的 hypotheses；**有足够 cluster 时会强制去重**，而非 Method 概念层“允许任意重复”。

---

## 9. Hypothesis Vectorization

### 9.1 数值向量

每个 hypothesis 最终必须映射为一个 \(d\) 维 choice vector：



\[
h_{t,j}\in\mathbb{R}^d.
\]



在 cluster-bound prebound representatives 路径中，设 hypothesis \(j\) 的有效代表物品集合为



\[
\mathcal R_{t,j}=\{i_{j,1},\ldots,i_{j,L_j}\},
\]



则：



\[
\bar h_{t,j}
=
\frac{1}{L_j}
\sum_{\ell=1}^{L_j}e_{i_{j,\ell}},
\]




\[
h_{t,j}
=
\frac{\bar h_{t,j}}{\|\bar h_{t,j}\|_2}.
\]



### 9.2 代表物品来源

根据 `representative_retrieval.mode`，向量可能来自：

- cluster-bound 预绑定代表物品；
- frontier-aware retrieval；
- cluster centroid 模式；
- BM25 或 keyword retrieval fallback；
- contrastive positive/negative retrieval。

在当前 integrated representative-item / frontier-aware 路径中，系统优先保持 hypothesis 与 frontier cluster 代表物品的绑定。

### 9.3 文本与数值分离

自然语言 renderer 或 surface writer 只将结构化 hypothesis 转成用户可读文本。用户看到的 option text 不会重新编码并替代 \(h_{t,j}\)。因此：

**离线模拟实现顺序（interrec260725，`session_simulator.py`）**：先在 **choice vector / option_id** 上由 simulator 完成确定性选择 → Laplace belief update → **再**调用蒸馏 Surface Writer 写入 `question_text` / 展示用 `option_text`。**蒸馏文案不参与模拟用户点击**，不改变已发生的 vector 选择；故 **推荐指标与 mock surface 在 deterministic_argmax 下对齐**，但 log 叙事是「先选后写文案」，不是严格在线「先展示问题再回答」。论文中应写明 **offline replay / post-hoc surface**，或在未来在线系统中把 surface 前移到 simulator 之前。

- 用户界面读取文本；
- choice model、VOI 和 posterior update 读取 \(h_{t,j}\)；
- 文本润色不应改变数值路径。

---

## 10. Choice Set

### 10.1 正常 v32 成功路径中的唯一三选项问题

正常 v32 cluster-bound 路径生成三个 hypotheses，配置通常设：



\[
n_{\mathrm{options}}=3.
\]



因此普通选择集合为：



\[
Q_t=
\{
h_t^{\mathrm{pre}},
h_t^{\mathrm{bdry}},
h_t^{\mathrm{exp}}
\}.
\]



**interrec260725 Final full** 三数据集 resolved 均为 `ask.include_none=true`，用户可见回答空间为：



\[
\mathcal A_t
=
\{
\mathrm{Preserve},
\mathrm{Boundary},
\mathrm{Explore},
\mathrm{NONE}
\}.
\]



这里的“唯一”是指：在 `cluster_bound_generation_v2` 成功并产出三个有效 hypotheses 时，
`select_best_question` 收到 \(M=3\) 且配置 \(n=3\)，因此只有一个候选子集；它不是指整个代码库
所有 fallback / legacy generation 模式都只能产生这一道问题。

### 10.2 不存在三角色之间的单独 VOI 竞争

在当前主路径中，三个角色共同构成同一道问题。系统不是分别计算：



\[
\operatorname{VOI}(\mathrm{Preserve}),\quad
\operatorname{VOI}(\mathrm{Boundary}),\quad
\operatorname{VOI}(\mathrm{Explore})
\]



再取最大值。

正确对象是整道问题：



\[
\operatorname{VOI}(Q_t)
=
\operatorname{VOI}
\left(
\{
h_t^{\mathrm{pre}},
h_t^{\mathrm{bdry}},
h_t^{\mathrm{exp}}
\}
\right).
\]



### 10.3 Generic question selector 中的 max

通用 `select_best_question` 支持从 \(M\) 个 hypotheses 中选择 \(n\) 个：



\[
Q_t^\star
=
\arg\max_{\substack{Q\subseteq\mathcal H_t\\|Q|=n}}
\operatorname{IG}(Q).
\]



但是在当前正常 v32 路径中：



\[
M=3,\qquad n=3,
\]



所以候选组合数量为：



\[
\binom{3}{3}=1.
\]



唯一组合就是 Preserve + Boundary + Explore。代码仍会经过通用 `best_ig` 逻辑，但不存在多组问题之间的实质比较。

只有在其他生成模式产生 \(M>3\) 个 hypotheses 时，组合最大化才真正发挥作用。

---

## 11. Choice Likelihood

### 11.1 VOI 内部的条件响应模型

给定潜在偏好 \(\theta\) 和三个普通 option vectors：



\[
H_t=
\begin{bmatrix}
(h_t^{\mathrm{pre}})^\top\\
(h_t^{\mathrm{bdry}})^\top\\
(h_t^{\mathrm{exp}})^\top
\end{bmatrix}
\in\mathbb{R}^{3\times d},
\]



普通选项 \(j\) 的 utility 为：



\[
z_j(\theta)=\frac{h_{t,j}^\top\theta}{\tau},
\]



其中 \(\tau>0\) 为 choice temperature。

通用 helper 在 `none_logit=None` 时退化为不包含 NONE 的普通 softmax：



\[
p(a_t=j\mid\theta,Q_t)
=
\frac{\exp(z_j(\theta))}
{\sum_{\ell=1}^{3}\exp(z_\ell(\theta))}.
\]



### 11.2 NONE logit

设当前 Memory mean 与三个 options 的最大相似度为：



\[
s_{\max}
=
\max_j h_{t,j}^\top\mu_t.
\]



项目构造 NONE logit：



\[
z_{\mathrm{none}}
=
b_{\mathrm{none}}
+
\frac{\delta_{\mathrm{none}}-s_{\max}}{\tau},
\]



其中：

- \(b_{\mathrm{none}}\) 为 `none_bias`；
- \(\delta_{\mathrm{none}}\) 为 `none_threshold`。

当所有普通选项与当前 Memory mean 的相似度都较低时，NONE logit 增大。

VOI 内部包含 NONE 时：



\[
p(a_t=j\mid\theta,Q_t)
=
\frac{\exp(z_j)}
{\sum_{\ell=1}^{3}\exp(z_\ell)+\exp(z_{\mathrm{none}})},
\]




\[
p(a_t=\mathrm{NONE}\mid\theta,Q_t)
=
\frac{\exp(z_{\mathrm{none}})}
{\sum_{\ell=1}^{3}\exp(z_\ell)+\exp(z_{\mathrm{none}})}.
\]



其中 \(z_{\mathrm{none}}\) 在一道给定问题中由 \((\mu_t,H_t)\) 计算，对 Monte Carlo 样本
\(\theta\) 保持常数。该公式描述 `src/ask/choice_model.py` 与
`src/ask/information_gain.py` 中用于问题选择的响应模型。

### 11.3 离线 user simulator 与 VOI 模型的区别

离线 `UserSimulator` 对普通选项使用模拟真值 \(\theta^\star\) 和
`tau_user_choice`，对 NONE 则调用同一 `none_logit` 形式，但输入为 \(\theta^\star\) 且温度为
`tau_none`：



\[
z_{\mathrm{none}}^{\mathrm{sim}}
=
b_{\mathrm{none}}
+
\frac{
\delta_{\mathrm{none}}
-
\max_j h_{t,j}^{\top}\theta^\star
}{
\tau_{\mathrm{none}}
}.
\]



正式实验的 `deterministic_argmax` 选择最大 logit；`stochastic_sample` 才按 softmax 抽样。因此
VOI 的内部预测概率与离线模拟回答机制共享几何形式，但不是同一个随机过程，也不应把
\(\theta^\star\) 写成在线策略可见变量。

### 11.4 边际回答概率

由于 \(\theta\) 本身服从当前 Memory：



\[
\theta\sim\mathcal{N}(\mu_t,\Sigma_t),
\]



系统需要计算：



\[
p(a_t=j\mid Q_t,M_t)
=
\mathbb{E}_{\theta\sim q_t}
\left[
p(a_t=j\mid\theta,Q_t)
\right].
\]



项目的 VOI 路径使用 Monte Carlo 近似。采样



\[
\theta^{(s)}\sim\mathcal{N}(\mu_t,\Sigma_t),
\qquad s=1,\ldots,S,
\]



并计算：



\[
\hat p_j
=
\frac{1}{S}
\sum_{s=1}^{S}
p(a_t=j\mid\theta^{(s)},Q_t).
\]



### 11.5 `include_none` 的当前实现边界

`include_none` 会控制 renderer / option writer 是否显示 NONE，但当前
`expected_posterior_entropy` 无此参数，始终加入 NONE 分支；`run_session` 调用离线
`UserSimulator.choose` 时也固定传入 `include_none=True`。因此，当前正式配置
`include_none=true` 时上述链路一致；但 `include_none=false` 不是一个已实现完整的端到端消融，
不能仅改该配置就声称策略、VOI 和模拟器均已移除 NONE。

---

## 12. Slot-Disambiguation Information Gain 与 VOI

### 12.1 当前 Memory 熵

\(d\) 维高斯 Memory 的微分熵为：



\[
H[M_t]
=
\frac{1}{2}
\log
\left(
(2\pi e)^d|\Sigma_t|
\right).
\]



等价地：



\[
H[M_t]
=
\frac{1}{2}
\left[
d\log(2\pi e)
+\log|\Sigma_t|
\right].
\]



项目通过 Cholesky 分解计算 log-determinant，并使用 jitter 保证数值稳定。

### 12.2 预演每种回答

对于每个普通答案 \(j\)，系统不立即真正更新会话，而是临时执行一次 Laplace update，并仅计算回答后的 posterior entropy：



\[
H_{t+1}^{(j)}
=
H[M_{t+1}\mid a_t=j,Q_t].
\]



对 NONE 分支：



\[
H_{t+1}^{(\mathrm{none})}
=
H[M_{t+1}\mid a_t=\mathrm{NONE},Q_t].
\]



### 12.3 期望后验熵

在当前 VOI 实现中，所有回答组成集合



\[
\mathcal A_t=
\{1,2,3,\mathrm{none}\},
\]



则：



\[
\mathbb{E}[H_{t+1}\mid Q_t]
=
\sum_{a\in\mathcal A_t}
\hat p(a\mid Q_t,M_t)
H[M_{t+1}\mid a,Q_t].
\]



### 12.4 信息增益

整道问题的信息增益为：



\[
\operatorname{IG}(Q_t)
=
H[M_t]
-
\mathbb{E}[H_{t+1}\mid Q_t].
\]



它衡量：

> 若现在让用户在 Preserve、Boundary、Explore 和 NONE 之间作答，平均能够减少多少 Bayesian Memory 不确定性？

### 12.5 VOI

项目核心 ask policy 为：



\[
\operatorname{VOI}(Q_t)
=
\operatorname{IG}(Q_t)-c_{\mathrm{ask}},
\]



其中 \(c_{\mathrm{ask}}\ge 0\) 表示用户打扰成本、交互成本或问题预算成本。

自然提问决策为：



\[
\operatorname{Ask}_t
=
\mathbf 1[
\operatorname{VOI}(Q_t)>0
].
\]



即：



\[
\operatorname{Ask}_t
=
\mathbf 1[
\operatorname{IG}(Q_t)>c_{\mathrm{ask}}
].
\]



### 12.6 为什么不是分别对三个 cluster 算 VOI

一个回答的价值取决于选项之间的相对几何关系。例如 Preserve 与 Boundary 极其相似时，即使二者单独看都合理，把它们同时放入问题也可能无法有效区分用户。VOI 必须对完整 choice set 联合计算，才能考虑：

- 三个 option vectors 之间的差异；
- 用户在它们之间的相对选择概率；
- 每一种回答导致的不同 posterior；
- NONE 的概率与熵后果。

因此 VOI 的对象是问题 \(Q_t\)，而不是孤立的 cluster。

---

## 13. 可选 Gate 与工程决策覆盖

### 13.1 两类不同位置的廉价 gate

代码中有两类容易混淆的“pre-gate”：

1. `frontier_clustered_v32_pregate.py`：在启用 lazy generation 时，于 frontier/hypothesis
   生成前根据预算、cooldown、停滞、推荐 margin 和 belief entropy 判断是否调用 v32 生成路径；
2. `voi_gate.py`：hypotheses 已向量化后，以 dominant-pairwise HRV 判断是否调用完整 Monte
   Carlo VOI。

下文公式专指第二类 HRV gate。它不是第一类生成前 gate。

### 13.2 Hypothesis utility

设 hypothesis matrix 为



\[
H_t\in\mathbb{R}^{3\times d}.
\]



基于当前均值计算普通选项概率：



\[
p_j
=
\operatorname{softmax}
\left(
\frac{H_t\mu_t}{\tau}
\right)_j.
\]



系统按概率从高到低选择累计质量达到阈值 \(\rho\) 的 dominant set \(D_t\)，并在集合内重新归一化：



\[
p_i^D=\frac{p_i}{\sum_{k\in D_t}p_k}.
\]



### 13.3 Pairwise posterior variance

对 dominant set 中的每一对 hypotheses \(i,j\)，计算 utility difference 的后验方差：



\[
u_{ij}
=
(h_i-h_j)^\top
\Sigma_t
(h_i-h_j).
\]



按 dominant probabilities 加权：



\[
\operatorname{HRV}(Q_t)
=
\frac{
\sum_{i<j,\ i,j\in D_t}
p_i^D p_j^D u_{ij}
}{
\sum_{i<j,\ i,j\in D_t}
p_i^D p_j^D
}.
\]



若 dominant set 退化为单一选项，则该 score 为 0，表示系统在当前均值下几乎不存在主要选项竞争。

### 13.4 Gate 的位置与 Final 口径

代码流水线为：



\[
\text{Hypotheses}
\rightarrow
\text{Pre-gate (HRV)}
\rightarrow
\text{Full IG/VOI}
\rightarrow
\text{Ask/Stop}.
\]



HRV pre-gate **不替代** \(\operatorname{IG}(Q_t)\)，也不分别对 Preserve、Boundary、Explore 计算独立 VOI。它在工程上可先算 cheap \(\operatorname{HRV}(Q_t)\) 并记录诊断；**interrec260725 主表**在 gate 关闭时每轮仍跑 full MC VOI（见 §13.4.1–§13.4.2）。

### 13.4.1 interrec260725 Final full：HRV gate 关闭，full MC VOI

本 release（`configs/runs/*_full_integrated_distill_surface_seed42.yaml`）中 **`ask.voi_gate.enabled: false`**。Ask 决策由 **full Monte Carlo VOI**（`select_best_question` / `should_ask`）驱动，**不**启用 Unified 文档早期版本的 dominant-pairwise HRV pre-gate 主表路径。

Book / ML / LF 在 integrated 配置下 **`use_mock_llm: true`**（hypothesis 走 frontier integrated representative，非 API LLM）。MovieLens / LastFM 的 `when_to_ask` 未显式设为 entropy-only 时，执行 **full VOI 提问族**（与 Book `when_to_ask: full_voi` 同族），并可叠加 supplemental entropy gate（见各数据集 yaml）。

历史上 `preserve_ask_decision: true` + HRV gate 的描述适用于 **旧 `final_unified` + legacy yaml** 对照实验，**不是 interrec260725 主表跑法**。

### 13.4.2 实时交互：VOI 阶段延迟（非「跳过 VOI」）

Final 主表的实时性论据 **不是**「跳过 Monte Carlo VOI」，而是：**在 PCA 压缩 embedding（Book/LastFM 128 维、MovieLens 256 维）与 `mc_samples=128` 的配置下，full MC VOI 本身足够快**，可在在线选择题交互中实时完成。

对 `experiments/final_results_seed42/` 的 `full_log.jsonl` 统计（字段 `question_selection.voi_gate.full_voi_latency_sec`，且 `full_voi_called=true`）：

| 数据集 | full VOI 调用轮次 | 平均延迟 | 最大延迟 | ≥0.5s 轮次 |
|--------|------------------:|---------:|---------:|-----------:|
| Book | 31,605 | **0.031s** | 0.121s | 0 |
| MovieLens | 70,965 | **0.032s** | 0.738s | 3（≈0.004%） |
| LastFM | 24,720 | **0.012s** | 0.306s | 0 |

三数据集 **平均 VOI 延迟均远低于 0.5s**，满足「每轮实时算出是否提问、选哪组 hypothesis」的交互预算。HRV pre-gate 在此语境下是 **VOI 流水线内的 cheap 预筛与 profiling 层**，而非用跳过 VOI 来换指标安全的权宜之计。

### 13.5 工程策略

不同配置还可能启用：

- entropy-only ask policy；
- mid-ask guard；
- force-delivery smoke；
- out-of-scope skip；
- session ask budget；
- rank-safe guard；
- near-miss rescue。

这些机制可能覆盖自然的 `VOI > 0` 决策。论文主 Method 应将
\(\operatorname{IG}(Q_t)-c_{\mathrm{ask}}>0\)
作为核心策略，并把这些机制报告为配置相关工程 guard 或消融项，避免将所有分支混写成一个不可复现公式。尤其是正式 LastFM final-full 配置启用了
`enable_cluster_bound_v2_entropy_ask_gate`：当自然 VOI 拒绝、v2 生成成功且 belief entropy
超过阈值时，它可以补充触发一次询问。因此“实际每次询问都满足 VOI \(>0\)”并不成立；准确说法是
“VOI 是自然策略，配置化 gate/guard 可覆盖最终决策”。

---

## 14. User-facing Question

### 14.1 结构化选项

每个普通选项至少包含：

- option ID；
- role；
- bound cluster ID；
- representative item IDs；
- representative item names；
- cluster phrase；
- option text；
- 与其他选项的区别。

Boundary 还需包含 preserve/shift 两侧语义。

### 14.2 NONE

当 `include_none=true` 时，系统额外展示：

> None of the above fits well.

NONE 是回答空间的一部分，但没有普通 hypothesis vector。它通过独立 NONE logit 参与回答概率，并通过协方差扩张参与 VOI 的预演。

### 14.3 选项顺序

系统保持用户可见 option 顺序与 selected hypotheses 顺序一致，以保证：



\[
\text{selected\_vector\_index}
\leftrightarrow
h_{t,j}
\]



不发生错位。

---

## 15. Bayesian Memory Write

### 15.1 普通回答的实现后验

若用户选择普通选项 \(j\)，`laplace_choice_update` 使用仅含三个普通 option vectors 的
multinomial-softmax likelihood：



\[
p(\theta\mid a_t=j,Q_t,M_t)
\propto
\tilde p(a_t=j\mid\theta,Q_t)
\mathcal{N}(\theta;\mu_t,\Sigma_t).
\]



其中



\[
\tilde p(a_t=j\mid\theta,Q_t)
=
\frac{\exp(h_{t,j}^{\top}\theta/\tau)}
{\sum_{\ell=1}^{3}\exp(h_{t,\ell}^{\top}\theta/\tau)}.
\]



这里的 \(\tilde p\) 与第 11 节 VOI 响应模型中的普通答案概率必须区分：VOI 的概率归一化分母
包含 NONE logit，而 Laplace 普通回答写回的分母不包含 NONE。即使 NONE logit 对 \(\theta\)
固定，加入分母仍会改变普通答案 log-likelihood 的梯度，因此两者并非数学等价。本文按代码
分别报告，不将其写成同一个 likelihood。

由于 softmax likelihood 与 Gaussian prior 不共轭，项目使用 Laplace approximation。

### 15.2 MAP 目标

新的 posterior mode 为：



\[
\mu_{t+1}
=
\arg\max_\theta
\left[
\log \tilde p(a_t=j\mid\theta,Q_t)
-
\frac{1}{2}
(\theta-\mu_t)^\top
\Sigma_t^{-1}
(\theta-\mu_t)
\right].
\]



### 15.3 梯度

设 option matrix 为 \(X=H_t\)，普通三选项 softmax probability 为



\[
p(\theta)
=
\operatorname{softmax}
\left(
\frac{X\theta}{\tau}
\right),
\]



one-hot answer vector为 \(y\)，则 log posterior 梯度为：



\[
g(\theta)
=
\frac{(y-p(\theta))^\top X}{\tau}
-
\Sigma_t^{-1}(\theta-\mu_t).
\]



### 15.4 负 Hessian

令：



\[
W(\theta)
=
\operatorname{diag}(p(\theta))
-
p(\theta)p(\theta)^\top,
\]



则负 Hessian 为：



\[
\mathcal H_-(\theta)
=
\Sigma_t^{-1}
+
\frac{1}{\tau^2}
X^\top W(\theta)X.
\]



### 15.5 Newton update

从



\[
\theta^{(0)}=\mu_t
\]



开始迭代：



\[
\theta^{(k+1)}
=
\theta^{(k)}
+
\mathcal H_-(\theta^{(k)})^{-1}
g(\theta^{(k)}).
\]



代码默认最多执行少量 Newton steps，并在 step norm 足够小时提前停止。

最终：



\[
\mu_{t+1}=\theta^{(\mathrm{MAP})}.
\]



### 15.6 Laplace 协方差

在 MAP 点重新计算负 Hessian：



\[
\Sigma_{t+1}
=
\left[
\Sigma_t^{-1}
+
\frac{1}{\tau^2}
X^\top
\left(
\operatorname{diag}(p)-pp^\top
\right)
X
+
\epsilon I
\right]^{-1},
\]



其中 \(\epsilon\) 是数值稳定 jitter。

### 15.7 Memory write 语义

上述已有 posterior update 被解释为：



\[
M_{t+1}
=
\operatorname{LaplaceWrite}
(M_t,Q_t,a_t).
\]



这不是新增 writer network，而是对原有 Bayesian update 的记忆写入语义。

### 15.8 VOI 预演与真实写回的一致范围

`expected_posterior_entropy` 对每个普通答案调用同一个
`laplace_choice_update(..., selected_index=j, entropy_only=True)`，因此 VOI 预演的后验熵与
真实普通回答写回使用同一三普通选项 Laplace Hessian；NONE 预演和真实写回也都使用
\(\Sigma_{t+1}=\alpha_{\mathrm{none}}\Sigma_t\)。不一致只在“回答分支权重”与“普通分支更新
likelihood”之间：前者来自含 NONE 的响应模型，后者来自不含 NONE 的三选项 softmax。

---

## 16. NONE Update

若用户选择 NONE，则代码不移动均值：



\[
\mu_{t+1}=\mu_t,
\]



并扩大协方差：



\[
\Sigma_{t+1}
=
\alpha_{\mathrm{none}}\Sigma_t,
\qquad
\alpha_{\mathrm{none}}>1.
\]



直观上：

- 当前 Preserve、Boundary、Explore 均未覆盖真实意图；
- 系统不应错误确认其中任何方向；
- 扩大 \(\Sigma_t\) 会提升后续探索范围；
- 下一轮 ranking、frontier 和 slots 都会随之变化。

NONE 并不是“没有更新”，而是一次明确的 uncertainty expansion。

---

## 17. 多轮闭环

完整闭环为：



\[
H_u^{\mathrm{obs}}
\rightarrow
M_t
\rightarrow
R_t^K
\rightarrow
F_t
\rightarrow
\{S_{t,r}\}
\rightarrow
\mathcal C_t
\rightarrow
Q_t
\rightarrow
\operatorname{VOI}(Q_t)
\rightarrow
a_t
\rightarrow
M_{t+1}.
\]



每轮执行：

1. 从当前 \(M_t\) 读取推荐分数；
2. 生成当前 Top-\(K\)；
3. 构造 decision-sensitive frontier；
4. 对 frontier 做 KMeans；
5. 将 clusters 解释为 transient slots；
6. 构造 cluster contrasts；
7. 生成 Preserve、Boundary、Explore 三个 hypotheses；
8. 将 hypotheses 向量化；
9. 可选执行向量化后的 HRV gate；
10. 在正常 v32 成功路径中，对唯一的三选项问题联合计算 IG 和 VOI；
11. 若不提问，保留当前 \(M_t\)；
12. 若提问，展示三个普通选项和可选 NONE；
13. 获取用户回答；
14. 普通回答执行 Laplace write，NONE 执行 uncertainty expansion；
15. 使用 \(M_{t+1}\) 重新排序并进入下一轮。

---

## 18. 停止与输出

### 18.1 基础停止

会话最多执行：



\[
t=1,\ldots,T_{\max}.
\]



如果某轮不提问，系统仍可记录当前推荐并进入后续轮；具体是否提前结束由 session 配置和工程策略决定。

### 18.2 可能的停止或抑制因素

- VOI 不大于 0；
- pre-gate 判定没有足够竞争不确定性；
- hypothesis 为空；
- frontier 太小且 fallback 失败；
- 达到最大提问次数；
- 达到最大会话轮数；
- 连续不提问策略；
- rank-safe / out-of-scope guard；
- 配置指定的 entropy-only / entropy supplemental gate（可能抑制，也可能补充触发询问）。

### 18.3 最终输出

会话输出至少包括：

- 最终 Bayesian Memory \((\mu_T,\Sigma_T)\)；
- 最终 Top-\(K\) 推荐；
- 每轮推荐与分数；
- 每轮 hypotheses 和问题；
- 用户选择；
- ask/no-ask 与 VOI/IG 诊断；
- belief entropy、covariance trace 等轨迹；
- 离线推荐指标。

---

## 附录：实现与报告边界

以下第 19–25 节服务于复现、审计与投稿取舍，不是论文 Method 正文必须全部保留的内容。

## 19. 算法伪代码

```text
Input:
    observed history H_obs
    history weights w
    item embeddings E
    candidate catalog I
    top-K size K
    max turns T_max

Initialize:
    mu_0 <- Normalize(weighted mean of observed item embeddings)
    Sigma_0 <- sigma_0^2 I
    M_0 <- (mu_0, Sigma_0)
    seen <- observed item IDs

For t = 0, ..., T_max - 1:
    # Bayesian Memory read
    for each unseen item i:
        uncertainty_i <- sqrt(max(e_i^T Sigma_t e_i, 0))
        score_i <- e_i^T mu_t + lambda_exp * uncertainty_i
    ranking <- sort unseen items by score descending
    TopK_t <- first K items

    # Decision-sensitive frontier
    F_top <- top frontier_top_k ranked items
    F_boundary <- items at configured boundary ranks
    F_uncertain <- highest-uncertainty items in ranking head
    F_t <- deduplicate(F_top union F_boundary union F_uncertain)

    # Transient slot induction
    normalize frontier embeddings
    clusters <- KMeans(frontier embeddings)
    compute cluster rank, overlap, uncertainty, history and representative fields
    transient slots <- semantic aliases of current-round clusters

    # Fixed three-role hypothesis construction
    preserve cluster <- max(top10 overlap, then best mean rank)
    explore cluster <- max(mean uncertainty, then low top10 overlap)
    boundary cluster <- first valid contrast alternative, otherwise median-overlap fallback
    hypotheses <- exactly one preserve, one boundary and one explore
    validate schema and semantic distinctness

    # Numerical option vectors
    for each hypothesis:
        h_j <- Normalize(mean embeddings of bound representative items)
    Q_t <- {h_preserve, h_boundary, h_explore}

    # Optional HRV gate after vectorization
    if configured HRV gate rejects Q_t:
        do_ask <- false
    else:
        # Joint VOI for the entire three-option question
        estimate P(Preserve), P(Boundary), P(Explore), P(NONE)
        preview posterior entropy for every possible answer
        IG(Q_t) <- prior entropy - expected posterior entropy
        VOI(Q_t) <- IG(Q_t) - c_ask
        do_ask <- (VOI(Q_t) > 0)

    apply configured guards/budgets to final do_ask

    if not do_ask:
        keep M_t unchanged
        continue or stop according to session policy

    render Q_t as user-facing MCQ
    answer a_t <- user choice among Preserve/Boundary/Explore/NONE

    if a_t is NONE:
        mu_{t+1} <- mu_t
        Sigma_{t+1} <- alpha_none * Sigma_t
    else:
        mu_{t+1} <- Laplace MAP under ordinary-options-only softmax likelihood
        Sigma_{t+1} <- inverse negative Hessian at MAP

Return final Top-K, final Memory and full interaction trajectory
```

---

## 20. 计算复杂度

设：

- 候选物品数为 \(N\)；
- embedding 维度为 \(d\)；
- frontier 大小为 \(P\)；
- cluster 数为 \(R\)；
- KMeans 迭代数为 \(I_K\)；
- option 数为 \(J=3\)；
- Monte Carlo 样本数为 \(S\)；
- Newton 迭代数为 \(I_L\)。

### 20.1 推荐评分

均值项复杂度约为：



\[
O(Nd).
\]



一般 full covariance 下，不确定性项
\(e_i^\top\Sigma e_i\)
的直接复杂度为：



\[
O(Nd^2).
\]



### 20.2 Frontier 聚类

KMeans 约为：



\[
O(I_KPRd).
\]



### 20.3 Choice probability

每个 Monte Carlo 样本计算三个 option logits：



\[
O(SJd).
\]



### 20.4 Laplace preview

每个普通回答分支均涉及 \(d\times d\) Hessian 的 Cholesky 或求解。若按稠密矩阵处理，主要成本近似为：



\[
O(JI_Ld^3).
\]



项目通过以下方式降低常数：

- 每道题共享 \(\Sigma^{-1}\)；
- 共享 \(\Sigma_t\) 的 Cholesky/逆矩阵和 Monte Carlo samples；
- VOI preview 只计算 posterior entropy，不总是显式构造完整 covariance；
- HRV pre-gate 提供 cheap 预筛与 profiling；**interrec260725 主表** `voi_gate.enabled=false`，每轮走 full MC VOI（§13.4.1）。历史上 `preserve_ask_decision=true` 仅适用于 archived `final_unified` protocol，不是当前 release 跑法。

---

## 21. 实现映射

| 方法对象 | 项目实现 |
|---|---|
| \(M_t=(\mu_t,\Sigma_t)\) | `src/belief/belief_state.py::BeliefState` |
| Memory 叙事 view | `src/memory/method_views.py::BayesianLatentIntentMemoryView` |
| Transient slot 叙事 view | `src/memory/method_views.py::TransientIntentSlotView` |
| Memory 初始化 | `src/memory/session_memory_integration.py::initialize_belief_with_memory`（主表 compatibility 数值退化到 `src/belief/initialization.py::initialize_belief`） |
| Memory read score | `src/recommendation/scorer.py::belief_scores` |
| 推荐排序 | `src/recommendation/recommender.py::BeliefRecommender` |
| Frontier 构造 | `src/ask/frontier_clustered_v32.py::build_frontier_pool` |
| Frontier KMeans | `src/ask/frontier_clustered_v32.py::cluster_frontier_pool` |
| Cluster contrasts | `src/ask/frontier_clustered_v32.py::select_frontier_cluster_contrasts` |
| 三角色 cluster 选择（Final integrated） | `src/ask/integrated_representative_backend.py::_pick_three_candidate_clusters` |
| cluster_bound v2 schema / mock-API 路径 | `src/ask/cluster_bound_generation_v2.py` |
| 三选项 schema | `src/ask/cluster_bound_generation_v2.py::validate_cluster_bound_v2_schema` |
| Hypothesis vector | `src/ask/hypothesis_vectorizer.py` |
| VOI 响应概率 / NONE logit | `src/ask/choice_model.py` |
| 离线模拟回答 | `src/simulation/user_simulator.py::UserSimulator` |
| Expected entropy / IG | `src/ask/information_gain.py` |
| 通用问题选择器 | `src/ask/question_selector.py` |
| VOI 决策 | `src/ask/ask_policy.py::should_ask` |
| 生成前 lazy pre-gate | `src/ask/frontier_clustered_v32_pregate.py` |
| 完整 VOI 前 HRV gate（可选） | `src/ask/voi_gate.py`（260725 主表关闭；见 §13.4.1） |
| Laplace write | `src/belief/laplace_update.py::laplace_choice_update` |
| 完整多轮编排 | `src/simulation/session_simulator.py::run_session` |

---

## 22. 与其他 Memory/Slot 实现的边界

项目中存在三套容易混淆的对象。

### 22.1 主路径 Bayesian Memory



\[
M_t=(\mu_t,\Sigma_t)
\]



对应 `BeliefState`，始终参与排序、VOI 和 posterior update。这是本文与 PDF 所称的 Bayesian Latent Intent Memory。

### 22.2 可选 IntentMemoryUnit

`src/memory/intent_memory_state.py::IntentMemoryUnit` 对 observed history 进行自适应分解，可选地投影或混合回 belief。它是实验扩展，不等于本文的 transient frontier slot；在主表 compatibility 配置中不改变 belief 数值路径。

### 22.4 Subexperiment 6 与 Memory 路径选择

子实验 6A 的 **Memory Variant** 激活 `memory_active` 整体路径，不是仅改变单个 \(\eta\) 的 OFAT 实验。对比结果表明：在三数据集上，**compatibility 读入 + Final 超参（\(c_{\mathrm{ask}}=0.35\) 等） consistently 优于 Memory Variant 与 Ask-Threshold Variant**。因此主表应使用 §3.3–§3.4 的 Final 配置，而非 memory_active 或单独调阈值变体作为 production 叙事。

---

### 22.3 学习型 latent slot 变体

`interrec_variants/latent_intent_memory_minrev` 包含：

- K \(\le 2\) 的持久 latent slots；
- encoder；
- memory reader；
- intent projection；
- recommendation contrastive loss；
- reader cross-entropy。

这是独立训练变体，不属于本文“belief 即 Memory、frontier cluster 即 transient slot”的叙事等价主路径。

---

## 23. 可报告的诊断指标

### 23.1 Slot / option diversity

对于三个 option vectors：



\[
D_{\mathrm{slot}}
=
\frac{2}{J(J-1)}
\sum_{j<\ell}
\left(
1-\cos(h_{t,j},h_{t,\ell})
\right).
\]



该指标检验三个选项是否具有足够区分度。

### 23.2 Memory movement

普通回答 \(a_t\) 后：



\[
A_{\mathrm{write}}
=
\cos
\left(
\mu_{t+1}-\mu_t,
h_{t,a_t}
\right).
\]



它检验 Memory mean 是否沿用户选择方向移动。

### 23.3 Uncertainty reduction



\[
\Delta U_t
=
\operatorname{tr}(\Sigma_t)
-
\operatorname{tr}(\Sigma_{t+1}),
\]



或：



\[
\Delta H_t
=
H[M_t]-H[M_{t+1}].
\]



普通回答通常预期减少不确定性；NONE 则可能使其增加。

### 23.4 Slot decision relevance

可以记录：

- bound cluster mean rank；
- Top-10 overlap；
- mean uncertainty；
- 回答前后 Top-\(K\) overlap；
- selected option 对 target-rank improvement 的关联。

### 23.5 跨轮可视化

可使用相邻轮 cluster prototype cosine 进行离线匹配，以展示 transient slots 的出现、迁移和消失。但该匹配不得反馈到运行算法，否则 slot 将不再是纯 transient。

---

## 24. 方法声明的准确边界

### 24.1 可以声明

- 系统将 evolving Gaussian posterior 作为 Bayesian latent intent memory；
- Memory 同时存储当前偏好中心和方向相关不确定性；
- 系统从 Memory-conditioned recommendation frontier 中动态诱导 transient intent slots；
- Preserve、Boundary、Explore 组成一组固定三角色澄清问题；
- VOI 针对整道问题联合计算；
- 普通回答通过 Laplace approximation 写回 Memory，NONE 通过协方差扩张写回；
- NONE 表示当前 slots 未覆盖真实意图，并扩大 Memory uncertainty；
- slot 每轮重建，不具有保证的跨轮 identity。

### 24.2 不能声明

- 不能声称 belief 经过新 neural network 后才转化为 Memory；
- 不能声称 frontier cluster 是长期保存的用户兴趣槽；
- 不能声称 Preserve、Boundary、Explore 分别计算 VOI 后取最大值；
- 不能声称当前 v32 正常路径在多组问题中做 max-VOI 选择，因为 \(\binom{3}{3}=1\)；
- 不能声称三个 roles 在底层函数中被强制绑定三个完全不同的 cluster；
- 不能把 PDF 无平方根评分公式当成代码的精确公式；
- 不能声称 VOI 的含 NONE 响应概率与 Laplace 写回 likelihood 完全相同；
- 不能声称 `include_none=false` 已端到端移除 NONE；
- 不能声称所有实际询问都满足自然策略 VOI \(>0\)，因为配置化 gate/guard 可覆盖决策；
- 不能声称 future/test 数据参与运行时决策；
- 不能将 `latent_intent_memory_minrev` 的训练损失写成主 InterRec 的训练目标；
- 不能声称自然语言 renderer 改变了 choice vector。

---

## 25. 一句话定义

> InterRec maintains an evolving Gaussian posterior as Bayesian latent intent memory, reads that memory to construct a decision-sensitive recommendation frontier, induces transient intent slots by clustering the frontier, organizes these slots into one Preserve–Boundary–Explore choice set, evaluates the joint value of asking that question, and writes an ordinary choice back through a Laplace-approximated update or NONE through covariance expansion.

中文表述：

> InterRec 将持续演化的高斯偏好后验维护为 Bayesian Latent Intent Memory，从该 Memory 条件化的推荐前沿中聚类出当前轮临时意图 slots，在正常 v32 成功路径中将其组织为 Preserve、Boundary、Explore 三角色的唯一澄清问题，联合评估该问题的询问价值，并以 Laplace 近似写回普通回答、以协方差扩张写回 NONE。

