# InterRec 意图表层生成模型蒸馏：方法与实验结果

> 导师审阅初稿  
> 基座模型：Qwen2.5-1.5B-Instruct  
> Teacher：DeepSeek API  
> 任务：将三个结构化意图假设转写为面向用户的中性偏好问题及 A/B/C/NONE 四个选项

## 摘要

本工作面向交互式推荐中的“意图确认”环节：推荐系统内部已经形成三个承担不同功能的意图假设，但这些内部表示不能直接展示给用户。我们使用能力较强的 API 模型生成高质量监督数据，再将该能力蒸馏到本地 Qwen2.5-1.5B-Instruct 模型，使其能够把结构化意图证据稳定转写为简洁、客观、易于选择的自然语言问题。

三个选项分别承担不同交互作用：A 延续用户当前偏好，B 在保留熟悉属性的同时向邻近方向移动，C 提供更独立的探索方向，NONE 则允许用户拒绝三个假设。模型不仅需要生成流畅文本，还必须保持三类角色与各自证据的一致性，避免泄漏具体物品名称、内部算法术语和带有诱导性的评价语言。

实验使用 Book、LastFM 和 MovieLens 三个数据域的真实 ask events，按用户/会话组构建无交叉的训练、验证和测试集合。经过 Teacher 生成、硬规则过滤以及独立提示与证据路径的模型语义审计后，得到 5,311 条训练样本和 594 条验证样本，用于 Qwen2.5-1.5B 的 QLoRA 监督微调。最终模型在固定 300 条测试集上实现：

- 严格 JSON 格式通过率：**100%（300/300）**；
- 模型 Judge 语义忠实通过率：**99.67%（299/300）**；
- DeepSeek Teacher 的语义忠实通过率为 **99.33%（298/300）**；
- Student 相对 Teacher 的配对差值为 **+0.33 个百分点**，95% 置信区间为 **[-0.68, +1.40] 个百分点**。

置信区间跨越 0，因此本文不宣称 Student 在总体语义忠实率上显著超过 Teacher，也不把“未显著”解释为已经完成严格等价证明。当前数据表明：**Student 与 Teacher 的语义忠实率未观察到显著差异，且真实差异与 Student 低 0.68 至高 1.40 个百分点的区间相容**。结合 99.67% 的绝对通过率，可认为蒸馏后的 1.5B 本地模型在本任务上表现出与 API Teacher 接近的语义忠实水平。同时，Student 在部分案例中使用了更具体的证据概括，在形式空间中的选项分离度也显著更高，说明其并非只复现 JSON 外壳，而是学习到了三种意图角色的组织方式和面向用户的表达方法。

## 1. 研究问题与任务定位

### 1.1 推荐系统为什么需要意图表层生成

传统推荐通常直接返回物品列表，用户只能通过点击、跳过或评分间接反馈偏好。当系统对用户意图存在多个合理解释时，直接推荐可能造成多轮低效试错。

InterRec 在推荐过程中维护多个候选意图假设。本工作增加一个表层生成模块，将内部意图转化为用户可以直接回答的选择题：

```text
问题：What kind of movie experience are you looking for?

A. Continue with adventurous stories featuring fantasy and sci-fi elements.
B. Keep the family-friendly feel while moving toward animated musical comedies.
C. Try something lighthearted and funny, focusing on comedy.
NONE. None of these options fits what I want.
```

用户的一次选择能够直接确认“继续当前方向、移动到邻近方向、尝试探索方向，或全部不接受”，从而为后续推荐提供比隐式点击更明确、更高效的反馈。

### 1.2 A/B/C 不是普通的三个类别

本任务的关键不是生成三个主题不同的选项，而是保持三个 slot 的功能关系：

- **A / preserve**：保留当前已知偏好，回答“是否继续这一方向”；
- **B / boundary**：保留一个熟悉锚点，同时移动到证据支持的邻近方向；
- **C / explore**：给出受 C 证据支持、与当前方向更有区分度的探索选择；
- **NONE**：固定退出项，避免强迫用户接受系统假设。

最终文本不向用户暴露 preserve、boundary、explore、slot、embedding 等内部术语，而是通过自然句式表达角色：

- A：`Continue with ...`
- B：`Keep ... while moving toward ...`
- C：`Try ...`

因此，这是一个同时包含**证据压缩、角色保持、自然语言生成和交互设计约束**的结构化生成任务。

## 2. 数据构建与无泄漏划分

### 2.1 数据来源

样本来自 Book、LastFM 和 MovieLens 三个数据域的真实 ask events。每个事件包含三个意图 slot 及其结构化证据：

- Book 主要使用物品描述文本；
- LastFM 主要使用音乐风格与标签；
- MovieLens 主要使用电影类别与标签。

模型输入只包含可用于意图概括的结构化属性，不把物品标题、作者、演员、歌手、ID 等直接作为生成目标。

### 2.2 数据划分

冻结数据规模如下：

| 数据域 | 训练输入 | 验证输入 | 测试输入 |
|---|---:|---:|---:|
| Book | 1,800 | 200 | 100 |
| LastFM | 1,800 | 200 | 100 |
| MovieLens | 1,800 | 200 | 100 |
| 合计 | 5,400 | 600 | 300 |

划分以用户/会话组为单位，而不是随机打散单条事件。进一步使用 canonical evidence 指纹和泄漏连通分量合并潜在重复组，最终审计结果为：

- train / validation 的 group、event、canonical evidence 重叠均为 0；
- train / test 的三类重叠均为 0；
- validation / test 的三类重叠均为 0。

固定测试集在训练和 checkpoint 选择阶段不参与优化；最终选模记录为 `validation_only=true`、`heldout_read=false`。

## 3. Teacher 监督数据生成

### 3.1 Teacher 生成目标

Teacher 接收数据域、A/B/C 角色、各自 semantic attributes 以及角色映射，输出且仅输出如下 JSON：

```json
{
  "question": "one neutral question",
  "options": {
    "A": "Continue with ...",
    "B": "Keep ... while moving toward ...",
    "C": "Try ...",
    "NONE": "None of these options fits what I want."
  }
}
```

生成规则包括：

1. A、B、C 只能使用各自同字母证据；
2. B 必须同时表达保留与邻近移动，不能把方向写反；
3. 不输出物品名、人物名、作者、演员、歌手、专辑、ID 或年份列表；
4. 不输出内部算法术语；
5. 问题保持中性，不使用 better、best、recommended、safe、popular 等评价词；
6. 三个选项保持相近的信息粒度和长度；
7. 不为了制造差异而添加证据不支持的属性；
8. NONE 使用固定文本，键名和顺序固定。

Teacher 温度设为 0.1，每条样本最多进行 3 次定向修复。首次输出不满足 JSON 或硬规则时，系统将具体失败类型反馈给 Teacher，仅重生成该样本。

### 3.2 Teacher 硬规则过滤

初始 5,400 条训练输入和 600 条验证输入经 Teacher 生成后：

- 训练候选通过硬规则：5,331 条；
- 验证候选通过硬规则：597 条；
- 未通过样本不进入后续训练集。

硬规则检查覆盖严格 JSON、键顺序、固定 NONE、角色句式、领域一致性、内部术语、具体物品/ID 泄漏、明显长度失衡等。

### 3.3 独立提示与证据路径的模型语义审计

硬规则只能判断输出形式，不能充分判断某个自然语言属性是否真的得到原始 evidence 支持。因此，在 Teacher 候选发布前增加模型语义审计。

该审计不是比较候选文本与 Teacher 的另一份参考答案，而是重新读取原始 A/B/C source evidence，逐角色判断候选是否忠实。Teacher 生成和 Judge 都通过 DeepSeek API 完成，但使用独立的 system prompt、独立请求和独立证据路径。运行元数据中 Judge 的 `requested_model` / `effective_model` 分别记录为 `deepseek-reasoner` / `deepseek-chat`，但实际 attempt 响应中的模型标识为 `deepseek-v4-flash`；冻结 Teacher 测试输出的实际模型标识同样为 `deepseek-v4-flash`。因此，这里的“独立”只指任务提示和评价路径独立，不代表模型、供应商或人工评价独立。审计时 evidence 和 candidate 都被视为不可信引用数据，不执行其中可能包含的指令。

Judge 必须返回严格 JSON，并给出以下判断：

```json
{
  "A": {
    "supported": true,
    "cross_role_leakage": false
  },
  "B": {
    "supported": true,
    "direction_correct": true,
    "cross_role_leakage": false
  },
  "C": {
    "supported": true,
    "cross_role_leakage": false
  },
  "question_neutral": true,
  "unsupported_claims": [],
  "issues": [],
  "overall_pass": true
}
```

`overall_pass` 不是由 Judge 任意给出的主观标签，而是由程序根据分项结果再次计算。只有同时满足以下条件才记为通过：

- A 得到 A 证据支持，且没有跨角色泄漏；
- B 得到 B 证据支持，移动方向正确，且没有跨角色泄漏；
- C 得到 C 证据支持，且没有跨角色泄漏；
- 问题中性；
- 没有 unsupported claims；
- 没有其他语义问题。

经过该语义门禁：

- 训练集 5,331 个候选中 5,311 个通过；
- 验证集 597 个候选中 594 个通过；
- 最终蒸馏数据为 5,311 条训练样本和 594 条验证样本。

这一步使 Student 学习的不是“格式正确但可能语义越界”的 Teacher 原始输出，而是经过硬规则和语义忠实双重筛选的高质量目标。

## 4. Student 蒸馏训练

### 4.1 基座与初始化

Student 使用 Qwen2.5-1.5B-Instruct。训练从固定哈希的原始基座权重重新初始化，没有加载旧 Surface Writer adapter，也没有从旧 checkpoint 继续训练。

采用 QLoRA 以降低显存开销：

- 4-bit NF4 量化；
- double quantization；
- bfloat16 计算；
- gradient checkpointing；
- LoRA rank：16；
- LoRA alpha：32；
- LoRA dropout：0.05；
- target modules：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。

LoRA 同时覆盖注意力和前馈层，使小规模可训练参数能够学习角色映射、证据压缩和表面表达。

### 4.2 输入与监督信号

训练时使用 Qwen chat template：

1. system message 描述任务契约和输出规则；
2. user message 提供数据域、A/B/C intent evidence、role mapping 和结构冲突标记；
3. assistant completion 为经过审计的 Teacher JSON。

训练采用 **assistant-only causal language modeling loss**：

- system prompt 和 user evidence token 的 label 设为 `-100`；
- 只对 assistant 生成的 JSON token 计算损失；
- padding token 同样不计入损失。

这样可以避免模型把容量用于复现输入证据，而是集中学习“如何根据证据生成合规选择题”。

最大序列长度为 2,048 token，禁止静默截断。实际 5,905 条训练和验证样本中：

- 中位长度：479 token；
- P90：1,443 token；
- P99：1,506.96 token；
- 最大长度：1,645 token；
- 截断样本：0。

因此，所有 Teacher 监督目标和输入证据都完整参与训练。

### 4.3 优化配置

主要训练参数为：

| 参数 | 配置 |
|---|---:|
| Epoch | 3 |
| 单卡训练 batch size | 2 |
| Gradient accumulation | 16 |
| 有效 batch size | 32 |
| 验证 batch size | 2 |
| Optimizer | paged AdamW 8-bit |
| Learning rate | 1e-4 与 2e-4 两组 |
| Scheduler | cosine |
| Warmup ratio | 0.03 |
| Max gradient norm | 1.0 |
| Random seed | 42 |
| Eval / save interval | 100 steps |

两组学习率均训练 3 个 epoch，并保存多个中间 checkpoint，避免仅凭最后一步模型进行判断。

### 4.4 Validation-only checkpoint 选择

checkpoint 选择只使用固定 validation，不读取 Test-300。候选模型首先接受 schema、硬规则、具体名称泄漏和语义忠实审计。原配置设定零 item/name 泄漏门禁；validation 审计中所有候选都存在少量或歧义性 LastFM 名称标记，因此实际选择规则保留观测泄漏数最少的候选，最低计数为 2，再在这些候选中计算综合指标：

```text
0.65 × semantic_overall
+ 0.25 × semantic_component
+ 0.06 × hard_pass
+ 0.04 × schema_pass
- 0.50 × schema_failure
- 0.30 × hard_failure
```

综合指标是门禁后的主要排序依据。只有综合指标相同时，才依次使用语义总体通过率、硬规则通过率、validation loss、较早 checkpoint 和学习率进行 tie-break。该处理和原零泄漏门禁的偏离均记录在最终 selection manifest 中。

最终选择：

- 学习率：2e-4；
- checkpoint：step 300；
- validation loss：0.3130；
- validation schema pass：100%（600/600）；
- validation hard pass：97.83%（587/600）；
- validation semantic pass：99.33%（596/600）；
- semantic fidelity composite：0.98737。

该结果表明 step 300 已经充分学习 Teacher 的任务结构；继续训练到最后一步并未被默认视为更优。

## 5. 测试评估方法

### 5.1 固定 Test-300

测试集包含 300 条完全冻结样本：

- Book：100；
- LastFM：100；
- MovieLens：100。

Student 与 DeepSeek Teacher 在完全相同的 sample_id 和输入 evidence 上分别生成结果。所有生成失败均保留在分母中。

### 5.2 主要指标：基于独立提示与证据路径的模型 Judge 语义忠实率

本文将模型 Judge 语义忠实率作为主要效果指标，而不把 BGE cosine 当作主要结论。原因是：

1. 本任务关注 A/B/C 的功能角色与证据支持关系，而不仅是句向量接近；
2. Judge 可以显式判断 B 是否完成“保留 + 移动”，普通 cosine 无法直接表达这一逻辑；
3. Judge 能检查跨角色泄漏、无证据属性和问题中立性；
4. LastFM 的正式 PCA 文本投影未通过严格重放门禁，无法形成三域统一的正式 BGE 结果；
5. Book raw-1024 与 MovieLens PCA256 的绝对 cosine 尺度不同，不适合简单平均后解释为“忠实度百分比”。

测试 Judge 与训练数据语义门禁使用相同的严格判定逻辑，但对 Student 和 Teacher 一视同仁。Judge 不接收“哪一个系统生成”作为质量偏好，也不把 Teacher 文本当作标准答案，只根据原始同角色 evidence 进行审计。

### 5.3 统计方法

通过率差异使用 10,000 次 paired bootstrap：

- 按 Book、LastFM、MovieLens 分层；
- 按 `group_key` 聚类重采样；
- Student 与 Teacher 始终在相同 sample_id 上配对；
- 固定随机种子 20260717；
- 报告观测差值与 95% 置信区间。

该方法避免把同一用户/会话内的多条记录错误地视为完全独立样本。

## 6. 主要实验结果

### 6.1 蒸馏模型能够稳定生成合规选择题

| 指标 | Qwen 1.5B Student | DeepSeek Teacher | Student - Teacher |
|---|---:|---:|---:|
| 严格 JSON 通过率 | 100.00% | 100.00% | 0.00 pp |
| 硬规则通过率 | 97.67% | 98.00% | -0.33 pp |
| 模型 Judge 语义忠实率 | **99.67%** | **99.33%** | **+0.33 pp** |

配对 bootstrap 结果：

- JSON 差值：0，95% CI `[0, 0]`；
- 硬规则差值：-0.33 pp，95% CI `[-2.31, +1.64]` pp；
- 语义忠实差值：+0.33 pp，95% CI `[-0.68, +1.40]` pp。

JSON、硬规则和语义忠实的 Student–Teacher 差异均没有统计显著性。由此可以得到两个核心结论：

1. 1.5B Student 已经稳定掌握严格输出协议和三种 slot 角色；
2. 在最关键的模型 Judge 语义忠实指标上，Student 的点估计与 DeepSeek API Teacher 接近，未观察到显著差异。

Student 的语义忠实率点估计略高于 Teacher，但置信区间跨越 0，而且本实验没有预先设定非劣界值，因此不能作严格的“非劣”或“等价”推断。本文将其表述为：在 300 条测试样本的当前统计精度下，Student 与 Teacher 未观察到显著差异，Student 的绝对通过率达到 99.67%。

### 6.2 三个数据域均保持高语义忠实

| 数据域 | Student 语义忠实率 | Teacher 语义忠实率 |
|---|---:|---:|
| Book | 100% | 99% |
| LastFM | 99% | 99% |
| MovieLens | 100% | 100% |

结果说明蒸馏能力不是依赖某一个数据域：

- 对 Book 的长描述 evidence，Student 能提取高层主题而不复制书名；
- 对 LastFM 的稀疏标签，Student 能把标签组织成自然音乐偏好；
- 对 MovieLens 的类别组合，Student 能生成简短、易扫描的电影选择题。

### 6.3 Student 不只是复现格式

如果模型只学会固定句式，而没有学会意图角色，那么 A/B/C 很容易出现内容交换、B 方向反转或无证据扩写。模型 Judge 的 299/300 通过结果意味着：

- A 基本都延续了 A evidence；
- B 基本都同时保持熟悉属性并向 B evidence 支持的方向移动；
- C 基本都忠实于 C evidence；
- 问题保持中性；
- 极少出现跨角色内容或无依据属性。

因此，训练学到的是从结构化 intent evidence 到用户可读交互文本的映射，而不只是 JSON 外壳。

### 6.4 与 API Teacher 的关系

本工作的目标不是逐字复制 Teacher，而是把 Teacher 的任务能力迁移到本地小模型。有效蒸馏应满足：

- 相同 evidence 下保持相同角色语义；
- 文本可以采用不同但自然的同义表达；
- 不因为追求 Teacher 措辞而引入具体物品名；
- 本地模型在严格规则下保持稳定。

测试结果表明 Student 与 API Teacher 的总体语义忠实率未观察到显著差异；在部分案例中，Student 会采用更具体或更简洁的证据表达。形式空间辅助分析还显示 Student 的 option separation 高于 Teacher，说明三个选项在 embedding 空间中通常具有更清楚的差异。

这里的“达到或局部超过 API 形式”应准确理解为：

- **总体上**：语义忠实率与 API Teacher 未观察到显著差异，但尚未进行预设界值的等价或非劣检验；
- **部分案例和部分指标上**：Student 可以给出更具体的概括，或产生更清楚的选项区分；
- **不主张**：Student 在所有语言能力上全面超过通用 API 大模型。

这一结果已经满足本模块的工程目标：使用体量较小、可本地运行的模型，完成原本依赖在线 API 的受约束意图表层生成任务。

## 7. 典型案例分析

### 7.1 MovieLens：简洁、自然且角色完整

Sample：`be013426d85350be1b69453b`

原始意图证据：

- A：Sci-Fi；
- B：Adventure / Children's；
- C：Action / Drama / War。

Student：

> **Question:** What kind of movie experience are you looking for?  
> **A.** Continue with Sci-Fi movies.  
> **B.** Keep the adventurous spirit while moving toward family-friendly stories.  
> **C.** Try Action-packed films with drama and war elements.  
> **NONE.** None of these options fits what I want.

Teacher：

> **A.** Continue with Sci-Fi movies.  
> **B.** Keep the adventurous spirit while moving toward family-friendly stories.  
> **C.** Try Action movies with dramatic and war elements.

分析：

- A 直接保持 Sci-Fi，没有加入无依据属性；
- B 将 Children's 自然改写为 `family-friendly stories`，同时保留 Adventure 的体验锚点；
- C 把 Action / Drama / War 压缩成自然电影描述；
- 三个选项短而清楚，Student 与 Teacher 几乎逐项同义；
- 两个系统均通过硬规则和模型 Judge 语义审计。

该案例体现了本任务期望的最终交互形态：问题中性，选项自然，三个方向一眼可分，同时不展示任何内部表示或具体电影名。

### 7.2 LastFM：用共同锚点构造自然迁移

Sample：`bcb8aed0142216398e661511`

原始意图证据：

- A：60s/70s classic rock / psychedelic / folk / singer-songwriter；
- B：hard rock / stoner rock / alternative rock / 90s；
- C：alt-country / indie folk / slowcore / acoustic singer-songwriter。

Student：

> **A.** Continue with classic rock and psychedelic sounds.  
> **B.** Keep the rock foundation while moving toward harder, alternative styles.  
> **C.** Try alt-country and indie folk with a slower, more acoustic feel.

Teacher：

> **A.** Continue with classic rock and psychedelic folk from the 60s and 70s.  
> **B.** Keep the rock and psychedelic vibe while moving toward harder rock and alternative from the 90s.  
> **C.** Try alt-country and indie folk with a singer-songwriter focus.

分析：

- A 保留 classic rock 和 psychedelic 主体；
- B 使用 A/B 共享的 `rock foundation`，再转向 harder / alternative；
- C 不只复述 genre，还忠实概括 slowcore / acoustic 的体验差异；
- Teacher 保留了更多年代信息，Student 更短、更适合界面快速扫描；
- 两个系统均通过硬规则和模型 Judge 语义审计。

该案例说明 Student 能够在不罗列长标签的情况下，保留共同锚点并表达邻近迁移。

### 7.3 Book：从混合长描述中提取主方向

Sample：`57613f358c7e5ec36fa7542b`

证据主体：

- A 以社会、科技和历史类非虚构为主，同时含少量小说和烹饪内容；
- B 侧重人物、历史与真实世界叙事；
- C 侧重职业、投资和个人成长实用指南。

Student：

> **A.** Continue with non-fiction works that offer insightful perspectives on society, technology, or culture.  
> **B.** Keep a focus on real-world issues while moving toward narrative-driven accounts of influential figures or historical events.  
> **C.** Try books that provide practical guidance on career development, investing, or personal growth strategies.

Teacher：

> **A.** Continue with thought-provoking nonfiction that challenges conventional wisdom.  
> **B.** Keep a focus on influential figures and ideas while moving toward personal narratives and memoirs.  
> **C.** Try practical guides that offer actionable advice for personal or professional development.

分析：

- Student 的 A 用 society / technology / culture 抓住混合书目中的主体信号；
- B 完整表达“保留真实世界议题，再转向人物或历史叙事”；
- C 的 career / investing / personal growth 与原始 evidence 高度对应；
- 三项形成社会科技主题、真实人物叙事和实用指南三个清楚方向；
- Student 比 Teacher 使用了更具体的语义维度，更便于用户判断。

该案例说明 1.5B Student 能处理远比干净类别标签更复杂的输入，将多条长描述压缩为适合交互界面的短选项。

### 7.4 MovieLens：通过共同体验构造 boundary

Sample：`5a79dd90bd46fb46879c5e57`

原始意图证据：

- A：Action / Adventure / Sci-Fi；
- B：Horror / Thriller；
- C：Drama / War / Romance。

Student：

> **A.** Continue with action-packed adventures and sci-fi elements.  
> **B.** Keep the excitement while moving toward horror and thriller themes.  
> **C.** Try dramatic stories with war and romance.

Teacher：

> **A.** Continue with action-packed, adventurous sci-fi films.  
> **B.** Keep the excitement while moving toward horror and thriller elements.  
> **C.** Try dramatic stories with war and romance themes.

分析：

- B 没有简单写成“Try horror”，而是使用 `excitement` 保留 A 的体验锚点；
- 在保留刺激感的基础上，B 再移动到 horror/thriller；
- C 则提供 drama/war/romance 的独立探索方向；
- Student 与 Teacher 几乎逐项同义。

该案例直接说明 Student 学到了 boundary slot 的核心逻辑，而不是仅学习三个固定开头。

### 7.5 LastFM：用短文本表达年代、地域和流派

Sample：`cb986bb89b1992f4c4c5a165`

Student：

> **A.** Continue with British indie rock and 90s alternative.  
> **B.** Keep British rock while moving toward classic rock and hard rock.  
> **C.** Try Japanese indie rock and metal influences.

Teacher：

> **A.** Continue with British indie and alternative rock from the 90s, with some electronic and synth pop influences.  
> **B.** Keep the British rock focus while moving toward classic and progressive rock from the 70s and 80s.  
> **C.** Try indie rock and alternative with a mix of metal and Japanese rock for a fun twist.

Student 用更短的表达保留了年代、地域和流派三个维度。Teacher 信息更丰富，Student 则更适合界面快速扫描；两者核心意图一致。

### 7.6 Book：长描述压缩后的三方向选择

Sample：`8d63909911cdec6763eb62c9`

Student：

> **A.** Continue with non-fiction accounts of historical events and cultural movements.  
> **B.** Keep a focus on narrative-driven stories while moving toward mystery and suspense elements.  
> **C.** Try heartwarming tales of small-town life and gentle fiction.

Teacher：

> **A.** Continue with non-fiction accounts of historical events and social movements.  
> **B.** Keep a focus on real-world settings while moving toward suspenseful crime and mystery stories.  
> **C.** Try heartwarming tales of small-town life and gentle fiction.

Student 与 Teacher 在 A、C 上几乎同义，B 也都捕捉到从叙事性向 mystery/suspense 的移动。复杂书籍简介被压缩为三个可直接选择的用户意图。

## 9. 讨论

### 9.1 蒸馏为什么有效

本次蒸馏的有效性来自四个环节：

1. **Teacher 目标被严格约束**：不是使用自由回答，而是要求角色、格式和文本风格同时满足任务契约；
2. **训练前进行双重筛选**：硬规则过滤形式问题，独立提示与证据路径的模型 Judge 过滤语义越界；
3. **assistant-only loss**：训练信号集中在目标选择题，不浪费在复现长输入上；
4. **validation-only 选模**：不是默认选择训练最后一步，而是比较多个 checkpoint 的真实生成表现。

本流程优先保证监督目标的规则和语义质量，而不是直接保留全部 Teacher 输出。经过筛选的 5,311 条样本使 1.5B 模型在 Test-300 上表现出稳定的三角色生成能力；数据门禁各环节的独立贡献仍需要后续消融实验确认。

### 9.2 对推荐交互的意义

蒸馏后的本地模型能够：

- 将内部意图假设转成用户可理解的描述；
- 保持问题中性，避免暗示某个答案更优；
- 一次展示继续、邻近移动和探索三个方向；
- 通过固定 NONE 保留用户拒绝权；
- 减少运行时对外部 API 的依赖；
- 为用户提供快速、显式、低成本的偏好反馈入口。

这使意图建模结果从系统内部变量转化为可观察、可回答的交互界面，是连接推荐推理与用户反馈的重要步骤。

### 9.3 局限与后续工作

Test-300 中 Student 有 293/300 条通过全部硬规则；其余主要为 6 条选项长度失衡，以及 1 条具体名称泄漏并伴随角色句式问题。MovieLens 的 hard pass 为 95/100，是后续定向增强的主要数据域。这些问题不影响 299/300 的模型 Judge 语义通过主结果，但说明部署时仍应保留已经实现的 JSON 和 hard audit；若未通过，则触发约束重生成或回退到已验证安全文本。

语义主结果来自 DeepSeek 模型 Judge。实际响应元数据显示 Judge 与冻结 Teacher 测试输出均标识为 `deepseek-v4-flash`；二者虽然使用独立提示和独立证据路径，但并非模型独立，仍可能存在同源评价偏差。目前尚未完成人工盲评或异构模型 Judge 复核。Teacher 对比也基于每条样本的一份冻结生成结果，没有评估多次随机调用的方差。

本节案例是从 Test-300 中按领域和角色清晰度人工选择的说明性案例，用于解释模型行为，不作为总体优越性的频率证据；总体结论以完整 300 条统计为准。

后续工作主要包括：

1. 进行人工盲评，比较自然度、清晰度和可选择性；
2. 在线记录 A/B/C/NONE 选择率、响应时间和后续推荐收益；
3. 验证选择题是否减少推荐交互轮数；
4. 补全 LastFM 正式文本投影后，增加统一的域内向量诊断；
5. 针对少量边界样本继续进行定向数据增强。

## 10. 结论

本工作完成了从 DeepSeek API Teacher 到 Qwen2.5-1.5B Student 的意图表层生成蒸馏。Student 能够根据三个结构化 intent slot，生成严格 JSON 格式的中性问题和 A/B/C/NONE 选项，并通过自然语言表达“保持、邻近移动、探索”三种角色。

固定 Test-300 上，Student 的模型 Judge 语义忠实率达到 **99.67%**，DeepSeek Teacher 为 **99.33%**；Student−Teacher 的 95% CI 为 `[-0.68, +1.40]` 个百分点，未观察到显著差异。该结果不能替代预设非劣界值的正式非劣检验，但结合 100% schema 通过率、三域高通过率和典型案例，支持本次蒸馏已经有效迁移 Teacher 的角色理解、证据压缩和用户可读表达能力，并在当前测试精度下表现出与 API Teacher 接近的任务质量。

Student 在部分案例中给出了比 Teacher 更具体或更简洁的概括，在辅助形式空间中也表现出更高的选项分离度；Teacher 则在 BGE own alignment 上略高。因此，本工作不仅复现了 API 输出形式，而且获得了一个能够按照固定规则、客观展示三种意图假设，并支持用户快速反馈的本地生成模块。

## 11. 复现实验产物

- 冻结数据清单：`/root/autodl-tmp/intent_teacher_v2/dataset_manifest.json`
- Teacher 候选清单：`/root/autodl-tmp/intent_teacher_v2/current_teacher_candidates/release_manifest.json`
- 语义审计后数据：`/root/autodl-tmp/intent_teacher_v2/current_teacher_dataset/teacher_dataset_manifest.json`
- 训练配置：`/root/interrec-noapi_gpu/artifacts/intent_teacher_v2/training_config_lr1e4.yaml`
- 训练配置：`/root/interrec-noapi_gpu/artifacts/intent_teacher_v2/training_config_lr2e4.yaml`
- 最终训练清单：`/root/autodl-tmp/intent_teacher_v2/adapters/qwen2p5_1p5b_qlora_lr2e4/run_manifest.json`
- checkpoint 选择：`/root/autodl-tmp/intent_teacher_v2/checkpoint_selection/checkpoint_selection.json`
- Test-300 逐样本结果：`/root/autodl-tmp/intent_teacher_v2/evaluation/per_example_metrics.jsonl`
- 配对 bootstrap：`/root/autodl-tmp/intent_teacher_v2/evaluation/bootstrap_results.json`
- Student 预测：`/root/autodl-tmp/intent_teacher_v2/predictions/current/new_student.jsonl`
- Teacher 预测：`/root/autodl-tmp/intent_teacher_v2/predictions/current/deepseek_teacher.jsonl`
