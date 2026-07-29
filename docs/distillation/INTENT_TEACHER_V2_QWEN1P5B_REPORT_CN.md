# Intent Teacher v2 / Qwen2.5-1.5B Test-300 详细评估报告

> **Release 权重：** `outputs/intent_teacher_v2/lora_adapter`（SHA256 与 staging winner 一致）。主表推荐指标不读本 Test-300，见 `docs/SURFACE_WRITER_LORA_PROVENANCE_CN.md`。

- 评估完成时间：2026-07-17T15:10:38.609710+00:00
- 评估器：`intent_teacher_v2_test300_evaluator_v3`
- 固定测试集：300 条，Book / LastFM / MovieLens 各 100 条
- heldout SHA256：`7d2fa63904dd1beba1978346af8f10200c5f45a2f947a3d7c51cbd4f961e54ce`
- 对比系统：新 Qwen2.5-1.5B Student 与 DeepSeek Teacher
- Student winner：`lr2e4/checkpoint-300`
- 统计方法：按数据域分层、按 `group_key` 聚类的 10,000 次 paired bootstrap

## 1. 结论先行

本次训练是有效的。1.5B Student 已经学会任务的核心，而不只是学会输出一个看起来像选择题的模板：

1. **格式稳定**：300/300 都能生成可解析的严格 JSON，与 Teacher 持平。
2. **角色忠实**：Student 的硬约束通过率为 **97.67%（293/300）**，Teacher 为 **98.00%（294/300）**。两者差值仅 -0.33 个百分点，95% CI 为 `[-2.31, +1.64]` 个百分点，不能认为 Student 显著弱于 Teacher。
3. **语义忠实**：独立 Judge 判定 Student **99.67%（299/300）** 通过，Teacher **99.33%（298/300）**。Student 反而高 0.33 个百分点，但 CI 跨 0，正确结论是“达到 Teacher 同一水平”，而不是宣称反超。
4. **形式空间中的选项分离度更高**：在可正式估计的 Book + MovieLens 200 条上，Student 的 option separation 为 **0.3082**，Teacher 为 **0.2896**；paired 差值 **+0.0186**，95% CI `[+0.0092, +0.0281]`，这是统计上稳定的优势。它说明 A/B/C 在 embedding 空间中平均更分离；是否让真实用户更容易选择，仍需人工评测或在线点击实验验证。
5. **Teacher 仍有小幅语义贴合优势**：正式 BGE own-alignment Student 为 **0.4506**，Teacher 为 **0.4602**；差值 **-0.0096**，95% CI `[-0.0153, -0.0042]`。差距绝对值不大，但真实存在，说明 Student 有时会把证据压缩得更概括，或把标签直接罗列出来。
6. **当前不建议无保护直接上线**：Student 有 **7/300 条硬约束失败**，其中 6 条为长度失衡，另 1 条同时包含 item/name 泄漏和缺失 B 角色立场。整体模型能力已经达到离线候选水平，但部署前应让生成后审计、重试和回退覆盖全部硬失败类型。

因此，最准确的产品判断是：

> **新 Student 已成功逼近 Teacher，核心意图映射、文本可读性和选项角色均达到可用水平；当前是“离线候选模型，尚非部署候选”，完成硬审计、重试/安全回退和全量回归后，才适合进入候选部署。**

## 2. 为什么能说明模型“接近原始 intent”

任务中的 A / B / C 不是普通分类标签，而是三个不同的推荐意图：

- A / preserve：保留当前偏好；
- B / boundary：保留一部分语义锚点，同时向邻近方向移动；
- C / explore：提供独立且更远的探索方向；
- NONE：固定退出项。

Student 是否接近原始 intent，不能只看生成文本是否流畅。本评估同时检查了四层证据：

### 2.1 独立语义 Judge

Judge 不读取 Teacher 答案，只读取原始 A/B/C evidence 与待评输出，逐项判断：

- A 是否受 A 证据支持；
- B 是否同时具备“保留锚点 + 邻近移动”；
- C 是否受 C 证据支持；
- 是否发生跨角色污染；
- 问题是否中性；
- 是否加入无证据主张。

Student 299/300 通过，说明它几乎总能把结构化 evidence 转成正确角色，而不是仅模仿 Teacher 表面措辞。语义通过不等于全部硬约束通过：长度平衡、item 泄漏等仍由独立 hard audit 管理。

### 2.2 BGE own alignment

正式空间中，Student 的 own alignment 为 0.4506，Teacher 为 0.4602。Teacher 小幅领先，符合强模型通常能更精细压缩证据的预期。paired bootstrap 已确认 -0.0096 的差距显著，因此只能说绝对差值较小，不能把两个 cosine 派生分数相除并解释成“能力保留百分比”。

### 2.3 Role retrieval

把每个输出选项与 A/B/C 三组证据比较，看它最接近自己的角色还是错误角色：

- Student：0.6667
- Teacher：0.6917
- paired 差值：-0.0250
- 95% CI：`[-0.0516, +0.0016]`

Teacher 有数值优势，但 CI 仍跨 0。结合 99.67% 独立语义通过率，说明不少 retrieval 错位来自上游 A/B/C 本身相近或重复，而不一定是自然语言角色写错。

### 2.4 Own-vs-cross margin

- Student：0.0641
- Teacher：0.0627
- paired 差值：+0.00146
- 95% CI：`[-0.00373, +0.00645]`

两者基本相当。Student 的选项平均仍更接近自己的 evidence，而不是其他角色的 evidence。

## 3. 为什么能说明文本质量好、而且接近 Teacher

### 3.1 严格 JSON 与固定交互契约

两个系统均为 300/300 schema pass。Student 可以稳定输出：

- 一个中性问题；
- 恰好 A / B / C / NONE 四个键；
- 固定 NONE 文本；
- 无多余解释、Markdown 或前后缀。

这对 1.5B 模型很重要，因为产品侧无需再做脆弱的自由文本解析。

### 3.2 句式表达了角色，不依赖内部术语

高质量 Student 输出稳定使用：

- A：`Continue with ...`
- B：`Keep ... while moving toward ...`
- C：`Try ...`

这不是单纯模板化缺点。模板承担的是角色语义，后半句内容仍由 evidence 决定。尤其 B 同时出现“保留”和“移动”两个动作，正是本任务最难的 boundary 语义。

### 3.3 Student 比 Teacher 更容易形成可选择的三个方向

正式 option separation：

- Student：0.3082
- Teacher：0.2896
- Student - Teacher：+0.0186
- 95% CI：`[+0.0092, +0.0281]`

这与用户给出的优秀 MovieLens 示例所强调的“问题中性、A 延续、B 邻近转向、C 明显探索”方向一致。Student 并非总在每个维度都优于 Teacher；当前能严格证明的是 embedding 形式空间中的分离度更高，“用户是否一眼看懂”仍需人工或在线实验确认。

### 3.4 与 Teacher 的关系是“意图一致、措辞不必逐字一致”

本任务不要求复制 Teacher 文本。好的 Student 应该：

- 与 Teacher 选择相同的语义锚点；
- 保持 A/B/C 角色一致；
- 可以用更短、更自然的措辞；
- 不泄漏具体 item 名称或内部构造信息。

下面的案例显示 Student 与 Teacher 往往只在同义表达和信息密度上有差异。

## 4. 特别精彩的案例

### 案例 1：MovieLens｜最接近参考图片的自然选项

Sample：`024cb6fe34e5e978d88ec4b9`

原始 evidence：

- A：Adventure / Children's / Sci-Fi / Fantasy
- B：Animation / Children's / Musical / Comedy
- C：Comedy

Student：

> Question: What kind of movie experience are you looking for?  
> A. Continue with adventurous stories featuring fantasy and sci-fi elements.  
> B. Keep the family-friendly feel while moving toward animated musical comedies.  
> C. Try lighthearted comedy films without complex narratives.  
> NONE. None of these options fits what I want.

Teacher：

> A. Continue with adventurous stories that mix sci-fi and fantasy elements.  
> B. Keep the family-friendly feel while moving toward animated musical comedies.  
> C. Try something lighthearted and funny, focusing on comedy.

为什么精彩：

- A 保持奇幻、科幻与冒险体验；
- B 用 `family-friendly feel` 建立自然桥梁，再转向动画歌舞喜剧；
- C 给出更简单、独立的轻喜剧探索方向；
- 三个方向完全可区分；
- 与 Teacher 在 A/B/C 上几乎一一同义。

这是最接近用户提供图片的风格：短、中性、自然、三项边界清楚。谨慎点是 Student 的 `without complex narratives` 不是直接 genre 标签；尽管已通过独立语义审计，Teacher 的 C 更保守。

### 案例 2：MovieLens｜从“兴奋感”建立自然的边界迁移

Sample：`5a79dd90bd46fb46879c5e57`

原始 evidence：

- A：Action / Adventure / Sci-Fi
- B：Horror / Thriller
- C：Drama / War / Romance

Student：

> A. Continue with action-packed adventures and sci-fi elements.  
> B. Keep the excitement while moving toward horror and thriller themes.  
> C. Try dramatic stories with war and romance.

Teacher：

> A. Continue with action-packed, adventurous sci-fi films.  
> B. Keep the excitement while moving toward horror and thriller elements.  
> C. Try dramatic stories with war and romance themes.

为什么精彩：

- B 找到了 A 与 B 的共同桥梁 `excitement`，既不是简单重复 A，也不是突然跳到 Horror；
- C 与 A/B 的高刺激类型形成清楚探索方向；
- Student 和 Teacher 的三项几乎一一同义，证明 Student 学到的是 Teacher 的意图组织方式。

### 案例 3：LastFM｜Student 比 Teacher 更严格贴证据

Sample：`5d91368b2d6d4b94ed7c89da`

原始 evidence 摘要：

- A：acoustic / singer-songwriter / indie folk
- B：new wave / synthpop / electronic / 80s / post-punk
- C：punk / garage rock / indie rock / electronic

Student：

> A. Continue with acoustic singer-songwriter and indie folk.  
> B. Keep a melodic feel while moving toward new wave and synthpop.  
> C. Try punk and garage rock with some electronic elements.

Teacher：

> A. Continue with acoustic singer-songwriter and indie folk.  
> B. Keep acoustic and indie while moving toward new wave and synth pop.  
> C. Try indie rock with punk and garage rock influences.

为什么精彩：

- A 两者都准确且简洁；
- Student 的 B 用安全的 `melodic feel` 作桥梁，没有声称 B evidence 本身含 acoustic/indie；
- Teacher 的 B 写成 `Keep acoustic and indie`，把 A 属性带入 B，因而独立语义审计失败；
- Student 的 C 直接覆盖 punk / garage rock / electronic；
- 这是最强的 Student 优于 Teacher 案例：优势来自减少跨角色无依据继承，而不是写得更多。

### 案例 4：LastFM｜跨年代与地域的三个清楚方向

Sample：`cb986bb89b1992f4c4c5a165`

Student：

> A. Continue with British indie rock and 90s alternative.  
> B. Keep British rock while moving toward classic rock and hard rock.  
> C. Try Japanese indie rock and metal influences.

Teacher：

> A. Continue with British indie and alternative rock from the 90s, with some electronic and synth pop influences.  
> B. Keep the British rock focus while moving toward classic and progressive rock from the 70s and 80s.  
> C. Try indie rock and alternative with a mix of metal and Japanese rock for a fun twist.

为什么精彩：

- Student 用非常短的句子保留了年代、地域、流派三个核心维度；
- A 保持 90s British indie；
- B 在 British rock 锚点上向 classic/hard rock 移动；
- C 明显探索 Japanese rock/metal；
- Teacher 信息更多，Student 更适合 UI 展示，且未损失核心方向。

### 案例 5：Book｜从混合书目提取可用主方向

Sample：`57613f358c7e5ec36fa7542b`

原始 evidence 摘要：

- A：以社会、科技与历史类非虚构为主，夹有小说和烹饪内容
- B：人物传记、宗教人物回忆、概率史与社会政治叙事
- C：职业发展、投资、批判思维和个人成长指南

Student：

> A. Continue with non-fiction works that offer insightful perspectives on society, technology, or culture.  
> B. Keep a focus on real-world issues while moving toward narrative-driven accounts of influential figures or historical events.  
> C. Try books that provide practical guidance on career development, investing, or personal growth strategies.

Teacher：

> A. Continue with thought-provoking nonfiction that challenges conventional wisdom.  
> B. Keep a focus on influential figures and ideas while moving toward personal narratives and memoirs.  
> C. Try practical guides that offer actionable advice for personal or professional development.

为什么精彩：

- A 的 society / technology / culture 抓住了混合 evidence 的主体信号，比 Teacher 的 `challenges conventional wisdom` 更具体，但没有覆盖其中的小说和烹饪支线；
- B 完整表达“保留真实世界议题，再转向人物或历史叙事”；
- C 的 career / investing / personal growth 与原 evidence 几乎逐项对应；
- 三项形成“社会科技主题 → 叙事型真实世界作品 → 可操作实用指南”的自然梯度；
- 说明 Student 不只是复述 Teacher，有时能以更具体、更可选择的方式概括原始 intent。

### 案例 6：Book｜历史社会 → 悬疑叙事 → 小镇温情

Sample：`8d63909911cdec6763eb62c9`

Student：

> A. Continue with non-fiction accounts of historical events and cultural movements.  
> B. Keep a focus on narrative-driven stories while moving toward mystery and suspense elements.  
> C. Try heartwarming tales of small-town life and gentle fiction.

Teacher：

> A. Continue with non-fiction accounts of historical events and social movements.  
> B. Keep a focus on real-world settings while moving toward suspenseful crime and mystery stories.  
> C. Try heartwarming tales of small-town life and gentle fiction.

为什么精彩：

- Student 与 Teacher 在 A、C 上几乎同义；
- B 都捕捉到叙事性与 mystery/suspense 的边界移动；
- 三个选项主题距离清楚，且 formal role retrieval 为 1.0；
- 长篇、噪声较多的书籍简介被压成适合选择题 UI 的一句话。

## 5. 失败与边界案例

### 5.1 最高风险失败：具体艺人名泄漏

Sample：`8ae957ff01587266bc2ae48b`

Student 的 A 退化为长标签列表，并直接包含 `britney spears`：

> Continue with pop, folk, 90s, rock, soundtrack, british, female vocalists, dance, latin, latin pop, legend, britney spears, teen pop, electronic.

问题：

- 泄漏具体 item/artist 名称；
- 文本像标签 dump，不像面向用户的自然选项；
- A/B/C 都偏长，可扫描性差。

这 1 条使 item/name leakage 从 0 变成 1/300，因此自动部署门槛没有通过。修复并不一定需要重训整个模型：优先增加 artist/name 词典审计，失败时进行一次约束重生成；随后再考虑把这种长标签样本加入针对性 SFT。

### 5.2 其余硬失败：6 条长度失衡

除上述泄漏样本外，Student 还有 6 条 `length_imbalance`：Book 1 条、MovieLens 5 条。它们通常不改变 A/B/C 的大方向，因此独立语义通过率仍可很高，但会降低 UI 可扫描性，也违反既定输出契约。部署防护不能只拦具体名称，还应检测选项长度比例并触发重生成。

### 5.3 上游结构冲突不是模型错误

部分样本 A/B 或 B/C representative item set 相同。此时任何文本模型都无法从相同 evidence 唯一恢复两个不同方向。评估已把这些样本标为 structural conflict，并保留在总体分母中。

这解释了为什么：

- 独立语义 Judge 通过率接近 100%；
- BGE role retrieval 只有约 2/3；
- 两者并不矛盾。

Judge 能结合角色要求判断文本是否合理；embedding retrieval 在两个证据向量相同或极近时无法分辨角色。

### 5.4 LastFM 正式 PCA128 不可估

LastFM 的冻结 PCA128 文本投影未通过严格重放，因此正式指标标记为 `NOT_ESTIMABLE`，没有把 raw 1024d 辅助结果冒充正式 PCA 结果。

辅助重建 raw 空间覆盖 300 条，可用于诊断：

- own alignment：Student 0.6136，Teacher 0.6223；
- role retrieval：Student 0.7567，Teacher 0.7689；
- option separation：Student 0.2723，Teacher 0.2595。

这些趋势与正式 Book + MovieLens 结果一致，但不能替代 LastFM PCA128 正式结论。

## 6. 指标总表

| 指标 | Student | Teacher | Student - Teacher | 95% paired CI | 判断 |
|---|---:|---:|---:|---:|---|
| Schema pass | 1.0000 | 1.0000 | 0.0000 | [0, 0] | 完全持平 |
| Hard pass | 0.9767 | 0.9800 | -0.0033 | [-0.0231, 0.0164] | 无显著差异 |
| Semantic pass | 0.9967 | 0.9933 | +0.0033 | [-0.0068, 0.0140] | 无显著差异 |
| Formal own alignment | 0.4506 | 0.4602 | -0.0096 | [-0.0153, -0.0042] | Teacher 小幅显著领先 |
| Formal role retrieval | 0.6667 | 0.6917 | -0.0250 | [-0.0516, 0.0016] | Teacher 数值领先，未达显著 |
| Formal own-vs-cross margin | 0.0641 | 0.0627 | +0.0015 | [-0.0037, 0.0064] | 基本持平 |
| Formal option separation | 0.3082 | 0.2896 | +0.0186 | [0.0092, 0.0281] | Student 显著领先 |

正式 BGE 仅覆盖 Book raw 1024d 与 MovieLens PCA256，共 200 条。

## 7. 向量空间门禁

- Book：`fixed_revision_reconstructed_raw_bge_m3_1024` — **VERIFIED_RECONSTRUCTED_RAW**
- MovieLens：`frozen_bge_m3_pca256` — **VERIFIED**
- LastFM：`frozen_bge_m3_pca128` — **NOT_ESTIMABLE**

Book 正式空间就是 raw BGE-M3 1024d。生成选项与代表 item 元数据由同一固定 revision 重编码，不混用不同模型或不同维度。Book PCA256 指纹冲突只影响 PCA 实验分支，不影响本次 raw 正式评估。

## 8. 训练与选模说明

根据 `checkpoint_selection/checkpoint_selection.json` 与两个训练目录的 `run_manifest.json`，最终选择 `lr2e4/checkpoint-300`；selection manifest 记录 `validation_only=true`、`heldout_read=false`。训练配置与 run manifest 记录训练从固定 Qwen2.5-1.5B-Instruct 基座快照开始、未从旧 adapter 初始化，并使用 assistant-only causal LM loss。这些是训练过程 provenance 声明，不是由 Test-300 指标本身反推得到。

过程偏差需要保留记录：两组训练沿用旧 preflight，曾读取 heldout 的 ID/group/evidence 指纹用于泄漏检查，但没有读取 target，也没有用于梯度更新或 checkpoint 选择。后续代码已改为 winner 冻结前不读取 heldout 内容。该偏差不太可能抬高本文的生成指标，但属于实验流程上的不完美，不能隐去。

## 9. 部署建议

建议状态：**离线候选模型，尚非部署候选；不建议无保护直出。**

最小上线闭环：

1. 保留严格 JSON parser；
2. 保留现有 hard audit；
3. 新增 artist / title / ID 泄漏词典、角色立场与选项长度比例检测；
4. 审计失败时约束重生成一次；
5. 第二次仍失败时，Teacher 回退也必须经过同一 hard audit；若仍失败，只回退已验证安全模板；
6. 用真实用户点击 / NONE 率继续验证 A/B/C 的产品价值。

先让固定 300 条上的 7 个 hard reject 都能被重生成或安全回退覆盖，并把 item/name leakage 从 1 降到 0；随后回归确认 schema / hard / semantic 不下降，才可进入候选部署。

## 10. 产物索引

- 总体与分域指标：`evaluation/summary_metrics.json`
- 配对 bootstrap：`evaluation/bootstrap_results.json`
- 逐样本指标：`evaluation/per_example_metrics.jsonl`
- 自动定性案例：`evaluation/qualitative_cases_cn.md`
- Student 预测：`predictions/current/new_student.jsonl`
- Teacher 预测：`predictions/current/deepseek_teacher.jsonl`
- checkpoint 选择：`checkpoint_selection/checkpoint_selection.json`
- 文本投影验证：`evaluation/text_projection_validation.json`
