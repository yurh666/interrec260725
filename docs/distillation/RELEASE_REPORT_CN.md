# Intent Teacher v2 蒸馏子实验详细报告

> **interrec-yrh / interrec260725：** 主实验 `surface_writer` 已接线 **本报告同一 Intent Teacher v2 Student**（`outputs/intent_teacher_v2/lora_adapter`）。Test-300 指标描述 **surface 生成质量**；Hit@10 等主表见 `../interrec-yrh-outcome/*/run_summary.json`（与 surface 文案解耦）。

## 结论摘要

在固定 Test-300（Book、MovieLens、LastFM 各 100）上，本地 1.5B Student 达到 100% schema、97.67% hard constraint、99.67% semantic judge。DeepSeek API Teacher 分别为 100%、98.00%、99.33%。Student 的主要发布价值是本地、固定权重、可复现且语义忠实率点估计与 API Teacher 相当；不能把 1 条样本的语义优势解释为全面胜出。

## 任务和资产身份

新版任务是 Intent Teacher v2：从三候选意图证据生成满足严格 schema 和证据约束的澄清问题/选项表面文本。旧版 `surface_writer_qwen` LoRA 是独立历史资产。新版训练 manifest 明确从干净 Qwen2.5-1.5B Base 初始化且 `old_adapter_loaded=false`；两者不可互换、叠加，也不能跨任务归因结果。

## 评估协议

Test-300 在 validation-only checkpoint 选择后只读访问。每个系统的分母固定为 300；生成、解析与审计失败均计失败。总体及域内二元指标报告 Wilson 95% CI；逐例、bootstrap、硬约束和投影验证文件完整保留。

## 主要结果

- Schema：双方均 300/300。
- Hard：Student 293/300，Teacher 294/300。分域 Student/Teacher：Book 99/98，LastFM 99/100，MovieLens 95/96。
- Semantic：Student 299/300，Teacher 298/300。分域：Book 100/99，LastFM 99/99，MovieLens 100/100。
- Student 在 Book 的 hard 与 semantic 点估计均高 1 个样本；在 LastFM hard 和 MovieLens hard 各低 1 个样本。
- 选项分离度正式 BGE（仅 Book+MovieLens 可估）Student 为 0.3082，Teacher 为 0.2896；但总体正式 BGE 均只有 200/300 coverage。LastFM 正式 BGE 为 0/100，任何全域比较都必须披露这一缺失。

## 优势与完整披露

可强调的优势是：本地 1.5B Student 不依赖在线 API；版本、Base 与 Adapter 均有哈希；schema 通过率无损；语义点估计略高；在部分 option-separation 诊断上点估计更高。必须同时披露：hard 总体少通过 1 条；MovieLens hard 是 95%；item leakage 为 1/300（Teacher 0）；正式 BGE 对 LastFM 不可估；固定 300 条不足以支持广泛泛化或显著优越性结论。

## 复现与 provenance

交接 tar 的 47 个成员经路径/链接安全审计，35 个 manifest 条目逐一通过大小和 SHA256 校验。Base `model.safetensors` 的 SHA256 与声明完全一致。机器真源、派生表、案例和版本链分别位于 `data/`、`evaluation/`、`tables/`、`cases/`、`provenance/`。

## Provenance 警告

顶层 `PACKAGE_MANIFEST` 的 35 个文件均通过校验，但嵌套历史 manifest 有两处版本引用差异：评估 manifest 的中文报告哈希指向较早版本；winner/prediction manifest 的 checkpoint `adapter_config.json` 哈希与发布态 inference config 不同。Adapter 权重哈希保持一致。详见 `RELEASE_VALIDATION.json` / `package_validation.json`，不得把这些历史引用描述为字节同一。
