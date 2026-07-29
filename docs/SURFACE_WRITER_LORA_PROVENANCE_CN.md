# Surface Writer LoRA 资产（interrec-yrh / interrec260725）

**复核日期：** 2026-07-26  
**Canonical Student：** **Intent Teacher v2**（Qwen2.5-1.5B-Instruct QLoRA，validation 选 checkpoint-300）

## 结论

| 项 | 值 |
|----|-----|
| Release 配置加载路径 | `outputs/intent_teacher_v2/lora_adapter` |
| `adapter_model.safetensors` SHA256 | `7efee94ca8689affcd207e1538c0afc8d5396ef20cbc2302ce4f91496608d8d6` |
| 与 Test-300 报告 | **同一 winner 权重**（见 `docs/distillation/INTENT_TEACHER_V2_QWEN1P5B_REPORT_CN.md`） |
| 对主表 Hit@10 / NDCG 等 | **不改变**（surface 在用户选择之后写 log；模拟用户按 hypothesis 向量 argmax） |

## 环境

```bash
export INTERREC_BASE_MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct
# 或放置于 interrec-yrh/models/Qwen2.5-1.5B-Instruct
```

推理配置：`outputs/intent_teacher_v2/inference_config.json`（`device`: 有 GPU 时可改为 `cuda` 并设 `load_in_4bit: true` 以省显存）。

## 历史 legacy adapter（勿与 v2 混归因）

旧目录 `outputs/surface_writer_qwen/lora_adapter/`（SHA256 `444433a9786…`）为 **早期 surface_writer_qwen** 训练线，**不是** 当前 canonical。保留仅供对照；论文/Test-300 数字对应 **v2**。

## 代码路径

- 配置键：`surface_writer.mode=local_small_model`，`local_model_path`，`inference_config_path`
- 实现：`src/interrec_noapi/surface_writer/writer.py` → `local_inference.generate_surface_text()`
- 相对路径解析：相对 **进程 cwd** 或 **仓库根**（`interrec-yrh/`）

## 已跑 full 与 v2 的关系

2026-07-25 归档的三数据集 full（见 `../interrec-yrh-outcome/`）在 **假设/VOI 链** 上已是 interrec260725 最终版；当时 surface 常因 LoRA 加载失败或未接 v2 而记为 `mock_template`，**主表 aggregate 仍有效**。后续仅用 v2 重跑 surface 时，预期 **aggregate 不变**、log 中 `surface_writer` / 问题文案可能变化。
