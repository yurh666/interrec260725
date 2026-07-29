# Intent Teacher v2 Surface Writer — Base 模型与推理

## 权重

- Adapter：`outputs/intent_teacher_v2/lora_adapter/`（Intent Teacher v2 winner，Test-300 同 lineage）
- 推理 JSON：`outputs/intent_teacher_v2/inference_config.json`

## Base 模型（必设其一）

1. 环境变量（推荐）：

```bash
export INTERREC_BASE_MODEL_PATH=/home/you/models/Qwen2.5-1.5B-Instruct
```

2. 或仓库内 symlink：`models/Qwen2.5-1.5B-Instruct` → 本地 HuggingFace 目录

3. 或本机 HF cache（`local_inference._resolve_base_model_path` 会尝试解析 `Qwen/Qwen2.5-1.5B-Instruct`）

## GPU / CPU

- 默认配置为 `device: cpu`（无 GPU 时可跑 smoke，full 很慢）。
- 有 GPU 时建议编辑 `inference_config.json`：

```json
"device": "cuda",
"load_in_4bit": true
```

训练为 4bit QLoRA；CPU 全精度 fp32 也可加载，需足够内存。

## 与主实验指标

Surface 在 `session_simulator.py` 中 **belief update 之后** 调用，**不参与** VOI 与用户模拟选题。换 v2 仅影响 log 文案与 `implementation_modes.surface_writer` 审计字段。

详见 `docs/SURFACE_WRITER_LORA_PROVENANCE_CN.md`。
