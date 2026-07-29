# Surface Writer LoRA — Base 模型接入说明

本包 **不包含** Qwen2.5-1.5B-Instruct 基座权重（体积过大）。已包含：

- `lora_adapter/adapter_model.safetensors`
- `lora_adapter/adapter_config.json`
- `inference_config.json`（`base_model_path` 支持 **`${INTERREC_BASE_MODEL_PATH}`**，代码会 `os.path.expandvars`）

`adapter_config.json` 内 **`base_model_name_or_path` 已改为 Hub 名** `Qwen/Qwen2.5-1.5B-Instruct`，不再引用训练机绝对路径；本地目录优先用环境变量或 `models/Qwen2.5-1.5B-Instruct`。

## 1. 下载基座

```bash
export INTERREC_BASE_MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct
# 或 symlink: ln -sfn /path/to/Qwen2.5-1.5B-Instruct models/Qwen2.5-1.5B-Instruct
```

## 2. Python 依赖（surface 推理）

```bash
pip install torch transformers accelerate sentencepiece protobuf peft
```

Tokenizer **从 base 目录加载**（非 adapter 目录）。

## 3. 运行前

```bash
export INTERREC_BASE_MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct
cd /path/to/interrec-yrh
python3 scripts/run_final_interrec.py --list-datasets
bash scripts/run_interrec_yrh_smoke_all.sh   # 需本包自带 data/
```

## 4. 与 Intent Teacher v2 报告的关系

见 `docs/distillation/RELEASE_REPORT_CN.md`：`surface_writer_qwen` LoRA ≠ Intent Teacher v2 adapter。
