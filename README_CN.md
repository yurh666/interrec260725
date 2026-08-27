# ChoiceRec

ChoiceRec 在 InterRec **260725 full**（integrated + γ=0.05 + Intent Teacher v2 surface LoRA）之上，增加了 **在线 valid 逐步释放** 的交互协议。

推荐 / Frontier / VOI / belief / memory 等核心模块与 InterRec 共用；ChoiceRec 仅新增协议层与配置开关（`simulation.protocol: choicerec`）。

## 文档

- **ChoiceRec 协议审计**：`CHOICEREC_PROTOCOL_AUDIT_CN.md`
- **InterRec 方法全文**：`METHOD_CN.md`
- **Pipeline 代码对照**：`PIPELINE_CN.md`
- **260725 跑数路径补充**：`docs/INTERREC260725_PATH_ADDENDUM_CN.md`
- **InterRec full 配置索引**：`configs/final_unified/manifest_yrh.yaml`

## 快速开始（ChoiceRec full，seed 42）

```bash
pip install -r requirements.txt

# 三数据集串行：LastFM → MovieLens → Book
bash scripts/run_choicerec_full_serial.sh

# 或单数据集
python3 scripts/run_main_experiment.py \
  --config configs/choicerec/book_full_seed42.yaml \
  --run-id choicerec_book_full_seed42
```

## ChoiceRec 新增代码

| 组件 | 路径 |
|------|------|
| 协议状态机 | `src/choicerec/protocol.py` |
| 动态 θ* | `src/choicerec/theta_builder.py` |
| 子实验辅助 | `src/choicerec/subexperiment_runtime.py` |
| Session 集成 | `src/simulation/session_simulator.py` |
| Full 配置 | `configs/choicerec/*_full_seed42.yaml` |

## InterRec full（原有用法不变）

```bash
python3 scripts/run_final_interrec.py --dataset book
```
