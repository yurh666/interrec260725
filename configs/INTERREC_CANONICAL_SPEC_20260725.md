# InterRec Canonical Spec — **interrec260725** (2026-07-25)

> **相对 interrec260724 的唯一 intentional 变更**：`surface_writer` 从 `mock_template` 换为 **Intent Teacher v2 LoRA**（`local_small_model` → `outputs/intent_teacher_v2/lora_adapter`）。
> 假设生成、cluster-bound、integrated 代表 item、belief、VOI、用户模拟（deterministic argmax）与 260724 **同一 yaml 继承链**，仅 surface 块 override。

**前置**：260724 Book mock full 跑满后再启动 260725 串行（见下方脚本）。

---

## 1. 配置（三数据集 full）

| 数据集 | 260724（mock surface） | 260725（distill surface） |
|--------|-------------------------|---------------------------|
| MovieLens | `configs/runs/movielens_full_integrated_mock_surface_seed42.yaml` | `configs/runs/movielens_full_integrated_distill_surface_seed42.yaml` |
| LastFM | `configs/runs/lastfm_full_integrated_mock_surface_seed42.yaml` | `configs/runs/lastfm_full_integrated_distill_surface_seed42.yaml` |
| Book | `configs/runs/book_full_integrated_mock_surface_seed42.yaml` | `configs/runs/book_full_integrated_distill_surface_seed42.yaml` |

共享 surface 片段：`configs/final_unified/shared/surface_distill_260725.yaml`

**run_id**：

- `movielens_full_integrated_distill_surface_seed42`
- `lastfm_full_integrated_distill_surface_seed42`
- `book_full_integrated_distill_surface_seed42`

---

## 2. 与 Method / final_results 的关系

- Method：MCQ **可见文案**由 surface writer 生成；**选项 ID / 假设向量 / VOI** 不依赖 surface 文本。  
- 用户模拟为 **deterministic_argmax**（按选项 ID），故在 LoRA 正常、无 fallback 改变选项结构时，**排序指标应与 260724 mock / `final_results_seed42` 假设链一致**（与 smoke 设计相同）。  
- 若 LoRA 加载失败会 `fallback_mode: mock_template`（与 260724 等价）。

260724 规范仍有效：`configs/INTERREC_CANONICAL_SPEC_20260724.md`（hypothesis 链、Book gate 等）。

---

## 3. 自动串行（Book mock 完成后 → 三数据集 distill）

```bash
cd /home/yrh666/interrec-final-github
bash scripts/run_interrec260725_distill_tmux.sh
```

| 组件 | 说明 |
|------|------|
| tmux `interrec260725_distill_queue` | 等待 `book_full_integrated_mock_surface` 结束且 **2107** session 完成 |
| 然后串行 | `run_interrec260725_full_distill_serial.sh`：**Book → LastFM → MovieLens** |
| 队列日志 | `experiments/final_unified_seed42/logs/interrec260725_distill_queue.log` |
| 跑数日志 | `experiments/final_unified_seed42/logs/interrec260725_full_distill_serial.log` |
| 完成标记 | `experiments/final_unified_seed42/logs/.interrec260725_distill_serial_done` |

监控：

```bash
cd /home/yrh666/interrec-final-github
bash scripts/watch_interrec260725_distill_progress.sh
```

（每 30s 刷新三数据集进度 + 10 项 partial 指标；`INTERVAL=10` 可加快。）

手动只跑 distill 串行（不等待）：

```bash
bash scripts/run_interrec260725_full_distill_serial.sh
```

**Distill 串行成功后自动接上 Subexp 6A/6B**（baseline = 三数据集 `*_full_integrated_distill_surface_seed42`，6A `compatible_update_ema=0.05`，6B `c_ask×1.25`，surface = 蒸馏 LoRA）：

- 脚本：`scripts/run_subexp6_interrec260725_serial.sh`（由 `wait_book_mock_then_interrec260725_distill.sh` 在 `.interrec260725_distill_serial_done` 后调用）
- 日志：`experiments/final_unified_seed42/logs/subexp6_interrec260725_serial.log`

**子实验 intent / 模糊度**（Book 已用 distill full；ML/LF 在 distill 完成前仍读 `final_results_seed42`）：

```bash
bash experiments/in_depth/scripts/run_refresh_subexp_intent_ambiguity_260725.sh
```

---

## 4. 变更记录

| 日期 | 内容 |
|------|------|
| 2026-07-25 | interrec260725：三数据集 full + distill surface；自动 queue 脚本 |
