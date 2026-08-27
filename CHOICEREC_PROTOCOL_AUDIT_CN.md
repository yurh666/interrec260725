# ChoiceRec Protocol Audit (vs interrec260725)

## 修改项

1. **初始可见历史** = `observed_history + future_train`（按时间序拼接）
2. **`future_valid` 在线释放**：每 turn 结束后按 chunk 规则 reveal，更新 behavioral history 与 `seen_items`
3. **Behavior / Choice 解耦**：
   - behavioral reveal：只更新 `mu`（via `behavior_mu + choice_residual`），**不改 Sigma**
   - multiple-choice：保持原 Laplace update（`mu + Sigma`）
4. **动态 simulator 目标** `theta*_t = Normalize(mean(remaining_future_valid + future_test))`
5. **评估 target**：仅 `future_test`（不含 train/valid）
6. **Turn 15 末**：reveal `V_15` → behavioral update → terminal recommendation

## 未修改项

- Frontier Preserve / Boundary / Explore（`build_frontier_pool`）
- KMeans、VOI、IG、c_ask=0.35、Laplace、NONE update
- UserSimulator deterministic argmax
- Recommendation scoring、lambda_explore
- max_turns=15、Top-K=10

## 代码位置

| 组件 | 路径 |
|------|------|
| Protocol state | `src/choicerec/protocol.py` |
| Dynamic θ* | `src/choicerec/theta_builder.py` |
| Session hooks | `src/simulation/session_simulator.py`（`protocol: choicerec`） |
| History init | `src/belief/initialization.py` → `initialize_belief` |
| Weights | `src/data/dataset.py` → `weights_for_items` |
| Laplace | `src/belief/laplace_update.py` |
| Frontier | `src/ask/frontier_clustered_v32.py` → `build_frontier_pool` |
| Mock surface | `configs/choicerec/*` 强制 `llm.mode: mock`, `surface_writer.mode: mock_template` |

## Leakage 边界

- **Policy 可见**：`observed + train + revealed valid`
- **Simulator 隐藏**：`remaining future_valid + future_test`
- **永不进入 policy**：未 reveal 的 valid、全部 future_test

## Smoke sessions (LastFM)

- `10_session`
- `100_session`
- `361_session`
