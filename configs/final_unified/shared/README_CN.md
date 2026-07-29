# `final_unified/shared` 说明（interrec260725）

## 当前执行路径

`scripts/run_final_interrec.py` 使用的配置为：

`configs/final_unified/runs/{book|movielens|lastfm}_seed42.yaml`

每条 **仅** `inherits: ../../runs/{dataset}_full_integrated_distill_surface_seed42.yaml`，**不**加载本目录下的 archived protocol。

## `protocol_seed42_ARCHIVED_20260722.yaml`

该文件是 2026-07-22「Unified + HRV gate on + `use_mock_llm: false`」的设计稿，**从未接入 interrec260725 distill full 跑数链**。保留仅供历史对照；**勿**在文档中将其描述为当前 canonical resolved 配置。

interrec260725 主表实测（`load_config` resolved）：

| 键 | Book | MovieLens | LastFM |
|----|------|-----------|--------|
| `ask.voi_gate.enabled` | false | false | false |
| `ask.voi_gate.preserve_ask_decision` | （未设置） | （未设置） | （未设置） |
| `...frontier_clustered_v32.use_mock_llm` | true | true | true |

详见 `docs/FINAL_UNIFIED_INTERREC_CODEBOOK_CN.md`（260725 修订）与 `PIPELINE_CN.md` §2。
