#!/usr/bin/env python3
"""Validate ChoiceRec LastFM smoke invariants and emit CHOICEREC_LASTFM_SMOKE_REPORT_CN.md."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.in_depth.scripts.common import load_full_log_episodes, load_sessions  # noqa: E402
from src.choicerec.protocol import ChoiceRecProtocol, build_valid_chunks  # noqa: E402
from src.choicerec.theta_builder import build_choicerec_theta_star  # noqa: E402
from src.embedding.index import EmbeddingIndex  # noqa: E402
from src.data.dataset import InterRecDataset  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _load_index(cfg: dict) -> EmbeddingIndex:
    ds_cfg = cfg["dataset"]
    pdir = PROJECT_ROOT / ds_cfg["processed_dir"]
    emb_cfg = cfg.get("embedding") or {}
    pre = emb_cfg.get("precomputed_path")
    emb_filename = Path(pre).name if pre else "item_embeddings.npy"
    return EmbeddingIndex.load(pdir, embedding_filename=emb_filename)


def check_session(
    sid: str,
    turns: list[dict],
    session: dict,
    dataset: InterRecDataset,
    index: EmbeddingIndex,
    belief_cfg: dict,
) -> dict:
    proto = ChoiceRecProtocol.from_session(session, dataset, index, belief_cfg, max_turns=15)
    results: dict = {"session_id": sid, "checks": {}, "pass": True}

    def fail(name: str, msg: str) -> None:
        results["checks"][name] = {"pass": False, "detail": msg}
        results["pass"] = False

    def ok(name: str, detail: str = "") -> None:
        results["checks"][name] = {"pass": True, "detail": detail}

    init_visible = list(proto.visible_behavior_history)
    expected_init = [str(x) for x in session["observed_history"]] + [str(x) for x in session.get("future_train", [])]
    if init_visible == expected_init:
        ok("initial_history", f"count={len(init_visible)}")
    else:
        fail("initial_history", f"got {len(init_visible)} expected {len(expected_init)}")

    chunks = build_valid_chunks([str(x) for x in session.get("future_valid", [])], 15)
    if chunks == proto.valid_chunks:
        ok("chunk_allocation", str([len(c) for c in chunks]))
    else:
        fail("chunk_allocation", f"expected sizes {[len(c) for c in chunks]}")

    revealed_order: list[str] = []
    choice_updates = 0
    leakage_ok = True
    for t in sorted(turns, key=lambda x: int(x["turn"])):
        cr = t.get("choicerec") or {}
        chunk = cr.get("valid_chunk_revealed_after_turn") or []

        rec_items = list((t.get("recommendation") or {}).get("recommended_items") or [])
        unrevealed = set(str(x) for x in session.get("future_valid", [])) - set(revealed_order)
        hidden_test = set(str(x) for x in session.get("future_test", []))
        leaked = [i for i in rec_items if i in unrevealed]
        if leaked:
            fail("leakage", f"turn {t['turn']} leaked items {leaked[:5]}")
            leakage_ok = False

        if chunk:
            revealed_order.extend([str(x) for x in chunk])
        if t.get("belief_update_executed"):
            choice_updates += 1
        if chunk and not cr.get("sigma_unchanged", True) and cr.get("behavior_update_applied"):
            fail("covariance_invariance", f"turn {t['turn']} Sigma changed on behavior reveal")

        remaining_before = cr.get("remaining_valid_before_turn") or []
        theta_logged = float(cr.get("simulator_theta_norm", 0))
        theta_info = build_choicerec_theta_star(
            remaining_future_valid=[str(x) for x in remaining_before],
            future_test=[str(x) for x in session.get("future_test", [])],
            index=index,
            session=session,
            dataset=dataset,
        )
        theta_norm = float(np.linalg.norm(theta_info["theta_star"]))
        if not (abs(theta_norm - theta_logged) < 1e-5 or (theta_norm < 1e-9 and theta_logged < 1e-9)):
            fail("dynamic_theta_star", f"turn {t['turn']} logged={theta_logged:.6f} recomputed={theta_norm:.6f}")

    if leakage_ok:
        ok("leakage", "no unrevealed future_valid in recommendation lists")

    if revealed_order == [str(x) for x in session.get("future_valid", [])]:
        ok("valid_order", f"revealed {len(revealed_order)}")
    elif not session.get("future_valid"):
        ok("valid_order", "no future_valid")
    else:
        fail("valid_order", f"order mismatch revealed={revealed_order[:5]}...")

    ow = []
    for t in turns:
        bridge = t.get("llm_semantic_bridge") or {}
        ow_audit = bridge.get("option_writer") or {}
        if ow_audit.get("n_api_calls"):
            fail("mock_surface", f"API calls turn {t['turn']}")
        ow.append(ow_audit.get("implementation_mode"))
    ok("mock_surface", f"modes={set(x for x in ow if x)}")

    final = turns[-1].get("final_recommendation_corrected") or {}
    results["summary"] = {
        "observed_count": len(session.get("observed_history", [])),
        "train_count": len(session.get("future_train", [])),
        "valid_count": len(session.get("future_valid", [])),
        "test_count": len(session.get("future_test", [])),
        "chunk_schedule": [len(c) for c in chunks],
        "ask_turns": sum(1 for t in turns if t.get("do_ask")),
        "choice_updates": choice_updates,
        "behavior_updates": sum(1 for t in turns if (t.get("choicerec") or {}).get("behavior_update_applied")),
        "final_hit10": final.get("HitRate@10"),
        "final_source": final.get("metric_source"),
    }
    return results


def main() -> None:
    run_dir = PROJECT_ROOT / "experiments/runs/choicerec_lastfm_smoke3_seed42"
    log_path = run_dir / "full_log.jsonl"
    if not log_path.exists():
        print(f"[error] missing {log_path}; run smoke first")
        sys.exit(1)

    cfg = load_config(str(PROJECT_ROOT / "configs/choicerec/lastfm_smoke3_seed42.yaml"))
    dataset = InterRecDataset.load(PROJECT_ROOT / cfg["dataset"]["processed_dir"])
    index = _load_index(cfg)
    belief_cfg = cfg.get("belief") or {}
    sessions = {str(s.get("episode_id") or s["user_id"]): s for s in load_sessions("lastfm")}
    episodes = load_full_log_episodes(log_path)

    all_results = []
    for sid, turns in episodes.items():
        if sid not in sessions:
            continue
        all_results.append(check_session(sid, turns, sessions[sid], dataset, index, belief_cfg))

    overall = all(r["pass"] for r in all_results)
    lines = [
        "# ChoiceRec LastFM Smoke Report",
        "",
        f"生成时间：{datetime.now(timezone.utc).astimezone().isoformat()}",
        "",
        f"Run: `{run_dir}`",
        "",
        f"## Overall: **{'PASS' if overall else 'FAIL'}**",
        "",
    ]
    for r in all_results:
        lines.append(f"### {r['session_id']} — {'PASS' if r['pass'] else 'FAIL'}")
        s = r.get("summary") or {}
        lines.append(
            f"- observed/train/valid/test: {s.get('observed_count')}/{s.get('train_count')}/"
            f"{s.get('valid_count')}/{s.get('test_count')}"
        )
        lines.append(f"- chunk schedule: {s.get('chunk_schedule')}")
        lines.append(f"- ask turns: {s.get('ask_turns')}, behavior updates: {s.get('behavior_updates')}")
        lines.append(f"- final Hit@10: {s.get('final_hit10')} ({s.get('final_source')})")
        for name, chk in r.get("checks", {}).items():
            if not name.startswith("theta_star_turn"):
                lines.append(f"- [{('OK' if chk['pass'] else 'FAIL')}] {name}: {chk.get('detail','')}")
        lines.append("")

    out = run_dir / "CHOICEREC_LASTFM_SMOKE_REPORT_CN.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] overall={'PASS' if overall else 'FAIL'} report={out}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
