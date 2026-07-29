#!/usr/bin/env python3
"""Run 1-session distill smoke per dataset; audit surface_writer (no mock fallback)."""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ("book", "smoke_book_distill_n1_verify_lora.yaml"),
    ("movielens", "smoke_movielens_distill_n1_verify_lora.yaml"),
    ("lastfm", "smoke_lastfm_distill_n1_verify_lora.yaml"),
]


def audit_log(log_path: Path) -> dict:
    sw = Counter()
    fb = Counter()
    episodes = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        episodes.add(o.get("episode_id"))
        im = o.get("implementation_modes") or {}
        sw[str(im.get("surface_writer"))] += 1
        bridge = (o.get("llm_semantic_bridge") or {}).get("surface_writer") or {}
        if isinstance(bridge, dict):
            fb[str(bridge.get("fallback_reason"))] += 1
    return {
        "episodes": len(episodes),
        "surface_writer_impl": dict(sw),
        "fallback_reasons": dict(fb),
    }


def main() -> int:
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)}
    results = {}
    ok_all = True
    for ds, cfg in DATASETS:
        print(f"=== run {ds} ===", flush=True)
        cp = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_main_experiment.py"), "--config", f"configs/runs/{cfg}"],
            cwd=ROOT,
            env=env,
        )
        if cp.returncode != 0:
            results[ds] = {"run": "fail", "rc": cp.returncode}
            ok_all = False
            continue
        runs = sorted((ROOT / "experiments/runs").glob(f"*_{ds}_interrec_seed42"), key=lambda p: p.stat().st_mtime)
        if not runs:
            results[ds] = {"run": "fail", "error": "no run dir"}
            ok_all = False
            continue
        run_dir = runs[-1]
        log = run_dir / "full_log.jsonl"
        audit = audit_log(log)
        metrics = json.loads((run_dir / "run_summary.json").read_text()) if (run_dir / "run_summary.json").exists() else {}
        bad_fb = [k for k in audit["fallback_reasons"] if k not in ("None", "null", None) and k and "local_model_load_failed" in k]
        lora_ok = audit["surface_writer_impl"].get("local_small_model", 0) > 0 and not bad_fb
        results[ds] = {"run_dir": str(run_dir.name), "audit": audit, "lora_surface_ok": lora_ok, "metrics": metrics.get("aggregate") or metrics}
        if not lora_ok:
            ok_all = False
        print(json.dumps(results[ds], indent=2, ensure_ascii=False), flush=True)

    out = ROOT / "LORA_SURFACE_VERIFICATION.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] wrote {out} ok_all={ok_all}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
