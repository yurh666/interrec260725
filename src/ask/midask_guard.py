"""Mid-ask guard: conservative post-decision policy on should_ask (VOI gate).

Runtime uses only online signals (belief summaries, rec scores, overlap diagnostics).
"""

from __future__ import annotations

from typing import Any


def low_confidence_midask(
    voi: float,
    *,
    belief_entropy: float,
    sigma_trace: float | None,
    margin: float | None,
    overlap: float,
    cfg: dict[str, Any],
) -> tuple[bool, list[str]]:
    """AND-combine enabled sub-conditions on top of the VOI window (if enabled).

    Returns (passed, list of active condition names for logging).
    """
    reasons: list[str] = []
    ok = True

    if bool(cfg.get("voi_window_enabled", True)):
        vl = float(cfg.get("voi_lower", -0.05))
        vu = float(cfg.get("voi_upper", 0.0))
        window_ok = vl <= float(voi) <= vu
        ok = ok and window_ok
        if window_ok:
            reasons.append("voi_window")

    if bool(cfg.get("entropy_enabled", False)):
        ent_thr = float(cfg.get("belief_entropy_min", 2.0))
        ent_ok = float(belief_entropy) >= ent_thr
        ok = ok and ent_ok
        if ent_ok:
            reasons.append("belief_entropy_high")

    if bool(cfg.get("sigma_trace_enabled", False)) and sigma_trace is not None:
        st_thr = float(cfg.get("sigma_trace_min", 0.0))
        st_ok = float(sigma_trace) >= st_thr
        ok = ok and st_ok
        if st_ok:
            reasons.append("sigma_trace_high")

    if bool(cfg.get("margin_enabled", False)) and margin is not None:
        mthr = float(cfg.get("margin_max", 0.05))
        m_ok = float(margin) <= mthr
        ok = ok and m_ok
        if m_ok:
            reasons.append("top1_top10_margin_low")

    if bool(cfg.get("overlap_enabled", False)):
        omx = float(cfg.get("overlap_min", 0.1))
        o_ok = float(overlap) >= omx
        ok = ok and o_ok
        if o_ok:
            reasons.append("option_overlap_high")

    return ok, reasons


def should_apply_midask_force(
    *,
    voi_eval: float,
    best_ids: list[str],
    natural_do_ask: bool,
    cfg: dict[str, Any],
    belief_entropy: float,
    sigma_trace: float | None,
    margin: float | None,
    overlap: float,
) -> tuple[bool, list[str]]:
    """Decide whether to override no-ask → ask for low-confidence mid-ask regime."""
    if not best_ids:
        return False, []

    only_false = bool(cfg.get("only_when_should_ask_false", True))
    if only_false and natural_do_ask:
        return False, []

    low_ok, rs = low_confidence_midask(
        voi_eval,
        belief_entropy=belief_entropy,
        sigma_trace=sigma_trace,
        margin=margin,
        overlap=overlap,
        cfg=cfg,
    )
    if not low_ok:
        return False, rs

    return True, rs
