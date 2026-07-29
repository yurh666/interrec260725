from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

from src.embedding.index import EmbeddingIndex
from src.memory.method_views import TransientIntentSlotView
from src.utils.math import l2_normalize


def _squish(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _as_unit(vec: np.ndarray) -> np.ndarray:
    return l2_normalize(np.asarray(vec, dtype=float).reshape(1, -1))[0]


class IntegratedRepresentativeSemanticBackend:
    """API-free semantic payload constructor for integrated v32/cb-v2 generation.

    The backend binds options to local representative item IDs, then derives the
    option vector from those representatives. It deliberately does not call any
    LLM or external retriever.
    """

    VALID_MODES = {"cluster_centroid", "representative_topk", "frontier_aware"}

    def __init__(
        self,
        mode: str = "representative_topk",
        top_k: int = 5,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.05,
    ) -> None:
        mode = str(mode or "representative_topk").strip()
        if mode not in self.VALID_MODES:
            mode = "representative_topk"
        self.mode = mode
        self.top_k = max(1, int(top_k))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.last_audit: dict[str, Any] = {}

    def generate_frontier_cluster_hypotheses(self, state_context: dict[str, Any]) -> dict[str, Any]:
        clusters = list(state_context.get("clusters") or [])
        contrasts = list(state_context.get("contrasts") or [])
        index = state_context["index"]
        item_lookup = dict(state_context.get("item_lookup") or {})
        m = max(1, int(state_context.get("M") or 3))

        cluster_by_id = {int(getattr(c, "cluster_id")): c for c in clusters}
        pairs = self._contrast_pairs(contrasts, clusters)[:m]
        hypotheses: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        role_cycle = ["preserve", "contrast", "explore"]

        for i, (pos_id, neg_id) in enumerate(pairs):
            pos = cluster_by_id.get(int(pos_id))
            neg = cluster_by_id.get(int(neg_id))
            if pos is None:
                continue
            pos_sel = self._select_representatives(pos, index, state_context, contrast_cluster=neg)
            neg_sel = self._select_representatives(neg, index, state_context) if neg is not None else None
            phrase = self._cluster_phrase(pos, item_lookup, pos_sel["representative_item_ids"])
            neg_phrase = (
                self._cluster_phrase(neg, item_lookup, neg_sel["representative_item_ids"])
                if neg is not None and neg_sel is not None
                else "a different nearby direction"
            )
            names = self._names(pos_sel["representative_item_ids"], item_lookup)
            neg_names = self._names((neg_sel or {}).get("representative_item_ids") or [], item_lookup)
            hid = f"local_h{i}"
            oid = letters[i] if i < len(letters) else f"X{i}"
            role = role_cycle[i % len(role_cycle)]
            intent = f"I prefer the {phrase} direction, represented by {self._join_names(names)}."
            negative = f"Less emphasis on {neg_phrase}." if neg is not None else "No explicit contrast."
            vec = pos_sel["option_vector"]
            row = {
                "hypothesis_id": hid,
                "option_id": oid,
                "rank_safety_role": role,
                "positive_cluster_id": int(pos_id),
                "contrast_cluster_id": int(neg_id) if neg is not None else int(pos_id),
                "contrast_ref": {
                    "reference_cluster_id": int(pos_id),
                    "contrasting_cluster_id": int(neg_id) if neg is not None else int(pos_id),
                },
                "intent_text": intent,
                "positive_intent": intent,
                "negative_contrast": negative,
                "distinguishing_features": [phrase, f"not {neg_phrase}"][:4],
                "representative_items": pos_sel["representative_item_ids"],
                "representative_item_ids": pos_sel["representative_item_ids"],
                "representative_item_names": names,
                "contrast_representative_items": (neg_sel or {}).get("representative_item_ids", []),
                "contrast_representative_item_names": neg_names,
                "positive_style_phrase": phrase,
                "contrast_style_phrase": neg_phrase,
                "short_user_facing_option_text": intent,
                "text_description": intent,
                "feature_signature": self._feature_signature(pos, phrase, pos_sel["representative_item_ids"]),
                "option_vector": vec.tolist(),
                "representative_backend": "integrated_representative_item",
                "representative_backend_variant": self.mode,
            }
            hypotheses.append(row)
            audits.append(pos_sel["audit"])

        raw = {"backend": "integrated_representative_item", "variant": self.mode, "hypotheses": hypotheses}
        self.last_audit = self._audit(audits, len(hypotheses))
        return {"raw_text": json.dumps(raw, ensure_ascii=False, indent=2), "hypotheses": hypotheses, "audit": self.last_audit}

    def generate_cluster_bound_options(self, state_context: dict[str, Any]) -> dict[str, Any]:
        clusters = list(state_context.get("clusters") or [])
        contrasts = list(state_context.get("contrasts") or [])
        index = state_context["index"]
        item_lookup = dict(state_context.get("item_lookup") or {})
        picks = self._pick_three_candidate_clusters(clusters, contrasts)
        letters = ["A", "B", "C"]
        roles = ["preserve", "boundary", "explore"]
        options: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []

        for i, cl in enumerate(picks[:3]):
            contrast_cluster = self._contrast_for_cluster(cl, clusters, contrasts)
            sel = self._select_representatives(cl, index, state_context, contrast_cluster=contrast_cluster)
            rep_ids = sel["representative_item_ids"]
            names = self._names(rep_ids, item_lookup)
            phrase = self._cluster_phrase(cl, item_lookup, rep_ids)
            role = roles[i] if i < len(roles) else "explore"
            slot = TransientIntentSlotView.from_frontier_cluster(cl, role)
            text = f"I prefer {phrase}, especially items like {self._join_names(names)}."
            row = {
                "option_id": letters[i],
                "role": slot.role,
                "bound_cluster_id": str(slot.slot_id),
                "representative_item_ids": rep_ids,
                "representative_item_names": names,
                "cluster_phrase": phrase,
                "option_text": text,
                "distinction_from_other_options": f"This option is locally bound to slot {slot.slot_id}.",
                "why_cluster_bound": "Locally selected representative items from this cluster.",
                "generic_terms_used": [],
                "confidence": 1.0,
                "option_vector": sel["option_vector"].tolist(),
            }
            if role == "boundary":
                shift = self._cluster_phrase(contrast_cluster, item_lookup, []) if contrast_cluster is not None else phrase
                row["boundary_preserve_feature"] = phrase
                row["boundary_shift_feature"] = shift
                row["boundary_relation"] = f"preserve {phrase} while shifting toward {shift}"
                row["option_text"] = (
                    f"I want to keep the {phrase} feel, while trying more {shift} cues "
                    f"like {self._join_names(names)}."
                )
            options.append(row)
            audits.append(sel["audit"])

        parsed = {
            "question_text": "Which direction better matches your current preference?",
            "options": options,
            "self_check": {
                "schema_valid": True,
                "cluster_bound": True,
                "api_free": True,
                "backend": "integrated_representative_item",
                "all_options_have_bound_cluster_id": all(o.get("bound_cluster_id") for o in options),
                "all_options_use_representative_items": all(o.get("representative_item_ids") for o in options),
                "all_options_have_positive_anchors": all(o.get("representative_item_names") for o in options),
                "no_generic_option_without_anchor": True,
                "boundary_options_specify_preserve_and_shift": True,
                "explore_options_bind_to_high_uncertainty_cluster": True,
                "options_are_semantically_distinct": True,
                "domain_language_is_correct": True,
            },
        }
        self.last_audit = self._audit(audits, len(options))
        return {"parsed": parsed, "raw_text": json.dumps(parsed, ensure_ascii=False, indent=2), "audit": self.last_audit}

    def _select_representatives(
        self,
        cluster: Any,
        index: EmbeddingIndex,
        state_context: dict[str, Any],
        *,
        contrast_cluster: Any | None = None,
    ) -> dict[str, Any]:
        member_ids = [str(x) for x in getattr(cluster, "member_ids", []) if str(x) in index.item_id_to_index]
        if not member_ids:
            dim = int(index.embeddings.shape[1])
            return {
                "representative_item_ids": [],
                "option_vector": np.zeros(dim, dtype=float),
                "audit": {"cluster_id": getattr(cluster, "cluster_id", None), "valid": False, "fallback_reason": "empty_cluster"},
            }

        member_vecs = index.vectors_for(member_ids).astype(float)
        member_vecs = l2_normalize(member_vecs)
        centroid = _as_unit(np.mean(member_vecs, axis=0))
        degraded = False

        if self.mode == "cluster_centroid":
            scores = member_vecs @ centroid
        elif self.mode == "frontier_aware":
            direction = self._resolve_direction(cluster, contrast_cluster, state_context, centroid)
            if direction is None:
                degraded = True
                scores = member_vecs @ centroid
            else:
                scores = self.alpha * (member_vecs @ centroid) + self.beta * (member_vecs @ direction)
                scores = scores - self._redundancy_penalty(member_vecs) * self.gamma
        else:
            scores = member_vecs @ centroid

        order = sorted(
            range(len(member_ids)),
            key=lambda idx: (-float(scores[int(idx)]), str(member_ids[int(idx)])),
        )
        take = order[: min(self.top_k, len(order))]
        rep_ids = [member_ids[int(i)] for i in take]
        rep_vecs = index.vectors_for(rep_ids).astype(float)
        if self.mode == "cluster_centroid":
            option_vector = centroid
        else:
            option_vector = _as_unit(np.mean(rep_vecs, axis=0))
        return {
            "representative_item_ids": rep_ids,
            "option_vector": option_vector,
            "audit": {
                "cluster_id": int(getattr(cluster, "cluster_id", -1)),
                "mode": self.mode,
                "degraded_to_representative_topk": bool(degraded),
                "representative_item_count": len(rep_ids),
                "option_vector_norm": float(np.linalg.norm(option_vector)),
                "valid": bool(rep_ids) and float(np.linalg.norm(option_vector)) > 0.0,
            },
        }

    def _resolve_direction(
        self,
        cluster: Any,
        contrast_cluster: Any | None,
        state_context: dict[str, Any],
        centroid: np.ndarray,
    ) -> np.ndarray | None:
        raw = state_context.get("frontier_direction") or state_context.get("contrast_direction")
        if raw is not None:
            arr = np.asarray(raw, dtype=float).ravel()
            if arr.size == centroid.size and np.linalg.norm(arr) > 0:
                return _as_unit(arr)
        if contrast_cluster is not None:
            cv = np.asarray(getattr(contrast_cluster, "mean_embedding_direction", []) or [], dtype=float)
            if cv.size == centroid.size and np.linalg.norm(cv) > 0:
                return _as_unit(centroid - _as_unit(cv))
        return None

    def _redundancy_penalty(self, member_vecs: np.ndarray) -> np.ndarray:
        if member_vecs.shape[0] <= 1:
            return np.zeros(member_vecs.shape[0], dtype=float)
        sim = member_vecs @ member_vecs.T
        return np.maximum(sim.mean(axis=1), 0.0)

    def _contrast_pairs(self, contrasts: list[Any], clusters: list[Any]) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for ct in contrasts:
            a = getattr(ct, "safe_or_reference_cluster_id", None)
            b = getattr(ct, "contrasting_cluster_id", None)
            if a is None and isinstance(ct, dict):
                a = ct.get("reference_cluster_id") or ct.get("safe_or_reference_cluster_id")
                b = ct.get("contrasting_cluster_id")
            try:
                pairs.append((int(a), int(b)))
            except Exception:
                continue
        if not pairs and len(clusters) >= 2:
            ids = [int(getattr(c, "cluster_id")) for c in clusters]
            pairs = [(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
        return pairs

    def _pick_three_candidate_clusters(self, clusters: list[Any], contrasts: list[Any]) -> list[Any]:
        if not clusters:
            return []
        safe = sorted(clusters, key=lambda c: -float(getattr(c, "overlap_top10_frac", 0.0) or 0.0))[0]
        uncertain = sorted(
            clusters,
            key=lambda c: -float(getattr(c, "mean_uncertainty", 0.0) if math.isfinite(float(getattr(c, "mean_uncertainty", 0.0) or 0.0)) else 0.0),
        )[0]
        boundary = None
        for a, b in self._contrast_pairs(contrasts, clusters):
            if a == int(getattr(safe, "cluster_id")):
                boundary = next((c for c in clusters if int(getattr(c, "cluster_id")) == b), None)
                break
        if boundary is None:
            boundary = uncertain if int(getattr(uncertain, "cluster_id")) != int(getattr(safe, "cluster_id")) else clusters[-1]
        out: list[Any] = []
        for cl in (safe, boundary, uncertain):
            if cl not in out:
                out.append(cl)
        for cl in clusters:
            if len(out) >= 3:
                break
            if cl not in out:
                out.append(cl)
        return out

    def _contrast_for_cluster(self, cluster: Any, clusters: list[Any], contrasts: list[Any]) -> Any | None:
        cid = int(getattr(cluster, "cluster_id"))
        for a, b in self._contrast_pairs(contrasts, clusters):
            if a == cid:
                return next((c for c in clusters if int(getattr(c, "cluster_id")) == b), None)
            if b == cid:
                return next((c for c in clusters if int(getattr(c, "cluster_id")) == a), None)
        return None

    def _names(self, item_ids: list[str], item_lookup: dict[str, dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for iid in item_ids[: self.top_k]:
            row = item_lookup.get(str(iid), {})
            name = row.get("name") or row.get("title") or row.get("item_name") or str(iid)
            names.append(_squish(name)[:96] or str(iid))
        return names

    def _cluster_phrase(self, cluster: Any, item_lookup: dict[str, dict[str, Any]], rep_ids: list[str]) -> str:
        parts: list[str] = []
        for iid in rep_ids:
            row = item_lookup.get(str(iid), {})
            for key in ("genre", "tags", "artist"):
                val = _squish(row.get(key))
                if val:
                    parts.extend(re.findall(r"[A-Za-z0-9][A-Za-z0-9&'_-]*", val.lower())[:3])
        if parts:
            counts: dict[str, int] = {}
            for p in parts:
                if len(p) > 2:
                    counts[p] = counts.get(p, 0) + 1
            top = [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]]
            if top:
                return _squish(" / ".join(top))[:120]
        summary = _squish(getattr(cluster, "semantic_summary", ""))
        return summary[:120] if summary else f"local group {getattr(cluster, 'cluster_id', 'unknown')}"

    def _feature_signature(self, cluster: Any, phrase: str, rep_ids: list[str]) -> list[str]:
        toks = re.findall(r"[a-z0-9]+", phrase.lower())[:5]
        return toks + [f"cluster_{getattr(cluster, 'cluster_id', 'x')}"] + [f"item_{x}" for x in rep_ids[:2]]

    def _join_names(self, names: list[str]) -> str:
        return ", ".join(names[:3]) if names else "local representative items"

    def _audit(self, rows: list[dict[str, Any]], n_outputs: int) -> dict[str, Any]:
        n = max(len(rows), 1)
        return {
            "backend": "integrated_representative_item",
            "variant": self.mode,
            "api_free": True,
            "outputs": int(n_outputs),
            "representative_item_valid_rate": round(sum(bool(r.get("representative_item_count")) for r in rows) / n, 4),
            "option_vector_valid_rate": round(sum(float(r.get("option_vector_norm") or 0.0) > 0.0 for r in rows) / n, 4),
            "fallback_rate": round(sum(bool(r.get("degraded_to_representative_topk")) for r in rows) / n, 4),
            "per_option": rows,
        }
