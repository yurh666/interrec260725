from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any

from src.agents.llm_cache import LLMRawResponseCache, payload_hex_key
from src.agents.stable_cache_key import prompt_hash, replay_cache_lookup_variants
from src.utils.run_profiling_context import (
    get_active_llm_component,
    profiling_get_ctx,
    set_active_llm_component,
)

_LATENCY_IO_LOCK = threading.Lock()


def _append_latency_profile_row(rec: dict[str, Any]) -> None:
    """Append one JSON line if INTERREC_LATENCY_PROFILE_PATH is set (profiling-only)."""

    path = os.environ.get("INTERREC_LATENCY_PROFILE_PATH")
    if not path:
        return
    try:
        merged = {**profiling_get_ctx(), **rec}
        line = json.dumps(merged, ensure_ascii=False) + "\n"
        with _LATENCY_IO_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        pass


def _json_nonempty(parsed: Any) -> bool:
    if parsed is None:
        return False
    if isinstance(parsed, dict):
        return len(parsed) > 0
    if isinstance(parsed, list):
        return len(parsed) > 0
    return True


class LLMCacheReplayMissError(RuntimeError):
    """Raised when cache-only replay cannot find a precommitted response."""


def _is_timeout_exc(exc: BaseException) -> bool:
    cls = type(exc).__qualname__
    mod = type(exc).__module__ or ""
    if cls == "TimeoutError":
        return True
    if "Timeout" in cls and ("httpcore" in mod or "httpx" in mod or "openai" in mod):
        return True
    if cls == "ReadTimeout" or cls == "WriteTimeout" or cls == "ConnectTimeout":
        return True
    return False


class LLMClient:
    """OpenAI-compatible or Anthropic client with structured JSON parsing, auditing, optional response cache."""

    def __init__(
        self,
        mode: str = "mock",
        provider: str = "none",
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        log_prompts: bool = True,
        log_responses: bool = True,
        max_retry: int = 3,
        *,
        cache_enabled: bool = True,
        cache_dir: str | None = None,
        max_structured_parse_retries: int = 3,
        cache_memory_max_entries: int = 0,
        api_key: str | None = None,
        cache_mode: str = "normal",
        cache_fail_fast_on_miss: bool = False,
        use_stable_cache_keys: bool = False,
        replay_lookup_aliases: bool = False,
        run_seed: int | None = None,
    ) -> None:
        self.mode = mode
        self.provider = provider
        self.base_url = base_url
        self.timeout = float(timeout)
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self.max_retry = max_retry
        self.max_structured_parse_retries = max(1, int(max_structured_parse_retries))
        self.implementation_mode = "mock_llm" if mode == "mock" else f"real_llm_{provider}"
        self._api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self._cache_memory_max_entries = max(0, int(cache_memory_max_entries))
        self._structured_response_memo: OrderedDict[str, str] = OrderedDict()

        if model:
            self.model = model
        elif provider == "anthropic":
            self.model = "claude-3-5-haiku-20241022"
        else:
            self.model = "gpt-4o-mini"

        self._client: Any = None
        if mode != "mock":
            self._init_real_client()

        cdir = cache_dir if cache_dir is not None else ".cache/interrec_llm"
        self._response_cache = LLMRawResponseCache(cdir, enabled=bool(cache_enabled) and mode != "mock")
        self.cache_mode = str(cache_mode or "normal")
        self.cache_fail_fast_on_miss = bool(cache_fail_fast_on_miss)
        self.use_stable_cache_keys = bool(use_stable_cache_keys)
        self.replay_lookup_aliases = bool(replay_lookup_aliases)
        self.run_seed = int(run_seed) if run_seed is not None else None

        self.last_structured_audit: dict[str, Any] = {}
        self._last_token_usage: dict[str, Any] | None = None

    def _memo_key(self, namespace: str, payload: Any) -> str:
        return f"{namespace}:{payload_hex_key(namespace, payload)}"

    def _structured_memo_get(self, key: str) -> str | None:
        if self._cache_memory_max_entries <= 0:
            return None
        raw = self._structured_response_memo.get(key)
        if raw is None:
            return None
        self._structured_response_memo.move_to_end(key)
        return raw

    def _structured_memo_put(self, key: str, raw: str) -> None:
        if self._cache_memory_max_entries <= 0:
            return
        self._structured_response_memo[key] = raw
        self._structured_response_memo.move_to_end(key)
        while len(self._structured_response_memo) > self._cache_memory_max_entries:
            self._structured_response_memo.popitem(last=False)

    def _init_real_client(self) -> None:
        if self.provider == "openai":
            import openai  # type: ignore

            api_key = self._api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI-compatible calls require llm.api_key in config, or DEEPSEEK_API_KEY / OPENAI_API_KEY in the environment."
                )
            kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**kwargs)
        elif self.provider == "anthropic":
            import anthropic  # type: ignore

            self._client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def chat(self, messages: list[dict[str, str]], model: str | None = None, **kwargs: Any) -> str:
        if self.mode == "mock":
            return self._mock_response(messages)
        m = model or self.model
        comp = get_active_llm_component() or "chat"
        path_on = bool(os.environ.get("INTERREC_LATENCY_PROFILE_PATH"))
        last_exc: Exception | None = None
        for attempt in range(self.max_retry):
            t0 = time.perf_counter()
            try:
                out = self._real_chat(messages, m, **kwargs)
                api_lat = time.perf_counter() - t0
                try:
                    from src.utils.session_guard_state import guard_note_llm_http

                    guard_note_llm_http(success=True, timeout=False)
                except Exception:
                    pass
                if path_on:
                    tok = self._last_token_usage or {}
                    _append_latency_profile_row(
                        {
                            "event": "llm_http",
                            "component": comp,
                            "latency_sec": round(api_lat, 6),
                            "http_attempt": attempt,
                            "http_success": True,
                            "http_called": True,
                            "request_timeout": False,
                            "cache_hit": False,
                            "retry_count": attempt,
                            "n_api_calls": 1,
                            "prompt_tokens": tok.get("prompt_tokens"),
                            "completion_tokens": tok.get("completion_tokens"),
                            "total_tokens": tok.get("total_tokens"),
                        }
                    )
                return out
            except Exception as exc:
                api_lat = time.perf_counter() - t0
                last_exc = exc
                is_to = _is_timeout_exc(exc)
                try:
                    from src.utils.session_guard_state import guard_note_llm_http

                    guard_note_llm_http(success=False, timeout=is_to)
                except Exception:
                    pass
                if path_on:
                    _append_latency_profile_row(
                        {
                            "event": "llm_http",
                            "component": comp,
                            "latency_sec": round(api_lat, 6),
                            "http_attempt": attempt,
                            "http_success": False,
                            "http_called": True,
                            "request_timeout": is_to,
                            "cache_hit": False,
                            "retry_count": attempt,
                            "n_api_calls": 0,
                            "error_type": type(exc).__name__,
                        }
                    )
                if attempt == self.max_retry - 1:
                    raise
                time.sleep(2**attempt)
        if last_exc:
            raise last_exc
        return ""

    def _real_chat(self, messages: list[dict[str, str]], model: str, **kwargs: Any) -> str:
        self._last_token_usage = None
        if self.provider == "openai":
            resp = self._client.chat.completions.create(model=model, messages=messages, **kwargs)
            if getattr(resp, "usage", None) is not None:
                u = resp.usage
                self._last_token_usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", None),
                    "completion_tokens": getattr(u, "completion_tokens", None),
                    "total_tokens": getattr(u, "total_tokens", None),
                }
            return resp.choices[0].message.content or ""
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[m for m in messages if m["role"] != "system"],
                system=next((m["content"] for m in messages if m["role"] == "system"), ""),
                **kwargs,
            )
            return resp.content[0].text
        raise NotImplementedError

    def _mock_response(self, messages: list[dict[str, str]]) -> str:
        last = messages[-1]["content"] if messages else ""
        if "direction" in last.lower():
            return json.dumps(
                [
                    {
                        "direction_id": 0,
                        "direction_name": "mock_direction_0",
                        "positive_side": "upbeat energetic tracks",
                        "negative_side": "slow calm music",
                        "is_meaningful": True,
                    }
                ]
            )
        sys0 = messages[0]["content"] if messages else ""
        if "INTERREC_CONTRASTIVE_V2_SCHEMA" in sys0:
            return json.dumps(
                [
                    {
                        "option_id": "opt1",
                        "positive_intent": "The user wants high-energy synth-pop suited to workouts.",
                        "negative_contrast": "Not sparse acoustic ballads or slow singer-songwriter material.",
                        "distinguishing_features": ["synth-pop", "danceable", "female vocals", "2010s"],
                        "feature_signature": ["synth", "pop", "energetic", "dance", "workout"],
                        "positive_retrieval_query": "synth pop energetic dance workout female vocals",
                        "negative_retrieval_query": "acoustic ballad slow sparse singer-songwriter",
                        "representative_diversity_hint": "Pick distinct artists within upbeat pop.",
                        "contrast_text": "danceable pop vs quiet acoustic",
                        "short_user_facing_option_text": "More interested in upbeat electronic-leaning pop than quiet acoustic tracks.",
                    },
                    {
                        "option_id": "opt2",
                        "positive_intent": "The user wants mellow indie rock for focused listening.",
                        "negative_contrast": "Not aggressive metal or high-BPM club music.",
                        "distinguishing_features": ["indie", "guitar", "mid-tempo", "melodic"],
                        "feature_signature": ["indie", "rock", "mellow", "guitar", "study"],
                        "positive_retrieval_query": "indie rock melodic guitar mellow study",
                        "negative_retrieval_query": "metal aggressive high bpm club EDM",
                        "representative_diversity_hint": "Vary guitar-driven indie acts.",
                        "contrast_text": "melodic indie vs heavy club",
                        "short_user_facing_option_text": "More aligned with relaxed guitar indie than heavy or club-oriented tracks.",
                    },
                    {
                        "option_id": "opt3",
                        "positive_intent": "The user wants Latin pop crossover with rhythmic percussion.",
                        "negative_contrast": "Not minimal techno or ambient drone soundscapes.",
                        "distinguishing_features": ["latin", "percussion", "pop", "vocal hooks"],
                        "feature_signature": ["latin", "pop", "rhythm", "percussion", "crossover"],
                        "positive_retrieval_query": "latin pop percussion rhythmic radio hooks",
                        "negative_retrieval_query": "minimal techno ambient drone",
                        "representative_diversity_hint": "Mix regional latin styles.",
                        "contrast_text": "Latin rhythm vs minimal techno",
                        "short_user_facing_option_text": "More drawn to Latin-flavored rhythmic pop than minimal electronic textures.",
                    },
                    {
                        "option_id": "opt4",
                        "positive_intent": "The user wants singer-songwriter folk with storytelling lyrics.",
                        "negative_contrast": "Not highly processed hyperpop or glitch-heavy production.",
                        "distinguishing_features": ["folk", "acoustic", "storytelling", "intimate"],
                        "feature_signature": ["folk", "acoustic", "songwriter", "story", "intimate"],
                        "positive_retrieval_query": "folk singer songwriter acoustic storytelling intimate",
                        "negative_retrieval_query": "hyperpop glitch heavy autotune electronic",
                        "representative_diversity_hint": "Diversify eras of folk-pop artists.",
                        "contrast_text": "storytelling folk vs hyperpop",
                        "short_user_facing_option_text": "More interested in intimate acoustic storytelling than heavily processed electronic pop.",
                    },
                    {
                        "option_id": "opt5",
                        "positive_intent": "The user wants 90s alternative rock with crunchy guitars.",
                        "negative_contrast": "Not modern trap beats or glossy K-pop production.",
                        "distinguishing_features": ["90s", "alternative", "grunge-adjacent", "guitars"],
                        "feature_signature": ["90s", "alternative", "rock", "guitar", "grunge"],
                        "positive_retrieval_query": "90s alternative rock guitar grunge radio",
                        "negative_retrieval_query": "trap hip hop kpop glossy production",
                        "representative_diversity_hint": "Sample different alt-rock scenes.",
                        "contrast_text": "90s alt-rock vs modern trap/K-pop",
                        "short_user_facing_option_text": "More aligned with guitar-driven 90s alternative than trap or glossy modern pop production.",
                    },
                ]
            )
        if "hypothesis" in last.lower():
            return json.dumps(
                [
                    {
                        "hypothesis_id": "h1",
                        "text_description": "User enjoys upbeat pop music for workouts.",
                        "feature_signature": ["energetic", "pop", "workout"],
                        "rationale": "Mock: direction 0 positive side.",
                    },
                    {
                        "hypothesis_id": "h2",
                        "text_description": "User prefers relaxing indie music for studying.",
                        "feature_signature": ["indie", "calm", "study"],
                        "rationale": "Mock: direction 0 negative side.",
                    },
                ]
            )
        return json.dumps({"result": "mock", "content": "No specific mock rule matched."})

    def structured_json(
        self,
        messages: list[dict[str, str]],
        *,
        cache_namespace: str | None = None,
        cache_payload: Any | None = None,
        max_parse_retries: int | None = None,
        profile_component: str | None = None,
        **chat_kwargs: Any,
    ) -> Any:
        """Parse assistant JSON with optional deterministic disk cache.

        Sets ``last_structured_audit``: cache_hit/cache_miss, parse_success,
        retry_count, raw_storage_path (cache file path if saved).

        ``profile_component`` + env ``INTERREC_LATENCY_PROFILE_PATH`` enable
        latency/token profiling (no effect on parsing outcomes when unset).
        """

        chat_kwargs.setdefault("temperature", 0.0)
        retries = (
            max_parse_retries if max_parse_retries is not None else self.max_structured_parse_retries
        )
        retries = max(1, int(retries))

        comp = profile_component or cache_namespace or "structured_json"
        tok0 = get_active_llm_component()
        set_active_llm_component(comp)
        try:
            audit: dict[str, Any] = {
                "component": comp,
                "cache_component": comp,
                "cache_hit": False,
                "cache_miss": False,
                "cache_key_sha256": None,
                "cache_lookup_alias": None,
                "parse_success": False,
                "retry_count": 0,
                "raw_response_saved": False,
                "raw_storage_path": None,
                "fallback_reason": None,
                "n_api_calls": 0,
                "run_seed": self.run_seed,
            }
            self.last_structured_audit = audit

            if self.mode == "mock":
                raw = self._mock_response(messages)
                parsed = parse_llm_json(raw)
                audit["parse_success"] = _json_nonempty(parsed)
                audit["fallback_reason"] = None if audit["parse_success"] else "mock_parse_empty"
                return parsed

            use_cache = cache_namespace is not None and cache_payload is not None
            cached_raw: str | None = None
            structured_cache_source: str | None = None
            memo_k: str | None = None
            audit["cache_mode"] = self.cache_mode
            audit["cache_key_version"] = (
                cache_payload.get("cache_key_version")
                if isinstance(cache_payload, dict)
                else None
            )
            audit["payload_hash"] = (
                payload_hex_key(cache_namespace, cache_payload) if use_cache else None
            )
            audit["prompt_hash"] = prompt_hash(messages) if messages else None

            if (
                use_cache
                and self.cache_mode == "trace_replay"
                and cache_namespace is not None
            ):
                audit["cache_miss"] = True
                audit["cache_miss_reason"] = "trace_replay_blocks_live_llm"
                audit["fallback_reason"] = "trace_replay_blocks_live_llm"
                raise LLMCacheReplayMissError(
                    f"trace_replay blocked namespace={cache_namespace}"
                )

            if use_cache:
                hk = payload_hex_key(cache_namespace, cache_payload)
                audit["cache_key_sha256"] = hk
                memo_k = self._memo_key(cache_namespace, cache_payload)
                memo_hit = self._structured_memo_get(memo_k)
                if memo_hit is not None:
                    cached_raw = memo_hit
                    audit["cache_hit"] = True
                    structured_cache_source = "memory"
                else:
                    cached_raw, hk_disk = self._response_cache.get(cache_namespace, cache_payload)
                    if hk_disk:
                        audit["cache_key_sha256"] = hk_disk
                    if cached_raw is None:
                        audit["cache_miss"] = True
                        audit["cache_miss_reason"] = "disk_and_memory_miss"
                    else:
                        audit["cache_hit"] = True
                        structured_cache_source = "disk"

                    if (
                        cached_raw is None
                        and self.replay_lookup_aliases
                        and self.cache_mode
                        in ("replay_strict", "cache_only", "cache_fill", "trace_cache_fill")
                    ):
                        for tag, variant, hint in replay_cache_lookup_variants(
                            cache_namespace, cache_payload
                        ):
                            if tag == "primary":
                                continue
                            alias_raw, alias_hk = self._response_cache.get(
                                cache_namespace, variant
                            )
                            if alias_raw is not None:
                                cached_raw = alias_raw
                                audit["cache_hit"] = True
                                audit["cache_miss"] = False
                                audit["cache_lookup_alias"] = tag
                                audit["cache_miss_reason"] = hint or "resolved_via_alias"
                                if alias_hk:
                                    audit["cache_key_sha256"] = alias_hk
                                structured_cache_source = "disk_alias"
                                memo_k = self._memo_key(cache_namespace, variant)
                                break

                if (
                    cached_raw is None
                    and self.cache_fail_fast_on_miss
                    and self.cache_mode in ("replay_strict", "cache_only")
                ):
                    audit["fallback_reason"] = "cache_replay_miss_fail_fast"
                    raise LLMCacheReplayMissError(
                        f"cache miss namespace={cache_namespace} sha256={audit.get('cache_key_sha256', '')[:16]}"
                    )

            parsed: Any = {}

            for attempt in range(retries):
                audit["retry_count"] = attempt
                if cached_raw is not None and attempt == 0:
                    raw = cached_raw
                    api_lat = 0.0
                    tok = None
                    try:
                        from src.utils.session_guard_state import guard_note_structured_cache

                        guard_note_structured_cache(True)
                    except Exception:
                        pass
                    _append_latency_profile_row(
                        {
                            "event": "structured_parse",
                            "ts": time.time(),
                            "component": comp,
                            "latency_sec": api_lat,
                            "cache_hit": True,
                            "cache_key_sha256": audit.get("cache_key_sha256"),
                            "structured_cache_source": structured_cache_source,
                            "http_called": False,
                            "request_timeout": False,
                            "fallback_used": False,
                            "retry_count": attempt,
                            "n_api_calls": 0,
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                            "parse_round": "cache_hit_first",
                        }
                    )
                else:
                    t0 = time.perf_counter()
                    raw = self.chat(messages, **chat_kwargs)
                    api_lat = time.perf_counter() - t0
                    audit["n_api_calls"] += 1
                    tok = self._last_token_usage
                    try:
                        from src.utils.session_guard_state import guard_note_structured_cache

                        guard_note_structured_cache(False)
                    except Exception:
                        pass
                    _append_latency_profile_row(
                        {
                            "event": "structured_parse",
                            "ts": time.time(),
                            "component": comp,
                            "latency_sec": round(api_lat, 6),
                            "cache_hit": False,
                            "cache_key_sha256": audit.get("cache_key_sha256"),
                            "structured_cache_source": None,
                            "http_called": True,
                            "request_timeout": False,
                            "fallback_used": False,
                            "retry_count": attempt,
                            "n_api_calls": 1,
                            "prompt_tokens": (tok or {}).get("prompt_tokens"),
                            "completion_tokens": (tok or {}).get("completion_tokens"),
                            "total_tokens": (tok or {}).get("total_tokens"),
                            "parse_round": "live_chat",
                        }
                    )

                parsed = parse_llm_json(raw)
                if _json_nonempty(parsed):
                    audit["parse_success"] = True
                    audit["fallback_reason"] = None
                    if use_cache and audit.get("cache_hit") and attempt == 0:
                        print(
                            f"[llm_cache] hit namespace={cache_namespace} "
                            f"sha256={audit.get('cache_key_sha256', '')[:16]}...",
                            flush=True,
                        )
                    if use_cache and audit["n_api_calls"] > 0:
                        p = self._response_cache.put(cache_namespace, cache_payload, raw)
                        if p is not None:
                            audit["raw_response_saved"] = True
                            audit["raw_storage_path"] = str(p)
                        if memo_k is not None:
                            self._structured_memo_put(memo_k, raw)
                    return parsed

                audit["fallback_reason"] = "json_parse_failed_or_empty"
                if cached_raw is not None and attempt == 0:
                    cached_raw = None
                    audit["cache_miss"] = True
                    audit["cache_hit"] = False
                    audit["fallback_reason"] = "cache_hit_unparseable_retry_live"

            audit["fallback_reason"] = audit.get("fallback_reason") or "exhausted_parse_retries"
            return parsed
        finally:
            set_active_llm_component(tok0)


def parse_llm_json(raw: str) -> Any:
    """Parse JSON from chat completion text (may include markdown fences). Never raises."""

    def strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            nl = t.find("\n")
            if nl != -1:
                t = t[nl + 1 :].strip()
            else:
                t = t.strip("`").removeprefix("json").strip()
            if t.endswith("```"):
                t = t[: -3].strip()
        return t.strip()

    def balanced_slice(text: str, open_ch: str, close_ch: str) -> str | None:
        start = text.find(open_ch)
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        quote = ""
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if in_string:
                if ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                continue
            if ch in "\"'":
                in_string = True
                quote = ch
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    snippet = strip_fences(raw)
    candidates: list[str] = [snippet]

    ba = balanced_slice(snippet, "[", "]")
    if ba:
        candidates.append(ba)

    jo = balanced_slice(snippet, "{", "}")
    if jo:
        candidates.append(jo)

    seen: set[str] = set()
    for s in candidates:
        if not s or s in seen:
            continue
        seen.add(s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            continue

    return {}
