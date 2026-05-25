"""Prompt evolver (Phase 7 of the agentic upgrade).

Per task type (domain), Hermes tracks the currently-active system-prompt
variant + its rolling success rate (from Phase 3 reflection signals).
When the active variant trails a proposed alternative after sufficient
samples, the evolver demotes the active and promotes the candidate.

Variants live in the vault, tagged `prompt:<task_type>` with status
metadata: `active`, `candidate`, `archived`. Outcomes are tagged
`prompt_outcome`. The cron (n8n) is responsible for calling
`maybe_propose()` and `maybe_promote()` daily.

This ships DISABLED by default (HERMES_PROMPT_EVOLVER_ENABLED=false) and
should be opted into after we've confirmed reflection signals are reliable
and the auxiliary model proposes sensible refinements.

API used by api_server: `current_prompt(domain)` — returns the active
prompt for a domain, or empty string to use the built-in default.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env gate + cache
# ---------------------------------------------------------------------------

def _enabled() -> bool:
    return os.environ.get("HERMES_PROMPT_EVOLVER_ENABLED", "false").lower() in ("1", "true", "yes", "on")


_CACHE: Dict[str, Dict[str, Any]] = {}  # domain -> {prompt, prompt_id, fetched_at}
_CACHE_TTL = 300.0
_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Vault access — uses obsidian-mind client preferentially (cheap + reliable)
# ---------------------------------------------------------------------------

def _mind_client():
    try:
        from plugins.memory.vault.obsidian_mind import from_env as _from_env
        return _from_env()
    except Exception:
        return None


def _vault_client():
    try:
        from plugins.memory.vault.client import from_env as _from_env
        return _from_env()
    except Exception:
        return None


def _search_prompts(domain: str, status_filter: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Pull prompt variants for a domain from whichever backend responds."""
    out: List[Dict[str, Any]] = []
    for backend in (_mind_client(), _vault_client()):
        if backend is None:
            continue
        try:
            if hasattr(backend, "search"):
                raw = backend.search(query=f"prompt {domain}", limit=limit)
            else:
                raw = backend.export(tag=f"prompt:{domain}", limit=limit)
        except Exception as exc:
            logger.debug("evolver: backend %s failed: %s", type(backend).__name__, exc)
            continue
        finally:
            try:
                backend.close()
            except Exception:
                pass
        if isinstance(raw, dict):
            raw = raw.get("results") or raw.get("memories") or raw.get("notes") or []
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") or {}
            if status_filter and meta.get("status") != status_filter:
                continue
            if meta.get("domain") and meta["domain"] != domain:
                continue
            out.append(item)
        if out:
            break
    return out


def _record_prompt_variant(domain: str, prompt_text: str, status: str,
                            samples: int = 0, success_rate: float = 0.0,
                            variant_id: Optional[str] = None) -> Optional[str]:
    """Write a prompt variant to the vault and return its variant_id."""
    backends = []
    m = _mind_client()
    if m is not None:
        backends.append(("mind", m))
    v = _vault_client()
    if v is not None:
        backends.append(("vault", v))
    if not backends:
        return None
    vid = variant_id or f"v-{int(time.time())}-{abs(hash(prompt_text)) % 0xffff:04x}"
    metadata = {
        "domain": domain,
        "status": status,
        "samples": samples,
        "success_rate": success_rate,
        "variant_id": vid,
    }
    content = (
        f"PROMPT_VARIANT [{vid}] domain={domain} status={status} "
        f"samples={samples} rate={success_rate:.3f}\n\n{prompt_text}"
    )
    for name, backend in backends:
        try:
            kwargs = {"tags": [f"prompt:{domain}", "hermes"], "source": "hermes-evolver", "metadata": metadata}
            if name == "mind":
                kwargs["title"] = f"prompt · {domain} · {status} · {vid}"
            backend.ingest(content, **kwargs)
        except Exception as exc:
            logger.debug("evolver: %s write failed: %s", name, exc)
        finally:
            try:
                backend.close()
            except Exception:
                pass
    return vid


# ---------------------------------------------------------------------------
# Public API used by api_server
# ---------------------------------------------------------------------------

def current_prompt(domain: str) -> str:
    """Return the active prompt for a domain, or "" to use the built-in default."""
    if not _enabled() or not domain:
        return ""
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE.get(domain)
        if cached and (now - cached.get("fetched_at", 0)) < _CACHE_TTL:
            return cached.get("prompt", "")
    active = _search_prompts(domain, status_filter="active", limit=3)
    if not active:
        with _CACHE_LOCK:
            _CACHE[domain] = {"prompt": "", "prompt_id": None, "fetched_at": now}
        return ""
    # Take the newest active (vault may have stragglers).
    chosen = active[0]
    content = chosen.get("content") or ""
    # Extract the prompt body (after the metadata header).
    m = re.search(r"PROMPT_VARIANT[^\n]*\n\n(.*)", content, re.DOTALL)
    prompt_text = m.group(1).strip() if m else ""
    meta = chosen.get("metadata") or {}
    with _CACHE_LOCK:
        _CACHE[domain] = {"prompt": prompt_text, "prompt_id": meta.get("variant_id"), "fetched_at": now}
    return prompt_text


def record_outcome(domain: str, variant_id: str, success: bool) -> None:
    """Called from the reflector when a turn completes with a known active variant.

    Stored as a separate `prompt_outcome` entry so the cron can aggregate.
    """
    if not _enabled() or not domain or not variant_id:
        return
    backends: List[Any] = []
    m = _mind_client()
    if m is not None:
        backends.append(m)
    v = _vault_client()
    if v is not None:
        backends.append(v)
    if not backends:
        return
    content = f"PROMPT_OUTCOME domain={domain} variant={variant_id} success={success} ts={datetime.now(timezone.utc).isoformat()}"
    metadata = {"domain": domain, "variant_id": variant_id, "success": success}
    for b in backends:
        try:
            b.ingest(content, tags=[f"prompt_outcome:{domain}"], source="hermes-evolver", metadata=metadata)
        except Exception as exc:
            logger.debug("evolver: outcome write failed: %s", exc)
        finally:
            try:
                b.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Propose / promote — called by n8n cron (not on hot path)
# ---------------------------------------------------------------------------

_PROPOSE_SYSTEM = (
    "You are a prompt engineer. Given a current system prompt and a sample of "
    "recent low-success outcomes (and any high-success outcomes for contrast), "
    "propose ONE refined system prompt that should improve performance.\n\n"
    "Return ONLY the new prompt text — no commentary, no markdown fences."
)


def _aux_call(*, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.3) -> Optional[str]:
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
    except Exception:
        return None
    try:
        client, model = get_text_auxiliary_client(task="prompt_evolver")
    except Exception:
        return None
    if client is None or not model:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None


def _aggregate_outcomes(domain: str, variant_id: str) -> Dict[str, int]:
    """Count successes vs failures for a variant across recent outcomes."""
    outcomes = _search_prompts(domain, limit=200)  # broad pull, we filter manually
    success = fail = 0
    for o in outcomes:
        meta = o.get("metadata") or {}
        if meta.get("variant_id") != variant_id:
            continue
        if meta.get("success") is True:
            success += 1
        elif meta.get("success") is False:
            fail += 1
    return {"success": success, "fail": fail, "total": success + fail}


def maybe_propose(domain: str, min_samples: int = 30, min_failure_rate: float = 0.25) -> Optional[str]:
    """If active variant has enough samples and is underperforming, propose a candidate.

    Returns the new variant_id on creation, None otherwise.
    """
    if not _enabled() or not domain:
        return None
    active = _search_prompts(domain, status_filter="active", limit=1)
    if not active:
        return None
    cur = active[0]
    meta = cur.get("metadata") or {}
    vid = meta.get("variant_id")
    if not vid:
        return None
    agg = _aggregate_outcomes(domain, vid)
    if agg["total"] < min_samples:
        return None
    failure_rate = agg["fail"] / max(agg["total"], 1)
    if failure_rate < min_failure_rate:
        return None
    # Pull the active prompt text and some failing/successful examples for context.
    content = cur.get("content") or ""
    m = re.search(r"PROMPT_VARIANT[^\n]*\n\n(.*)", content, re.DOTALL)
    cur_text = m.group(1).strip() if m else ""
    if not cur_text:
        return None
    user_prompt = (
        f"CURRENT PROMPT (for domain `{domain}`):\n{cur_text}\n\n"
        f"OUTCOMES: {agg['success']} success / {agg['fail']} fail ({agg['total']} total).\n"
        f"Failure rate is {failure_rate:.0%} — please propose a refined version that "
        "addresses likely weaknesses. Return only the new prompt text."
    )
    new_text = _aux_call(system=_PROPOSE_SYSTEM, user=user_prompt)
    if not new_text or new_text.strip() == cur_text.strip():
        return None
    new_id = _record_prompt_variant(domain, new_text, status="candidate")
    logger.info("hermes.evolver.proposed domain=%s new_id=%s failure_rate=%.0f%%",
                 domain, new_id, failure_rate * 100)
    return new_id


def maybe_promote(domain: str, min_samples: int = 30, min_improvement: float = 0.05) -> Optional[str]:
    """Promote a candidate that beats the active variant by >= min_improvement.

    Returns the promoted variant_id or None.
    """
    if not _enabled() or not domain:
        return None
    active = _search_prompts(domain, status_filter="active", limit=1)
    candidate = _search_prompts(domain, status_filter="candidate", limit=1)
    if not active or not candidate:
        return None
    a_meta = active[0].get("metadata") or {}
    c_meta = candidate[0].get("metadata") or {}
    a_id, c_id = a_meta.get("variant_id"), c_meta.get("variant_id")
    if not a_id or not c_id:
        return None
    a_agg = _aggregate_outcomes(domain, a_id)
    c_agg = _aggregate_outcomes(domain, c_id)
    if c_agg["total"] < min_samples:
        return None
    a_rate = a_agg["success"] / max(a_agg["total"], 1)
    c_rate = c_agg["success"] / max(c_agg["total"], 1)
    if c_rate - a_rate < min_improvement:
        return None
    # Demote active to archived, promote candidate to active.
    a_text_match = re.search(r"PROMPT_VARIANT[^\n]*\n\n(.*)", active[0].get("content") or "", re.DOTALL)
    c_text_match = re.search(r"PROMPT_VARIANT[^\n]*\n\n(.*)", candidate[0].get("content") or "", re.DOTALL)
    if not (a_text_match and c_text_match):
        return None
    _record_prompt_variant(domain, a_text_match.group(1).strip(), status="archived",
                            samples=a_agg["total"], success_rate=a_rate, variant_id=a_id)
    _record_prompt_variant(domain, c_text_match.group(1).strip(), status="active",
                            samples=c_agg["total"], success_rate=c_rate, variant_id=c_id)
    with _CACHE_LOCK:
        _CACHE.pop(domain, None)
    logger.info("hermes.evolver.promoted domain=%s old=%s(%.0f%%) new=%s(%.0f%%)",
                 domain, a_id, a_rate * 100, c_id, c_rate * 100)
    return c_id
