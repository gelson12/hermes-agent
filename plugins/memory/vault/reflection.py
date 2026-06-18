"""Post-turn reflection (Phase 3 of the agentic upgrade).

After each turn, an auxiliary model reviews (user, assistant, tools_used,
domain) and emits a structured score that gets written to the vault tagged
`reflection`. The vault entry feeds:

  - the routing maturity cron (Phase 4 of the vault loop)
  - the skill distiller (Phase 4 of the agentic upgrade) — fires when a
    multi-step task succeeds with high confidence
  - the prompt evolver (Phase 7) — promotes prompts whose reflections
    consistently score well

Fully off the main response path: fire-and-forget background thread + the
auxiliary client (cheap LLM), so user-visible latency is unaffected.

Env gates:
  HERMES_REFLECTOR_ENABLED   default: true
  HERMES_REFLECTOR_MODEL     default: "" — empty means use auxiliary client's default
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reflection prompt — kept tight, structured output only.
# ---------------------------------------------------------------------------

_REFLECT_SYSTEM = (
    "You are Hermes's post-turn reflector. You read a (question, answer) pair "
    "from a previous turn and return ONLY a single JSON object with this shape:\n"
    "{\n"
    '  "confidence": <number 0..1>,         // how confident the answer is correct + useful\n'
    '  "success": <true|false>,             // did the answer actually address the question?\n'
    '  "learning": "<one short sentence>",  // what this turn teaches Hermes for future turns\n'
    '  "refusal_risk": <true|false>,        // did the answer hedge / refuse / dodge?\n'
    '  "follow_up_needed": <true|false>,    // would a follow-up question help the user?\n'
    '  "tags": ["<short>", ...]             // 1-3 short topic tags, lowercase, kebab-case\n'
    "}\n"
    "Respond with the JSON only. No prose, no markdown fences, no explanations."
)


def _build_reflect_user_prompt(user_text: str, assistant_text: str, domain: str) -> str:
    user_trim = (user_text or "").strip()[:1200]
    asst_trim = (assistant_text or "").strip()[:2400]
    return (
        f"Domain: {domain}\n\n"
        f"USER QUESTION:\n{user_trim}\n\n"
        f"ASSISTANT ANSWER:\n{asst_trim}\n\n"
        "Return JSON only."
    )


# ---------------------------------------------------------------------------
# Robust JSON extraction — auxiliary models sometimes wrap in code fences.
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Strip code fences if present.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    # First try the whole string.
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Fall back to first top-level {...} block.
    m = _JSON_BLOCK_RE.search(cleaned)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _normalize(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce loose model output into the expected shape, with safe defaults."""
    def _bool(v, default=False) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1", "y")
        return default

    def _num(v, default=0.5) -> float:
        try:
            f = float(v)
        except Exception:
            return default
        return max(0.0, min(1.0, f))

    raw_tags = parsed.get("tags") or []
    if not isinstance(raw_tags, list):
        raw_tags = [str(raw_tags)]
    tags = []
    for t in raw_tags[:3]:
        if not isinstance(t, str):
            continue
        slug = re.sub(r"[^a-z0-9-]+", "-", t.lower().strip()).strip("-")
        if slug:
            tags.append(slug[:40])

    return {
        "confidence": _num(parsed.get("confidence"), 0.5),
        "success": _bool(parsed.get("success"), False),
        "learning": str(parsed.get("learning") or "").strip()[:280],
        "refusal_risk": _bool(parsed.get("refusal_risk"), False),
        "follow_up_needed": _bool(parsed.get("follow_up_needed"), False),
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# Auxiliary call wrapper — uses Hermes's existing auxiliary client.
# ---------------------------------------------------------------------------

def _call_auxiliary(user_prompt: str) -> Optional[str]:
    """Invoke the auxiliary LLM (cheap side-task channel) for reflection.

    Uses Hermes's `agent.auxiliary_client.get_text_auxiliary_client(task="reflection")`
    which returns an OpenAI-style (client, model_slug) tuple after walking the
    provider chain (main → OpenRouter → Nous → direct keys). Returns the raw
    text or None on any failure (best-effort; reflection is non-critical).
    """
    model_override = os.environ.get("HERMES_REFLECTOR_MODEL", "").strip() or None
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
    except Exception as exc:
        logger.debug("auxiliary_client import failed: %s", exc)
        return None

    try:
        client, model = get_text_auxiliary_client(task="reflection")
    except Exception as exc:
        logger.debug("get_text_auxiliary_client failed: %s", exc)
        return None

    if client is None or not model:
        logger.debug("reflector: no auxiliary backend resolved")
        return None

    use_model = model_override or model

    try:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": _REFLECT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=0.2,
        )
    except Exception as exc:
        logger.debug("reflector: chat.completions.create failed (%s): %s", use_model, exc)
        return None

    try:
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.debug("reflector: response shape unexpected: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry — fire-and-forget.
# ---------------------------------------------------------------------------

_RECENT_REFLECTIONS: "dict[str, dict]" = {}
_REFLECT_LOCK = threading.Lock()
_MAX_TRACKED = 256


def last_reflection(session_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent reflection for a session (consumed by distiller)."""
    if not session_id:
        return None
    with _REFLECT_LOCK:
        return _RECENT_REFLECTIONS.get(session_id)


def _store_recent(session_id: str, reflection: Dict[str, Any]) -> None:
    if not session_id:
        return
    with _REFLECT_LOCK:
        _RECENT_REFLECTIONS[session_id] = reflection
        if len(_RECENT_REFLECTIONS) > _MAX_TRACKED:
            # Evict oldest 25%.
            victims = sorted(_RECENT_REFLECTIONS.items(), key=lambda kv: kv[1].get("ts", 0))
            for k, _ in victims[: _MAX_TRACKED // 4]:
                _RECENT_REFLECTIONS.pop(k, None)


def _do_reflect(
    session_id: str,
    user_text: str,
    assistant_text: str,
    domain: str,
    platform: str,
    vault_client,
    mind_client,
) -> None:
    """Run reflection and persist to vault. Best-effort throughout."""
    raw = _call_auxiliary(_build_reflect_user_prompt(user_text, assistant_text, domain))
    if not raw:
        logger.debug("reflector: no auxiliary backend available")
        return

    parsed = _extract_json(raw)
    if not parsed:
        logger.debug("reflector: could not parse JSON from auxiliary output (first 120 chars): %r", raw[:120])
        return

    norm = _normalize(parsed)
    norm["ts"] = datetime.now(timezone.utc).timestamp()
    norm["domain"] = domain
    norm["platform"] = platform
    norm["session_id"] = session_id

    _store_recent(session_id, norm)
    logger.info(
        "hermes.reflector.score session=%s conf=%.2f success=%s refusal=%s followup=%s tags=%s",
        session_id, norm["confidence"], norm["success"], norm["refusal_risk"],
        norm["follow_up_needed"], norm["tags"],
    )

    now_iso = datetime.fromtimestamp(norm["ts"], tz=timezone.utc).isoformat()
    content = (
        f"REFLECTION ({now_iso} | {domain}): "
        f"conf={norm['confidence']:.2f} success={norm['success']} "
        f"refusal={norm['refusal_risk']} follow_up={norm['follow_up_needed']}\n"
        f"learning: {norm['learning']}\n"
        f"tags: {', '.join(norm['tags']) or '(none)'}"
    )
    metadata = {
        "session_id": session_id,
        "domain": domain,
        "platform": platform,
        "confidence": norm["confidence"],
        "success": norm["success"],
        "refusal_risk": norm["refusal_risk"],
        "follow_up_needed": norm["follow_up_needed"],
        "reflection_tags": norm["tags"],
    }

    # Write to whichever backends are configured.
    if vault_client is not None:
        try:
            vault_client.ingest(
                content,
                tags=["reflection", domain],
                source="hermes-reflector",
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("reflector: super-agent write failed: %s", exc)

    if mind_client is not None:
        try:
            title = f"reflection · {domain} · {now_iso[:19]}"
            mind_client.ingest(
                content,
                tags=["reflection", domain, "hermes"],
                title=title,
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("reflector: obsidian-mind write failed: %s", exc)

    # Phase 7 — feed the prompt-evolver this turn's success signal for the domain's
    # active variant. This is the link that was missing everywhere: without it the
    # evolver had no outcomes to aggregate and could never propose/improve. Gated by
    # the evolver's own flag; best-effort (never affects reflection).
    try:
        from agent import prompt_evolver as _evolver
        if _evolver._enabled():
            vid = _evolver.active_variant_id(domain) or _evolver.ensure_seed(domain)
            if vid:
                _evolver.record_outcome(domain, vid, bool(norm["success"]))
    except Exception as exc:
        logger.debug("reflector: prompt-evolver outcome skipped: %s", exc)


def reflect_async(
    *,
    session_id: str,
    user_text: str,
    assistant_text: str,
    domain: str,
    platform: str = "",
    vault_client=None,
    mind_client=None,
) -> None:
    """Spawn a background thread to reflect on a completed turn.

    Best-effort: failure anywhere is logged at debug and dropped. The user
    response has already been delivered by the time this is called — nothing
    we do here affects the live turn.
    """
    if not user_text or not assistant_text:
        return
    if not (vault_client or mind_client):
        return  # nowhere to persist
    thread = threading.Thread(
        target=_do_reflect,
        args=(session_id, user_text, assistant_text, domain, platform, vault_client, mind_client),
        name="hermes-reflector",
        daemon=True,
    )
    thread.start()
