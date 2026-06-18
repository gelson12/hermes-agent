"""Re-attach the vault self-improvement loop to the tool-calling passthrough.

THE BUG this fixes: every Avengers/fj2 voice turn carries ``tools`` (the worker
registers hand-off / device / OSIRIS function schemas), so api_server
short-circuits it to ``_handle_toolcalling_passthrough`` — a raw provider proxy
that BYPASSES the AIAgent and therefore the entire self-improvement loop: recall,
write-back, reflection, distillation, goal-tracking. The loop was silently dead
for the primary consumer (voice), which also starved the brain of recalled
grounding (a driver of confabulation) and meant it never learned across sessions.

This bridges the loop onto that path WITHOUT routing through the AIAgent (which
cannot return OpenAI ``tool_calls`` to the worker — the reason the passthrough
exists in the first place):

  * recall_block()  — ZERO-latency recall: returns the PRIOR turn's cached vault
                      recall (to inject into this turn's prompt) and queues THIS
                      turn's query for the next. Same pattern the AIAgent uses, so
                      no per-turn latency is added.
  * write_back()    — sync_turn() in the background: ingest the Q/A pair, then fire
                      the reflector + distiller. Reuses the exact VaultProvider the
                      AIAgent path uses (cached per session-key so the recall queue
                      persists across turns), so the learning behaviour is identical.

Fail-OPEN by design: every entry point is wrapped so ANY error leaves the raw
passthrough behaviour unchanged — memory can never break a voice turn. Gated by
HERMES_PASSTHROUGH_MEMORY (default on); set =0 to disable instantly.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.passthrough_memory")

# One cached VaultProvider per long-term scope (session-key). Caching is what makes
# the zero-latency queue recall work: turn N's queue_prefetch warms turn N+1's
# prefetch. Voice uses a stable key ("voice-jarvis"), so this is a tiny dict.
_PROVIDERS: Dict[str, Any] = {}
_LOCK = threading.Lock()

# Cumulative write-back outcomes, surfaced on X-Hermes-Memory-Writes so the WRITE
# half of the loop is observable (attach alone doesn't prove a turn actually
# persisted — sync_turn runs in the background and could fail against the backend).
_WRITES: Dict[str, Any] = {"ok": 0, "fail": 0, "skip": 0, "last_error": ""}


def writes_summary() -> str:
    with _LOCK:
        return "ok=%d fail=%d skip=%d" % (_WRITES["ok"], _WRITES["fail"], _WRITES["skip"])


def enabled() -> bool:
    return os.environ.get("HERMES_PASSTHROUGH_MEMORY", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def _get_provider(session_key: str, session_id: str, platform: str):
    """Cached, initialized VaultProvider for this scope. Created once. None on any
    failure or when neither vault backend is configured."""
    if not session_key:
        return None
    with _LOCK:
        prov = _PROVIDERS.get(session_key)
    if prov is not None:
        return prov
    try:
        from plugins.memory.vault.provider import VaultProvider
        prov = VaultProvider()
        # Skip if neither super-agent vault nor obsidian-mind is configured.
        if not prov.is_available() and not os.environ.get("OBSIDIAN_MIND_URL", "").strip():
            return None
        prov.initialize(
            session_id or session_key,
            platform=platform or "api_server",
            gateway_session_key=session_key,
        )
        with _LOCK:
            # Double-checked: another thread may have created one meanwhile.
            existing = _PROVIDERS.get(session_key)
            if existing is not None:
                return existing
            _PROVIDERS[session_key] = prov
        logger.info("passthrough_memory: vault loop ATTACHED to passthrough for scope=%s", session_key)
        return prov
    except Exception as exc:  # noqa: BLE001
        # WARNING (not debug): production runs the gateway logger at WARNING, so a
        # silent no-op here would be invisible. A failed attach is worth one line.
        logger.warning("passthrough_memory: provider init failed (loop NOT attached): %s", exc)
        return None


def status(session_key: str, session_id: str = "", *, platform: str = "api_server") -> str:
    """Cheap, log-independent state of the loop for THIS request, for an observability
    header. 'off' (disabled), 'skip' (no scope key / no backend), 'attached' (live),
    'err'. After recall_block() the provider is cached, so this is a dict hit."""
    if not enabled():
        return "off"
    if not session_key:
        return "skip"
    try:
        return "attached" if _get_provider(session_key, session_id, platform) is not None else "skip"
    except Exception:  # noqa: BLE001
        return "err"


def last_user_text(body: Dict[str, Any]) -> str:
    """The latest user message's plain text from an OpenAI chat-completions body."""
    try:
        for msg in reversed(body.get("messages") or []):
            if (msg or {}).get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):  # multimodal parts
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                return " ".join(t for t in parts if t).strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def inject_context(messages: List[Dict[str, Any]], *, recall: str = "", goals: str = "",
                   evolved: str = "") -> List[Dict[str, Any]]:
    """Return a NEW messages list with recalled vault context, active goals AND/OR the
    evolved-prompt addendum folded into the system message (or prepended as one). Never
    mutates the input; on any issue returns the original list (request sent unchanged)."""
    blocks: List[str] = []
    if recall:
        blocks.append(
            "Relevant memory from earlier sessions (use it; if it conflicts with the "
            "user, trust the user — do NOT invent facts not grounded here):\n" + recall
        )
    if goals:
        blocks.append(goals)      # active_goals_block() is already self-describing
    if evolved:
        blocks.append(evolved)    # evolved guidance addendum (appended, never replaces)
    if not blocks:
        return messages
    try:
        block = "\n\n".join(blocks)
        out = [dict(m) for m in (messages or [])]
        for m in out:
            if m.get("role") == "system" and isinstance(m.get("content"), str):
                m["content"] = m["content"].rstrip() + "\n\n" + block
                return out
        # No system message → prepend one.
        return [{"role": "system", "content": block}] + out
    except Exception:  # noqa: BLE001
        return messages


def inject_recall(messages: List[Dict[str, Any]], recall: str) -> List[Dict[str, Any]]:
    """Back-compat shim: fold just the recall block in (see inject_context)."""
    return inject_context(messages, recall=recall)


def recall_block(session_key: str, session_id: str, user_text: str, *, platform: str = "api_server") -> str:
    """Prior-turn cached vault recall for THIS query (zero latency), and queue this
    query for next turn. '' on anything missing/failing."""
    if not (enabled() and session_key and user_text):
        return ""
    try:
        prov = _get_provider(session_key, session_id, platform)
        if prov is None:
            return ""
        ctx = ""
        try:
            ctx = prov.prefetch(user_text, session_id=session_id) or ""
        except Exception:  # noqa: BLE001
            ctx = ""
        try:
            prov.queue_prefetch(user_text, session_id=session_id)
        except Exception:  # noqa: BLE001
            pass
        return ctx.strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("passthrough_memory recall failed: %s", exc)
        return ""


# Pure chit-chat / acknowledgements that carry no recallable fact. Gating these out
# keeps the vault (and the per-turn reflector aux call) focused on substance.
# Conservative by design: when in doubt, the turn is treated as substantive.
# A single throwaway token/phrase…
_TRIVIAL_TOKEN = (
    r"(?:hi|hey+|hello|yo|jarvis|ok|okay|kay|thanks|thank\s+you|ty|cool|nice|great|"
    r"got\s+it|sounds\s+good|good\s+(?:morning|afternoon|evening|night)|"
    r"are\s+you\s+(?:there|awake|up)|you\s+(?:there|up)|test(?:ing)?|please|sir)"
)
# …and the input is trivial only if it is made ENTIRELY of them ("hey jarvis", "ok thanks").
_TRIVIAL_USER_RE = re.compile(r"^\s*(?:%s[\s!.,?]*)+$" % _TRIVIAL_TOKEN, re.I)


def is_substantive(user_text: str, assistant_text: str) -> bool:
    """Whether a finished turn is worth learning from. Skips greetings, bare
    acknowledgements and one-liners so the vault and the reflector aren't fed pure
    chit-chat. Conservative — anything non-trivial returns True (learn it)."""
    a = (assistant_text or "").strip()
    u = (user_text or "").strip()
    if len(a) < 16:                              # "Done, sir.", "Okay." — confirmations
        return False
    if _TRIVIAL_USER_RE.match(u) and len(a) < 160:  # greeting in AND short reply out
        return False
    return True


def write_back(session_key: str, session_id: str, user_text: str, assistant_text: str,
               *, platform: str = "api_server") -> None:
    """Persist the finished turn (ingest Q/A + reflect + distill) via the provider's
    sync_turn, off the hot path. Never raises; no-op if disabled/unconfigured or if
    the turn is pure chit-chat (no recallable substance)."""
    if not (enabled() and session_key and user_text and assistant_text):
        return
    if not is_substantive(user_text, assistant_text):
        return

    def _go() -> None:
        try:
            prov = _get_provider(session_key, session_id, platform)
            if prov is None:
                with _LOCK:
                    _WRITES["skip"] += 1
                return
            prov.sync_turn(user_text, assistant_text, session_id=session_id)
            with _LOCK:
                _WRITES["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            with _LOCK:
                _WRITES["fail"] += 1
                _WRITES["last_error"] = str(exc)[:200]
            # WARNING (visible at prod log level): a failing write means nothing is
            # being learned — that must not be silent.
            logger.warning("passthrough_memory write_back failed: %s", exc)
        # Goal create + complete/progress share this background thread (off the hot
        # path). Cheap pre-filters inside each gate the auxiliary-LLM judge.
        try:
            if prov is not None:
                maybe_track_goal(session_key, prov, user_text, assistant_text)
                maybe_update_goal(session_key, prov, user_text, assistant_text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("passthrough_memory goal step skipped: %s", exc)

    threading.Thread(target=_go, name="passthrough-writeback", daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Goal-tracking on the voice path.
#
# The AIAgent path injects active goals into the system prompt every turn and lets
# the model call goal_* tools. Neither happens on the passthrough: the AIAgent is
# bypassed, and a goal_* tool_call would be handed to the WORKER (which only knows
# its own hand-off/device tools, not goals). So this bridges both halves:
#   READ  — inject the active-goals block (zero-latency, bridge-cached + bg refresh)
#   WRITE — a cheap regex pre-filter gates a strict auxiliary-LLM "is this a durable
#           goal?" judge; only a clear yes becomes a goal_add. Background only.
# Gated by HERMES_VOICE_GOALS (default on) AND HERMES_GOALS_ENABLED (the tracker's
# own gate). Fail-open everywhere.
# ─────────────────────────────────────────────────────────────────────────────

_GOALS_CACHE: Dict[str, Any] = {}          # session_key -> (ts, block_str)
_GOALS_TTL = 120.0
_RECENT_GOALS: Dict[str, List[str]] = {}   # session_key -> normalized goal texts (dupe guard)
_GOALS_COUNTS: Dict[str, int] = {"added": 0, "skip": 0, "dupe": 0, "fail": 0, "done": 0, "prog": 0}

_GOAL_HINT_RE = re.compile(
    r"\b(?:my goal|goal:|remind me to|i want to|i'd like to|i would like to|"
    r"i need to|i'm trying to|i am trying to|i plan to|i intend to|"
    r"by (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next week|"
    r"end of (?:the )?(?:day|week|month))|"
    r"help me (?:build|launch|finish|write|plan|learn|set up|create|organi[sz]e))\b",
    re.I,
)

_GOAL_JUDGE_SYSTEM = (
    "You decide whether the user's utterance states a PERSISTENT goal worth tracking "
    "across sessions (spanning multiple turns or days), versus a transient single-turn "
    "request. Reply with JSON ONLY: "
    '{"is_goal": true|false, "goal": "<one short imperative sentence, <12 words>", '
    '"due": "<YYYY-MM-DD or empty>", "priority": "low|normal|high"}. '
    "is_goal=false for questions, chit-chat, one-off commands (\"open my email\", "
    "\"what's the weather\"), or anything already completed. is_goal=true only for durable "
    "intents like \"I want to launch the app by Friday\", \"remind me to renew the domain\", "
    "\"my goal is to learn Spanish\"."
)

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(cleaned)
    except Exception:  # noqa: BLE001
        pass
    m = _JSON_BLOCK_RE.search(cleaned)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def goals_enabled() -> bool:
    voice = os.environ.get("HERMES_VOICE_GOALS", "1").strip().lower() not in ("0", "false", "no", "off", "")
    tracker = os.environ.get("HERMES_GOALS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
    return voice and tracker


def goals_summary() -> str:
    with _LOCK:
        return "added=%d skip=%d fail=%d done=%d prog=%d" % (
            _GOALS_COUNTS["added"], _GOALS_COUNTS["skip"] + _GOALS_COUNTS["dupe"],
            _GOALS_COUNTS["fail"], _GOALS_COUNTS["done"], _GOALS_COUNTS["prog"],
        )


def active_goal_count(block: str) -> int:
    """Number of goals listed in an active_goals_block() string (lines like '- [id] …')."""
    if not block:
        return 0
    return sum(1 for ln in block.splitlines() if ln.lstrip().startswith("- ["))


def goals_block(session_key: str, provider) -> str:
    """Zero-latency active-goals block for the system prompt. Returns the bridge-cached
    value and kicks a background refresh when stale (active_goals_block may hit the
    network, so it must never run on the hot path). '' when disabled / no goals."""
    if not (goals_enabled() and session_key and provider is not None):
        return ""
    now = time.time()
    ts, block = _GOALS_CACHE.get(session_key, (0.0, ""))
    if now - ts > _GOALS_TTL:
        def _refresh() -> None:
            try:
                b = provider._goals.active_goals_block(max_goals=5) or ""
                _GOALS_CACHE[session_key] = (time.time(), b)
            except Exception as exc:  # noqa: BLE001
                logger.debug("passthrough_memory goals refresh failed: %s", exc)
        threading.Thread(target=_refresh, name="goals-refresh", daemon=True).start()
    return block  # may be empty/stale on the first turn; warmed for the next


def goals_block_for(session_key: str, session_id: str, *, platform: str = "api_server") -> str:
    """Active-goals block via the cached provider for this scope. '' on anything missing."""
    if not (enabled() and goals_enabled() and session_key):
        return ""
    try:
        prov = _get_provider(session_key, session_id, platform)
        if prov is None:
            return ""
        return goals_block(session_key, prov)
    except Exception as exc:  # noqa: BLE001
        logger.debug("passthrough_memory goals_block_for failed: %s", exc)
        return ""


def _looks_like_goal(user_text: str) -> bool:
    """Cheap pre-filter so the auxiliary-LLM judge only runs on plausible candidates."""
    u = (user_text or "").strip()
    if len(u) < 8:
        return False
    return bool(_GOAL_HINT_RE.search(u))


def _extract_goal_llm(user_text: str, assistant_text: str) -> Optional[Dict[str, Any]]:
    """Strict auxiliary-LLM judge: return {text, due?, priority?} if the turn states a
    durable goal, else None. Reuses Hermes's cheap auxiliary client (same as the
    reflector). None on any failure → conservatively, no goal is created."""
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
        client, model = get_text_auxiliary_client(task="reflection")
    except Exception as exc:  # noqa: BLE001
        logger.debug("passthrough_memory goals: aux client unavailable: %s", exc)
        return None
    if client is None or not model:
        return None
    use_model = os.environ.get("HERMES_GOALS_MODEL", "").strip() or model
    try:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": _GOAL_JUDGE_SYSTEM},
                {"role": "user", "content": "User said: %s\nAssistant replied: %s\nJSON:" % (
                    (user_text or "")[:500], (assistant_text or "")[:300])},
            ],
            max_tokens=120,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("passthrough_memory goals: aux call failed: %s", exc)
        return None
    parsed = _extract_json(raw)
    if not parsed or not parsed.get("is_goal"):
        return None
    goal = str(parsed.get("goal") or "").strip().rstrip(".!").strip()
    if not (4 <= len(goal) <= 200):
        return None
    out: Dict[str, Any] = {"text": goal}
    due = str(parsed.get("due") or "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", due):
        out["due"] = due
    prio = str(parsed.get("priority") or "").strip().lower()
    if prio in ("low", "normal", "high"):
        out["priority"] = prio
    return out


def _norm_goal(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def maybe_track_goal(session_key: str, provider, user_text: str, assistant_text: str) -> None:
    """Background: if the turn states a durable goal, persist it via the GoalTracker.
    Pre-filter (cheap) → LLM judge (only on candidates) → dupe guard → goal_add."""
    if not goals_enabled() or provider is None or not _looks_like_goal(user_text):
        return
    extracted = _extract_goal_llm(user_text, assistant_text)
    if not extracted:
        with _LOCK:
            _GOALS_COUNTS["skip"] += 1
        return
    key = _norm_goal(extracted["text"])
    seen = _RECENT_GOALS.setdefault(session_key, [])
    if any(key in s or s in key for s in seen):
        with _LOCK:
            _GOALS_COUNTS["dupe"] += 1
        return
    try:
        res = provider._goals.handle("goal_add", extracted) or ""
        ok = '"ok"' in res and '"error"' not in res
        with _LOCK:
            _GOALS_COUNTS["added" if ok else "fail"] += 1
        if ok:
            seen.append(key)
            del seen[:-20]                       # cap the dupe-guard memory
            _GOALS_CACHE.pop(session_key, None)  # so the new goal shows next turn
            logger.warning("passthrough_memory: goal tracked: %r", extracted["text"][:80])
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _GOALS_COUNTS["fail"] += 1
        logger.warning("passthrough_memory goal_add failed: %s", exc)


# Closes the goal loop: detect when the user reports COMPLETING or PROGRESSING an
# existing goal, so active goals don't accumulate forever (and stop polluting the
# always-injected goal block once achieved).
_GOAL_UPDATE_RE = re.compile(
    r"\b(?:done|finished|completed?|wrapped up|shipped|launched|sorted|"
    r"made (?:some )?progress|making progress|started (?:on|working)|got .* done|"
    r"i'?ve (?:now )?(?:done|finished|completed|shipped|launched|sent|built))\b",
    re.I,
)

_GOAL_UPDATE_SYSTEM = (
    "You are given the user's ACTIVE goals (id + text) and what they just said. Decide if "
    "the utterance reports COMPLETING or PROGRESSING one of those exact goals. Reply JSON "
    'ONLY: {"action":"complete|progress|none","goal_id":"<id from the list or empty>",'
    '"note":"<short progress note or empty>"}. Use "complete" only for clear completion of '
    'a listed goal; "progress" for incremental movement; "none" otherwise. goal_id MUST be '
    "one of the provided ids — never invent one."
)


def _judge_goal_update(user_text: str, assistant_text: str, goals_brief: str) -> Optional[Dict[str, Any]]:
    """Strict auxiliary-LLM judge: does the turn complete/progress a listed goal? Reuses
    the reflector's cheap aux client. None on any failure (→ no change)."""
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
        client, model = get_text_auxiliary_client(task="reflection")
    except Exception as exc:  # noqa: BLE001
        logger.debug("passthrough_memory goal-update: aux unavailable: %s", exc)
        return None
    if client is None or not model:
        return None
    use_model = os.environ.get("HERMES_GOALS_MODEL", "").strip() or model
    try:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": _GOAL_UPDATE_SYSTEM},
                {"role": "user", "content": "ACTIVE GOALS:\n%s\n\nUser said: %s\nAssistant replied: %s\nJSON:" % (
                    goals_brief, (user_text or "")[:400], (assistant_text or "")[:200])},
            ],
            max_tokens=80,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("passthrough_memory goal-update: aux call failed: %s", exc)
        return None
    return _extract_json(raw)


def maybe_update_goal(session_key: str, provider, user_text: str, assistant_text: str) -> None:
    """Background: if the turn reports finishing/progressing an ACTIVE goal, mark it done
    or append a progress note. Pre-filter → judge (only with the active list) → handle().
    Conservative: only acts on a goal_id that's actually in the active list."""
    if not goals_enabled() or provider is None or not _GOAL_UPDATE_RE.search(user_text or ""):
        return
    try:
        actives = provider._goals._cached_active(8) or []
    except Exception:  # noqa: BLE001
        actives = []
    if not actives:
        return
    valid = {g.get("id"): g.get("text", "") for g in actives if g.get("id")}
    if not valid:
        return
    brief = "\n".join("- [%s] %s" % (gid, txt) for gid, txt in valid.items())
    judged = _judge_goal_update(user_text, assistant_text, brief)
    if not judged:
        return
    action = str(judged.get("action") or "none").lower()
    gid = str(judged.get("goal_id") or "").strip()
    if gid not in valid:
        return
    try:
        if action == "complete":
            res = provider._goals.handle("goal_complete", {"id": gid}) or ""
            ok = '"ok"' in res and '"error"' not in res
            with _LOCK:
                _GOALS_COUNTS["done" if ok else "fail"] += 1
            if ok:
                _GOALS_CACHE.pop(session_key, None)   # drop it from the injected block
                logger.warning("passthrough_memory: goal completed [%s] %r", gid, valid[gid][:60])
        elif action == "progress":
            note = str(judged.get("note") or "").strip()[:200] or "progress noted"
            res = provider._goals.handle("goal_progress", {"id": gid, "note": note}) or ""
            ok = '"ok"' in res and '"error"' not in res
            with _LOCK:
                _GOALS_COUNTS["prog" if ok else "fail"] += 1
            if ok:
                logger.warning("passthrough_memory: goal progress [%s]: %r", gid, note[:60])
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _GOALS_COUNTS["fail"] += 1
        logger.warning("passthrough_memory goal update failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt-evolver on the voice path.
#
# The AIAgent appends the domain's active evolved prompt (a guidance addendum) to the
# system prompt; the passthrough never did. Bridge the READ: inject current_prompt()
# for the voice domain, zero-latency (bridge-cached + background refresh, since
# current_prompt may hit the network and seed the baseline). The OUTCOME signal that
# drives evolution is wired at the reflector (see reflection.py); promotion stays
# deliberate, so nothing unproven is ever auto-applied to this shared brain.
# ─────────────────────────────────────────────────────────────────────────────

_EVOLVER_CACHE: Dict[str, Any] = {}        # session_key -> (ts, prompt_str, domain, candidate_count)
_EVOLVER_TTL = 120.0


def _evolver_domain(session_key: str) -> str:
    from plugins.memory.vault.provider import _domain_from_kwargs
    return _domain_from_kwargs("api_server", session_key or "")


def evolved_prompt_block(session_key: str, session_id: str = "", *, platform: str = "api_server") -> str:
    """Active evolved-prompt addendum for the voice domain. Zero-latency: returns the
    bridge-cached value and refreshes in the background (current_prompt may hit the
    network + seeds the baseline). '' when the evolver is disabled / no active addendum."""
    if not (enabled() and session_key):
        return ""
    try:
        from agent import prompt_evolver as _ev
        if not _ev._enabled():
            return ""
    except Exception:  # noqa: BLE001
        return ""
    now = time.time()
    ts, prompt, _dom, _cands = _EVOLVER_CACHE.get(session_key, (0.0, "", "", 0))
    if now - ts > _EVOLVER_TTL:
        def _refresh() -> None:
            try:
                from agent import prompt_evolver as _ev2
                dom = _evolver_domain(session_key)
                _ev2.ensure_seed(dom)                 # bootstrap a baseline so outcomes have a target
                p = _ev2.current_prompt(dom) or ""
                cands = _ev2.candidate_count(dom)     # proposals awaiting review
                _EVOLVER_CACHE[session_key] = (time.time(), p, dom, cands)
            except Exception as exc:  # noqa: BLE001
                logger.debug("passthrough_memory evolver refresh failed: %s", exc)
        threading.Thread(target=_refresh, name="evolver-refresh", daemon=True).start()
    return prompt


def evolver_summary(session_key: str) -> str:
    _ts, prompt, dom, cands = _EVOLVER_CACHE.get(session_key, (0.0, "", "", 0))
    return "domain=%s applied=%d candidates=%d" % (dom or "?", 1 if prompt else 0, cands or 0)


class SSEContentAccumulator:
    """Accumulate assistant ``content`` from a streamed OpenAI SSE response while it
    is forwarded verbatim. Buffers partial lines across chunk boundaries; ignores
    tool_call deltas and any unparsable line. Used to capture the full reply for
    write-back without altering the proxied bytes."""

    def __init__(self) -> None:
        self._buf = ""
        self._parts: List[str] = []

    def feed(self, chunk: bytes) -> None:
        try:
            self._buf += chunk.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
                delta = (obj.get("choices") or [{}])[0].get("delta", {}) or {}
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    self._parts.append(piece)
            except Exception:  # noqa: BLE001
                continue

    def text(self) -> str:
        return "".join(self._parts).strip()
