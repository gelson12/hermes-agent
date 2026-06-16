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
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.passthrough_memory")

# One cached VaultProvider per long-term scope (session-key). Caching is what makes
# the zero-latency queue recall work: turn N's queue_prefetch warms turn N+1's
# prefetch. Voice uses a stable key ("voice-jarvis"), so this is a tiny dict.
_PROVIDERS: Dict[str, Any] = {}
_LOCK = threading.Lock()


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
        logger.debug("passthrough_memory: provider init failed: %s", exc)
        return None


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


def inject_recall(messages: List[Dict[str, Any]], recall: str) -> List[Dict[str, Any]]:
    """Return a NEW messages list with the recalled vault context folded into the
    system message (or prepended as one). Never mutates the input; on any issue
    returns the original list so the request is sent unchanged."""
    if not recall:
        return messages
    try:
        block = (
            "Relevant memory from earlier sessions (use it; if it conflicts with the "
            "user, trust the user — do NOT invent facts not grounded here):\n" + recall
        )
        out = [dict(m) for m in (messages or [])]
        for m in out:
            if m.get("role") == "system" and isinstance(m.get("content"), str):
                m["content"] = m["content"].rstrip() + "\n\n" + block
                return out
        # No system message → prepend one.
        return [{"role": "system", "content": block}] + out
    except Exception:  # noqa: BLE001
        return messages


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


def write_back(session_key: str, session_id: str, user_text: str, assistant_text: str,
               *, platform: str = "api_server") -> None:
    """Persist the finished turn (ingest Q/A + reflect + distill) via the provider's
    sync_turn, off the hot path. Never raises; no-op if disabled/unconfigured."""
    if not (enabled() and session_key and user_text and assistant_text):
        return

    def _go() -> None:
        try:
            prov = _get_provider(session_key, session_id, platform)
            if prov is not None:
                prov.sync_turn(user_text, assistant_text, session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("passthrough_memory write_back failed: %s", exc)

    threading.Thread(target=_go, name="passthrough-writeback", daemon=True).start()


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
