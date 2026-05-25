"""HTTP handlers for vault feedback (Phase 3) and routing maturity (Phase 4).

These get registered on the gateway aiohttp app by api_server.py so the
self-improvement loop can be closed without the AIAgent on the request path:

  POST /v1/feedback         — OpenJarvis (and any client) marks the most-recent
                              vault entry for a session with success_flag.
                              Body: {"session_id": str, "signal": "accepted"|"corrected"}
                              Effect: PATCHes the last vault entry's metadata so
                              the n8n maturity cron can compute success rates.

  PUT  /v1/routing/maturity  — n8n cron (Phase 4) pushes updated
                              (domain, model, samples, success_rate, mature)
                              tuples into the in-process routing table.
                              Body: {"updates": [{...}, ...]} OR a single update dict.
                              Effect: vault_router.MaturityTable absorbs the
                              changes so the next chat/completions call honors
                              them.

  GET  /v1/routing/maturity  — Read the current maturity table (for debugging
                              + the n8n cron to compute deltas).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Per-session "last vault entry id" cache. The api_server records into this
# whenever a vault write succeeds for a given session_id so the /v1/feedback
# PATCH knows which entry to update.  Bounded eviction (LRU-ish) — chats are
# small, this never grows past a few hundred entries.

_LAST_ENTRY_LOCK = threading.Lock()
_LAST_ENTRY: "dict[str, tuple[float, str]]" = {}  # session_id -> (ts, memory_id)
_MAX_TRACKED = 512


def record_vault_entry(session_id: str, memory_id: str) -> None:
    if not session_id or not memory_id:
        return
    with _LAST_ENTRY_LOCK:
        _LAST_ENTRY[session_id] = (time.time(), memory_id)
        if len(_LAST_ENTRY) > _MAX_TRACKED:
            # Evict the oldest 25% so we don't churn every insert.
            victims = sorted(_LAST_ENTRY.items(), key=lambda kv: kv[1][0])
            for k, _ in victims[: _MAX_TRACKED // 4]:
                _LAST_ENTRY.pop(k, None)


def last_entry_for(session_id: str) -> Optional[str]:
    if not session_id:
        return None
    with _LAST_ENTRY_LOCK:
        rec = _LAST_ENTRY.get(session_id)
        return rec[1] if rec else None


# ---------------------------------------------------------------------------
# aiohttp handlers
# ---------------------------------------------------------------------------

async def handle_feedback(request) -> "Any":
    """POST /v1/feedback — mark the last vault entry for a session.

    Body shape:
        {"session_id": "...", "signal": "accepted" | "corrected", "note": "optional"}
    """
    from aiohttp import web
    from .client import from_env as _client

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    session_id = str(body.get("session_id", "")).strip()
    signal = str(body.get("signal", "")).strip().lower()
    if not session_id or signal not in ("accepted", "corrected"):
        return web.json_response(
            {"error": "session_id and signal (accepted|corrected) required"},
            status=400,
        )

    memory_id = last_entry_for(session_id)
    if not memory_id:
        return web.json_response({"ok": False, "reason": "no tracked entry for session"}, status=404)

    c = _client()
    if c is None:
        return web.json_response({"ok": False, "reason": "vault not configured"}, status=503)

    metadata: Dict[str, Any] = {
        "success_flag": signal,
        "feedback_ts": time.time(),
    }
    note = str(body.get("note", "")).strip()
    if note:
        metadata["feedback_note"] = note[:280]

    try:
        ok = c.patch_metadata(memory_id, metadata)
    finally:
        c.close()

    return web.json_response({"ok": bool(ok), "memory_id": memory_id, "signal": signal})


async def handle_maturity_get(request) -> "Any":
    from aiohttp import web
    from agent.vault_router import get_maturity_table
    return web.json_response({"maturity": get_maturity_table().snapshot()})


async def handle_maturity_put(request) -> "Any":
    """PUT /v1/routing/maturity — n8n cron pushes updated maturity rows.

    Body shape (either single or batch):
        {"domain": "voice", "model": "groq/llama-...", "samples": 42,
         "success_rate": 0.91, "mature": true}
        OR
        {"updates": [ {...}, {...} ]}
    """
    from aiohttp import web
    from agent.vault_router import get_maturity_table

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    if isinstance(body, dict) and "updates" in body and isinstance(body["updates"], list):
        rows = body["updates"]
    elif isinstance(body, dict):
        rows = [body]
    elif isinstance(body, list):
        rows = body
    else:
        return web.json_response({"error": "expected object or list"}, status=400)

    table = get_maturity_table()
    accepted = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        domain = str(r.get("domain", "")).strip()
        if not domain:
            continue
        model = str(r.get("model", "")).strip()
        try:
            samples = int(r.get("samples", 0) or 0)
            success_rate = float(r.get("success_rate", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        mature = bool(r.get("mature", False))
        table.update(domain, model, samples, success_rate, mature)
        accepted += 1
        logger.info(
            "hermes.routing.maturity domain=%s model=%s samples=%d rate=%.2f mature=%s",
            domain, model, samples, success_rate, mature,
        )

    return web.json_response({"ok": True, "accepted": accepted, "snapshot": table.snapshot()})
