"""Tool failure sentinel (Phase 5 of the agentic upgrade).

When any tool call fails, the sentinel records the pattern to the vault
tagged `tool_failure`. Before subsequent calls to the same tool, recent
failure warnings can be injected into the model's tool-decision context so
it doesn't repeat the same mistake.

Sentinel state is bounded (in-memory ring buffer per tool) and the vault
write is fire-and-forget. The injection path returns ≤3 short bullets to
keep prompt overhead minimal.

Env gates:
  HERMES_TOOL_SENTINEL_ENABLED   default: true   — record failures
  HERMES_TOOL_SENTINEL_INJECT    default: true   — inject warnings pre-call
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Module-level globals (sentinel is a process-wide singleton).
_LOCK = threading.Lock()
_VAULT_CLIENT: Any = None  # set by provider on init
_MIND_CLIENT: Any = None   # set by provider on init
_RECENT: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=10))


def register_backends(vault_client=None, mind_client=None) -> None:
    """Provider hook: called once after VaultProvider.initialize."""
    global _VAULT_CLIENT, _MIND_CLIENT
    with _LOCK:
        _VAULT_CLIENT = vault_client
        _MIND_CLIENT = mind_client


# ---------------------------------------------------------------------------
# Args-shape redaction — never leak values, only key/shape info.
# ---------------------------------------------------------------------------

def _redact_args(args: Any, depth: int = 0) -> Any:
    """Replace values with their type names. Recursive, bounded depth."""
    if depth > 3:
        return "<deep>"
    if args is None:
        return None
    if isinstance(args, bool):
        return "bool"
    if isinstance(args, (int, float)):
        return type(args).__name__
    if isinstance(args, str):
        return f"str[{len(args)}]"
    if isinstance(args, list):
        if not args:
            return "list[0]"
        return [_redact_args(args[0], depth + 1), f"...(n={len(args)})"]
    if isinstance(args, dict):
        return {k: _redact_args(v, depth + 1) for k, v in list(args.items())[:12]}
    return type(args).__name__


def _classify_error(error: Any) -> str:
    """Map an exception or error-string to a coarse class for grouping."""
    if isinstance(error, str):
        msg = error.lower()
    else:
        msg = (str(error) or type(error).__name__).lower()
    if any(k in msg for k in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(k in msg for k in ("connection", "connreset", "econnrefused", "network")):
        return "network"
    if any(k in msg for k in ("not found", "404", "no such")):
        return "not_found"
    if any(k in msg for k in ("forbidden", "401", "403", "unauthorized", "permission")):
        return "auth"
    if any(k in msg for k in ("rate limit", "429", "too many")):
        return "rate_limit"
    if any(k in msg for k in ("invalid", "bad request", "400", "malformed", "schema")):
        return "invalid_input"
    if any(k in msg for k in ("missing", "required")):
        return "missing_field"
    return "other"


def _short(text: str, n: int = 120) -> str:
    return (text or "").replace("\n", " ").strip()[:n]


# ---------------------------------------------------------------------------
# Record (called from tools/registry.py on failure)
# ---------------------------------------------------------------------------

def record_failure(
    tool_name: str,
    args: Any = None,
    error: Any = None,
    context_summary: str = "",
) -> None:
    """Best-effort: log + cache + async vault write."""
    if os.environ.get("HERMES_TOOL_SENTINEL_ENABLED", "true").lower() not in ("1", "true", "yes", "on"):
        return
    if not tool_name:
        return

    err_msg = _short(str(error) if not isinstance(error, str) else error, 240)
    err_class = _classify_error(error)
    shape = _redact_args(args)
    now_iso = datetime.now(timezone.utc).isoformat()
    entry = {
        "ts": time.time(),
        "iso": now_iso,
        "tool": tool_name,
        "error_class": err_class,
        "error_msg": err_msg,
        "args_shape": shape,
        "context": _short(context_summary, 120),
    }

    with _LOCK:
        _RECENT[tool_name].append(entry)
        v_client = _VAULT_CLIENT
        m_client = _MIND_CLIENT

    logger.info(
        "hermes.tool_sentinel.record tool=%s class=%s msg=%r",
        tool_name, err_class, err_msg[:80],
    )

    if not (v_client or m_client):
        return

    # Vault write — background, never blocks the failed call's recovery path.
    threading.Thread(
        target=_write_to_vault,
        args=(entry, v_client, m_client),
        name="tool-sentinel-write",
        daemon=True,
    ).start()


def _write_to_vault(entry: Dict[str, Any], v_client, m_client) -> None:
    content = (
        f"TOOL_FAILURE ({entry['iso']} | {entry['tool']}): "
        f"{entry['error_class']} — {entry['error_msg']}\n"
        f"args_shape: {entry['args_shape']}\n"
        f"context: {entry['context'] or '(none)'}"
    )
    metadata = {
        "tool": entry["tool"],
        "error_class": entry["error_class"],
        "error_msg": entry["error_msg"],
        "args_shape": entry["args_shape"],
    }
    if v_client is not None:
        try:
            v_client.ingest(
                content,
                tags=["tool_failure", entry["tool"]],
                source="hermes-tool-sentinel",
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("tool_sentinel super-agent write failed: %s", exc)
    if m_client is not None:
        try:
            m_client.ingest(
                content,
                tags=["tool_failure", entry["tool"], "hermes"],
                title=f"tool_failure · {entry['tool']} · {entry['iso'][:19]}",
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("tool_sentinel obsidian-mind write failed: %s", exc)


# ---------------------------------------------------------------------------
# Warnings (called from tools/registry.py BEFORE dispatch, if INJECT enabled)
# ---------------------------------------------------------------------------

def warnings_for(tool_name: str, max_bullets: int = 3) -> List[str]:
    """Short bullet list of recent failure modes for a tool. Empty if none.

    Recent = last 24h, grouped by error_class. Returns ≤max_bullets bullets.
    """
    if os.environ.get("HERMES_TOOL_SENTINEL_INJECT", "true").lower() not in ("1", "true", "yes", "on"):
        return []
    if not tool_name:
        return []
    cutoff = time.time() - 86400  # last 24h
    with _LOCK:
        candidates = [e for e in _RECENT.get(tool_name, ()) if e["ts"] >= cutoff]
    if not candidates:
        return []
    # Group by error_class, surface the most-recent message per class.
    by_class: Dict[str, Dict[str, Any]] = {}
    for e in candidates:
        cls = e["error_class"]
        if cls not in by_class or e["ts"] > by_class[cls]["ts"]:
            by_class[cls] = e
    # Sort by recency.
    items = sorted(by_class.values(), key=lambda e: e["ts"], reverse=True)
    bullets = []
    for e in items[:max_bullets]:
        bullets.append(f"{e['error_class']}: {e['error_msg'][:100]}")
    return bullets


def warnings_block(tool_name: str) -> str:
    """Pre-call prompt fragment to inject when warnings exist. Empty otherwise."""
    bullets = warnings_for(tool_name)
    if not bullets:
        return ""
    lines = [f"NOTE — recent failures of `{tool_name}` in the last 24h:"]
    lines += [f"- {b}" for b in bullets]
    lines.append("Adjust arguments / context to avoid these.")
    return "\n".join(lines)
