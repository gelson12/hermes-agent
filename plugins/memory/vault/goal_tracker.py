"""Persistent goal tracking across sessions (Phase 6 of the agentic upgrade).

User-stated goals get persisted to the vault tagged `goal`. Every turn pulls
active goals and injects them into the system prompt so the agent stays
aligned with what the user is actually trying to accomplish over days/weeks,
not just the current turn.

Tools exposed to the model:
  goal_add(text, due?, priority?)      — create a goal
  goal_list(status?)                   — list goals (default: active only)
  goal_complete(id)                    — mark a goal done
  goal_archive(id)                     — archive a goal (paused/abandoned)
  goal_progress(id, note)              — append a progress note

Goals live in the vault with tag `goal`. Metadata carries the structured
state. ID is a short hex slug derived from creation timestamp + content
hash so it's stable + reproducible.

Env gate: HERMES_GOALS_ENABLED (default true).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

GOAL_ADD_SCHEMA = {
    "name": "goal_add",
    "description": (
        "Record a persistent goal for the user. Use whenever the user expresses "
        "an intent that spans multiple turns or days (\"I want to X\", \"goal: X\", "
        "\"by Friday I need\"). Active goals are shown in every future turn's system "
        "prompt so the assistant stays aligned. Don't use for transient single-turn requests."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The goal, one short sentence."},
            "due": {"type": "string", "description": "Optional ISO date (YYYY-MM-DD)."},
            "priority": {"type": "string", "enum": ["low", "normal", "high"], "description": "Default normal."},
        },
        "required": ["text"],
    },
}

GOAL_LIST_SCHEMA = {
    "name": "goal_list",
    "description": "List goals. Default returns only active goals.",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["active", "done", "archived", "all"], "description": "Default active."},
            "limit": {"type": "integer", "description": "Max results (default 20)."},
        },
        "required": [],
    },
}

GOAL_COMPLETE_SCHEMA = {
    "name": "goal_complete",
    "description": "Mark a goal as done. Use only when the user has explicitly accomplished it.",
    "parameters": {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Goal ID (8-char hex)."}},
        "required": ["id"],
    },
}

GOAL_ARCHIVE_SCHEMA = {
    "name": "goal_archive",
    "description": "Archive a goal (no longer active but not completed). Use when the user pauses or abandons it.",
    "parameters": {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Goal ID."}},
        "required": ["id"],
    },
}

GOAL_PROGRESS_SCHEMA = {
    "name": "goal_progress",
    "description": "Append a progress note to an active goal. Use to record incremental movement toward a goal.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Goal ID."},
            "note": {"type": "string", "description": "One short sentence describing the progress."},
        },
        "required": ["id", "note"],
    },
}


# ---------------------------------------------------------------------------
# Goal model
# ---------------------------------------------------------------------------

def _make_goal_id(text: str, ts: float) -> str:
    """Stable short ID = first 8 hex of sha1(text + ts)."""
    raw = f"{text.strip()[:120]}::{int(ts)}".encode("utf-8", errors="replace")
    return hashlib.sha1(raw).hexdigest()[:8]


def _format_goal_content(goal: Dict[str, Any]) -> str:
    """Vault content body — human-readable + machine-parseable."""
    parts = [
        f"GOAL [{goal['id']}] {goal['status'].upper()}",
        f"text: {goal['text']}",
        f"priority: {goal.get('priority', 'normal')}",
        f"created: {goal['created_at']}",
    ]
    if goal.get("due"):
        parts.append(f"due: {goal['due']}")
    if goal.get("notes"):
        parts.append("progress:")
        for n in goal["notes"][-5:]:
            parts.append(f"  - {n.get('ts', '?')}: {n.get('text', '')}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class GoalTracker:
    """Goal CRUD against the vault. Both super-agent /memory and obsidian-mind
    backends are written in parallel when available.

    Keeps a 5-minute in-memory snapshot of active goals so `active_goals_block()`
    is fast on the hot path (called every turn).
    """

    def __init__(self, vault_client=None, mind_client=None):
        self._vault = vault_client
        self._mind = mind_client
        self._snapshot: List[Dict[str, Any]] = []
        self._snapshot_at: float = 0.0
        self._snapshot_ttl = 300.0  # 5 min

    # -- Tool dispatch ---------------------------------------------------

    def tool_schemas(self) -> List[Dict[str, Any]]:
        if os.environ.get("HERMES_GOALS_ENABLED", "true").lower() not in ("1", "true", "yes", "on"):
            return []
        return [
            GOAL_ADD_SCHEMA, GOAL_LIST_SCHEMA, GOAL_COMPLETE_SCHEMA,
            GOAL_ARCHIVE_SCHEMA, GOAL_PROGRESS_SCHEMA,
        ]

    def handle(self, tool_name: str, args: Dict[str, Any]) -> str:
        try:
            if tool_name == "goal_add":
                return json.dumps(self._add(args.get("text", ""), args.get("due"), args.get("priority")))
            if tool_name == "goal_list":
                return json.dumps(self._list(args.get("status", "active"), int(args.get("limit", 20))))
            if tool_name == "goal_complete":
                return json.dumps(self._set_status(args.get("id", ""), "done"))
            if tool_name == "goal_archive":
                return json.dumps(self._set_status(args.get("id", ""), "archived"))
            if tool_name == "goal_progress":
                return json.dumps(self._append_note(args.get("id", ""), args.get("note", "")))
            return json.dumps({"error": f"unknown tool {tool_name}"})
        except Exception as exc:
            logger.debug("goal_tracker tool %s failed: %s", tool_name, exc)
            return json.dumps({"error": str(exc)[:200]})

    # -- System-prompt block (called every turn) ------------------------

    def active_goals_block(self, max_goals: int = 5) -> str:
        """Short markdown block listing active goals — injected into system prompt."""
        if os.environ.get("HERMES_GOALS_ENABLED", "true").lower() not in ("1", "true", "yes", "on"):
            return ""
        goals = self._cached_active(max_goals)
        if not goals:
            return ""
        lines = ["# Active Goals", ""]
        for g in goals:
            line = f"- [{g['id']}] ({g.get('priority', 'normal')}) {g['text']}"
            if g.get("due"):
                line += f"  _due {g['due']}_"
            lines.append(line)
        lines.append("")
        lines.append("Keep responses aligned with these goals. If a response moves one forward, use goal_progress.")
        return "\n".join(lines)

    # -- CRUD primitives -------------------------------------------------

    def _add(self, text: str, due: Optional[str], priority: Optional[str]) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"error": "text required"}
        now = time.time()
        gid = _make_goal_id(text, now)
        goal = {
            "id": gid,
            "text": text[:300],
            "status": "active",
            "priority": (priority or "normal").lower(),
            "due": (due or "").strip() or None,
            "created_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "notes": [],
        }
        self._persist(goal, action="add")
        self._invalidate_snapshot()
        logger.info("hermes.goal_tracker.add id=%s text=%r", gid, text[:80])
        return {"ok": True, "goal": goal}

    def _set_status(self, gid: str, status: str) -> Dict[str, Any]:
        if not gid:
            return {"error": "id required"}
        existing = self._find(gid)
        if not existing:
            return {"error": f"goal {gid} not found"}
        existing["status"] = status
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._persist(existing, action="status")
        self._invalidate_snapshot()
        logger.info("hermes.goal_tracker.status id=%s status=%s", gid, status)
        return {"ok": True, "goal": existing}

    def _append_note(self, gid: str, note: str) -> Dict[str, Any]:
        if not gid or not note:
            return {"error": "id and note required"}
        existing = self._find(gid)
        if not existing:
            return {"error": f"goal {gid} not found"}
        notes = existing.get("notes") or []
        notes.append({"ts": datetime.now(timezone.utc).isoformat(), "text": note.strip()[:280]})
        existing["notes"] = notes[-20:]  # cap memory
        self._persist(existing, action="progress")
        logger.info("hermes.goal_tracker.update id=%s note=%r", gid, note[:60])
        return {"ok": True, "goal": existing}

    def _list(self, status: str, limit: int) -> Dict[str, Any]:
        entries = self._fetch_all(limit=max(limit, 20))
        if status != "all":
            entries = [g for g in entries if g.get("status") == status]
        return {"goals": entries[:limit]}

    # -- Persistence -----------------------------------------------------

    def _persist(self, goal: Dict[str, Any], *, action: str) -> None:
        content = _format_goal_content(goal)
        metadata = {
            "goal_id": goal["id"],
            "goal_status": goal["status"],
            "goal_priority": goal.get("priority"),
            "goal_due": goal.get("due"),
            "goal_action": action,
        }
        # Encode the FULL goal record as JSON in metadata for round-tripping.
        metadata["goal_data"] = goal

        for backend_name, backend, write_kwargs in self._backend_writers():
            try:
                backend.ingest(
                    content,
                    tags=write_kwargs["tags"],
                    source="hermes-goals",
                    metadata=metadata,
                    **{k: v for k, v in write_kwargs.items() if k not in ("tags",)},
                )
            except Exception as exc:
                logger.debug("goal_tracker %s write failed: %s", backend_name, exc)

    def _backend_writers(self):
        """Yield (name, client, write_kwargs) for each configured backend."""
        if self._vault is not None:
            yield "vault", self._vault, {"tags": ["goal"]}
        if self._mind is not None:
            yield "mind", self._mind, {
                "tags": ["goal", "hermes"],
                "title": f"goal · {datetime.now(timezone.utc).isoformat()[:10]}",
            }

    # -- Fetch -----------------------------------------------------------

    def _fetch_all(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Pull goal entries from vault, reconstruct from metadata."""
        entries: List[Dict[str, Any]] = []
        # Prefer obsidian-mind (newer/canonical going forward); fall back to vault.
        for backend in (self._mind, self._vault):
            if backend is None:
                continue
            try:
                raw = backend.search(query="goal", limit=limit) if hasattr(backend, "search") else backend.export(tag="goal", limit=limit)
            except Exception as exc:
                logger.debug("goal_tracker fetch failed (backend %s): %s", type(backend).__name__, exc)
                continue
            if isinstance(raw, dict):
                raw = raw.get("results") or raw.get("memories") or raw.get("notes") or []
            if not isinstance(raw, list):
                continue
            for item in raw:
                if not isinstance(item, dict):
                    continue
                meta = item.get("metadata") or {}
                data = meta.get("goal_data")
                if isinstance(data, dict) and data.get("id"):
                    entries.append(data)
            if entries:
                break
        # Latest version per id wins (newer overrides older).
        latest: Dict[str, Dict[str, Any]] = {}
        for g in entries:
            gid = g.get("id")
            if not gid:
                continue
            existing = latest.get(gid)
            if not existing or g.get("updated_at", g.get("created_at", "")) >= existing.get("updated_at", existing.get("created_at", "")):
                latest[gid] = g
        return list(latest.values())

    def _find(self, gid: str) -> Optional[Dict[str, Any]]:
        for g in self._fetch_all(limit=100):
            if g.get("id") == gid:
                return g
        return None

    # -- Snapshot cache --------------------------------------------------

    def _cached_active(self, max_goals: int) -> List[Dict[str, Any]]:
        now = time.time()
        if (now - self._snapshot_at) > self._snapshot_ttl or not self._snapshot:
            try:
                all_goals = self._fetch_all(limit=50)
            except Exception:
                all_goals = []
            self._snapshot = [g for g in all_goals if g.get("status") == "active"]
            # Order by priority then due date.
            prio_order = {"high": 0, "normal": 1, "low": 2}
            self._snapshot.sort(
                key=lambda g: (prio_order.get(g.get("priority", "normal"), 1), g.get("due") or "9999-12-31")
            )
            self._snapshot_at = now
        return self._snapshot[:max_goals]

    def _invalidate_snapshot(self) -> None:
        self._snapshot_at = 0.0
        self._snapshot = []
