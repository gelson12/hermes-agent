"""VaultProvider — MemoryProvider implementation backed by the Obsidian Vault gateway.

Wires the self-improvement loop:

  Step 1 (recall, this file): each turn-start, search the vault for prior Q+A pairs whose
  keywords overlap the current user message. Inject the top matches as context so the
  model can answer directly without redoing work.

  Step 4 (write-back, this file): each turn-end, ingest the (user, assistant) pair tagged
  by domain (voice/code/desktop/crypto/general) and platform. Future turns recall it.

  Step 2 (routing, separate file agent/vault_router.py): a getter exposes the top match's
  confidence score so the api_server layer can downgrade to a cheaper model when the
  vault already has a strong answer.

  Step 3 (success feedback, separate file feedback.py + OpenJarvis hooks): /v1/feedback
  PATCHes the most-recent vault entry with success_flag=accepted|corrected.

Industry-grade techniques applied here:
  - Background prefetch (queue at end-of-turn, consume at start-of-next-turn -> zero
    added latency, same pattern retaindb uses).
  - Stop-word filtered token overlap with IDF-weighted scoring + recency decay (cheap,
    works without embeddings; embeddings deferred until vault > 10k entries).
  - Domain segmentation via tags so voice questions don't pollute code recall.
  - Secret redaction on every write via Hermes's existing agent.redact module.
  - Best-effort: vault outages never break the chat path.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider

from .client import VaultClient, from_env as client_from_env
from .obsidian_mind import ObsidianMindClient, from_env as mind_from_env
from . import tool_sentinel as _tool_sentinel
from .goal_tracker import GoalTracker

logger = logging.getLogger(__name__)

# Lightweight English + filler stop-words. Kept short — over-aggressive filtering
# eats too many distinguishing keywords ("how do I", "where is" etc. are routine).
_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "i", "in", "is", "it", "its", "of", "on", "or", "so", "that", "the",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
    "you", "your", "yours", "we", "our", "this", "these", "those", "do", "does",
    "did", "can", "could", "would", "should", "will", "shall", "may", "might",
    "just", "please", "thanks", "hey", "hi", "hello", "ok", "okay", "yes", "no",
})

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS and len(t) > 2]


def _domain_from_kwargs(platform: str, gateway_session_key: str) -> str:
    """Derive the vault tag bucket from runtime context.

    The vault uses tags as the primary segmentation axis (per the existing n8n
    crypto workflow which tags `crypto_signal`). We pick a domain so that voice
    recall doesn't shadow CLI/code recall and vice-versa.
    """
    key = (gateway_session_key or "").lower()
    if key.startswith("livekit") or "voice" in key or "friday" in key or "jarvis" in key:
        return "voice"
    plat = (platform or "").lower()
    if plat in ("livekit", "voice", "openjarvis"):
        return "voice"
    if plat in ("cli", "hermes-cli"):
        return "code"
    if plat in ("telegram", "discord", "matrix", "signal"):
        return "chat"
    if plat == "cron":
        return "cron"
    return "general"


def _entry_text(entry: Dict[str, Any]) -> str:
    """Pull the content text out of a vault entry, defensive against schema drift."""
    if not isinstance(entry, dict):
        return ""
    for k in ("content", "text", "body", "summary"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _entry_timestamp(entry: Dict[str, Any]) -> float:
    """Best-effort created_at unix timestamp; returns now() if unknown."""
    if not isinstance(entry, dict):
        return time.time()
    for k in ("created_at", "createdAt", "timestamp", "ts"):
        v = entry.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                # Strip trailing Z, parse ISO8601.
                return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
            except Exception:
                continue
    return time.time()


def _score_entry(query_tokens: List[str], entry_tokens: List[str], age_seconds: float) -> float:
    """Score one entry vs the query.

    confidence = idf_weighted_overlap * recency_decay

    - idf_weighted_overlap: count of matching tokens weighted by 1/log(1+token_freq).
      Pragmatic stand-in for true IDF without needing a corpus index — rare tokens
      score higher than common ones across the entry.
    - recency_decay: exp(-age_days / 30). Half-life of ~20 days. Knowledge from
      last week is worth more than knowledge from last quarter.

    Bounded to [0, 1] for downstream routing thresholds.
    """
    if not query_tokens or not entry_tokens:
        return 0.0
    entry_set = set(entry_tokens)
    entry_freq: Dict[str, int] = {}
    for t in entry_tokens:
        entry_freq[t] = entry_freq.get(t, 0) + 1

    overlap = 0.0
    for qt in set(query_tokens):
        if qt in entry_set:
            overlap += 1.0 / math.log(1 + entry_freq[qt] + 1)

    # Normalize against the query side so longer queries don't inflate the score.
    overlap_norm = overlap / max(len(set(query_tokens)), 1)

    age_days = max(age_seconds, 0) / 86400.0
    recency = math.exp(-age_days / 30.0)

    return max(0.0, min(1.0, overlap_norm * recency))


class VaultProvider(MemoryProvider):
    """Obsidian Vault-backed cross-session memory.

    Single external memory provider in Hermes (subject to the one-external-provider
    limit in MemoryManager).
    """

    def __init__(self):
        self._client: Optional[VaultClient] = None
        self._mind: Optional[ObsidianMindClient] = None
        # Phase 6 — GoalTracker is created here (before MemoryManager indexes
        # tool schemas) but its backends are wired in initialize() once
        # VAULT_SECRET / OBSIDIAN_MIND_URL are resolved.  This lets the
        # MemoryManager register the goal_* tool schemas at startup, even
        # though the backends become available a moment later.
        self._goals: GoalTracker = GoalTracker(vault_client=None, mind_client=None)
        self._session_id: str = ""
        self._domain: str = "general"
        self._platform: str = ""
        self._user_id: str = ""

        # Background prefetch state — same pattern as retaindb to avoid added latency.
        self._lock = threading.Lock()
        self._pending_context: str = ""
        self._pending_top: Tuple[float, Dict[str, Any]] = (0.0, {})
        self._prefetch_threads: List[threading.Thread] = []

        # Last-turn tracking so Phase 3 /v1/feedback can patch the right entry.
        self._last_ingest_id: Optional[str] = None
        self._last_ingest_lock = threading.Lock()

    # -- Identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "vault"

    def is_available(self) -> bool:
        import os
        return bool(os.environ.get("VAULT_SECRET", "").strip())

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "secret",
                "description": "Vault gateway secret (X-Vault-Secret header)",
                "secret": True,
                "required": True,
                "env_var": "VAULT_SECRET",
            },
            {
                "key": "base_url",
                "description": "Vault gateway base URL",
                "default": "https://super-agent-production.up.railway.app",
                "env_var": "VAULT_BASE_URL",
            },
        ]

    # -- Lifecycle ---------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._client = client_from_env()
        self._mind = mind_from_env()  # Secondary recall source — gracefully None.
        # Either backend is enough for the loop (super-agent may be intentionally off).
        if not (self._client or self._mind):
            logger.warning("vault provider activated but neither VAULT_SECRET nor OBSIDIAN_MIND_URL set")
            return
        self._session_id = session_id
        self._platform = str(kwargs.get("platform", "") or "")
        self._user_id = str(kwargs.get("user_id", "") or "")
        gateway_key = str(kwargs.get("gateway_session_key", "") or "")
        self._domain = _domain_from_kwargs(self._platform, gateway_key)

        # Phase 5 — tool sentinel needs the same backends to write failure entries.
        try:
            _tool_sentinel.register_backends(vault_client=self._client, mind_client=self._mind)
        except Exception as exc:
            logger.debug("tool_sentinel register failed: %s", exc)

        # Phase 6 — goal tracker: wire backends now that env is resolved.
        # GoalTracker instance was created in __init__ so schemas are available
        # at MemoryManager indexing time; we only attach the live backends here.
        try:
            self._goals._vault = self._client
            self._goals._mind = self._mind
        except Exception as exc:
            logger.debug("goal_tracker backend wiring failed: %s", exc)

        logger.info(
            "vault provider initialized session=%s domain=%s platform=%s mind=%s goals=%s",
            session_id, self._domain, self._platform, bool(self._mind), bool(self._goals),
        )

    def system_prompt_block(self) -> str:
        parts = [
            "# Vault Memory",
            f"Active (domain: {self._domain}). Prior Q+A pairs may appear in <memory-context>. "
            "If the retrieved context already answers the user, respond directly without redoing "
            "research or tool calls. The vault is authoritative for past decisions and learned facts.",
        ]
        # Phase 6 — inject active goals so the agent stays aligned across sessions.
        if self._goals is not None:
            try:
                gb = self._goals.active_goals_block(max_goals=5)
                if gb:
                    parts.append(gb)
            except Exception as exc:
                logger.debug("goal_tracker active_goals_block failed: %s", exc)
        return "\n\n".join(parts)

    # -- Background prefetch (retaindb-style: queue post-turn, consume next turn-start) --

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._client or not query:
            return
        for t in self._prefetch_threads:
            t.join(timeout=1.5)
        thread = threading.Thread(
            target=self._do_prefetch,
            args=(query,),
            name="vault-prefetch",
            daemon=True,
        )
        self._prefetch_threads = [thread]
        thread.start()

    def _do_prefetch(self, query: str) -> None:
        # Pull from primary (super-agent /memory) and secondary (obsidian-mind)
        # in parallel.  Both are best-effort; either failure leaves the other.
        primary_entries: List[Dict[str, Any]] = []
        mind_entries: List[Dict[str, Any]] = []

        def _pull_primary() -> None:
            nonlocal primary_entries
            try:
                primary_entries = self._client.export(tag=self._domain, limit=24)
            except Exception as exc:
                logger.debug("vault prefetch primary failed: %s", exc)

        def _pull_mind() -> None:
            nonlocal mind_entries
            if not self._mind:
                return
            try:
                mind_entries = self._mind.search(query, limit=8)
            except Exception as exc:
                logger.debug("obsidian-mind prefetch failed: %s", exc)

        threads = [
            threading.Thread(target=_pull_primary, daemon=True),
            threading.Thread(target=_pull_mind, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.5)

        if not primary_entries and not mind_entries:
            return

        query_tokens = _tokenize(query)
        if not query_tokens:
            return

        now = time.time()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for e in primary_entries:
            text = _entry_text(e)
            if not text:
                continue
            score = _score_entry(query_tokens, _tokenize(text), now - _entry_timestamp(e))
            if score > 0:
                scored.append((score, e))
        # Mind results — boost by 1.15 because they came from a curated
        # knowledge layer (search relevance already pre-filtered).
        for e in mind_entries:
            text = _entry_text(e)
            if not text:
                continue
            score = _score_entry(query_tokens, _tokenize(text), now - _entry_timestamp(e)) * 1.15
            score = min(1.0, score)
            if score > 0:
                e = dict(e)
                e.setdefault("source", "obsidian-mind")
                scored.append((score, e))
        if not scored:
            return
        scored.sort(key=lambda p: p[0], reverse=True)
        top = scored[: 3]
        top_score, top_entry = top[0]

        lines = ["[Vault Recall]"]
        for score, e in top:
            origin = "mind" if e.get("source") == "obsidian-mind" else "qa"
            snippet = _entry_text(e)[:480].replace("\n", " ")
            lines.append(f"- ({origin} score {score:.2f}) {snippet}")
        context = "\n".join(lines)

        with self._lock:
            self._pending_context = context
            self._pending_top = (top_score, top_entry)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        with self._lock:
            ctx = self._pending_context
            self._pending_context = ""
            # Note: keep self._pending_top so vault_router can read confidence after prefetch.
        return ctx

    # -- Step 2 routing read-side ------------------------------------------

    def last_top_confidence(self) -> float:
        """Confidence (0..1) of the highest-scoring vault hit for the most recent
        prefetch.  Consumed by agent/vault_router.py to pick cheap vs big model.
        Returns 0.0 if no prefetch has run or no hits were found.
        """
        with self._lock:
            return float(self._pending_top[0])

    def last_top_hits_count(self) -> int:
        """Number of useful hits in the last prefetch (>= 0)."""
        with self._lock:
            return 1 if self._pending_top[0] > 0 else 0

    def current_domain(self) -> str:
        return self._domain

    # -- Step 4 write-back -------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Either backend (primary super-agent /memory OR secondary obsidian-mind)
        # is enough to keep the loop running.  This is the post-super-agent
        # design: super-agent may be intentionally off for cost; obsidian-mind
        # is the durable source of truth.
        if (not self._client and not self._mind) or not user_content or not assistant_content:
            return

        # Skip writes when the system prompt smells like it contains secrets.
        # Conservative: drop the whole turn rather than try to surgically redact.
        if _looks_secret(user_content) or _looks_secret(assistant_content):
            logger.debug("vault sync_turn skipped: secret-shaped content")
            return

        # Apply Hermes's existing display-time redactor as belt-and-braces.
        try:
            from agent.redact import redact_sensitive_text
            user_safe = redact_sensitive_text(user_content, force=True)
            asst_safe = redact_sensitive_text(assistant_content, force=True)
        except Exception:
            user_safe = user_content
            asst_safe = assistant_content

        now = datetime.now(timezone.utc).isoformat()
        compacted = (
            f"Q ({now} | {self._domain}): {user_safe.strip()[:1200]}\n"
            f"A: {asst_safe.strip()[:2400]}"
        )
        metadata = {
            "session_id": session_id or self._session_id,
            "platform": self._platform,
            "user_id": self._user_id,
            "domain": self._domain,
        }

        sid = session_id or self._session_id

        # Each successful write registers the entry id so Phase 3 /v1/feedback
        # can patch it later.  Either source counts — first-wins recording so
        # we don't overwrite a primary id with a mind id (or vice versa).
        def _maybe_record(mem_id: str) -> None:
            with self._last_ingest_lock:
                if not self._last_ingest_id:
                    self._last_ingest_id = mem_id
            try:
                from .feedback import record_vault_entry
                record_vault_entry(sid, mem_id)
            except Exception:
                pass

        # Fire writes to both vault sources in parallel background threads.
        def _write_primary():
            if not self._client:
                return
            try:
                result = self._client.ingest(
                    compacted,
                    tags=["qa", self._domain],
                    source="hermes-vault",
                    metadata=metadata,
                )
                if isinstance(result, dict):
                    mem_id = result.get("id") or result.get("memory_id") or (result.get("memory") or {}).get("id")
                    if mem_id:
                        _maybe_record(str(mem_id))
            except Exception as exc:
                logger.debug("vault sync_turn primary write failed: %s", exc)

        def _write_mind():
            if not self._mind:
                return
            try:
                # Brief title so the note shows up usefully in the Obsidian UI.
                title = (user_safe.strip()[:60] or self._domain) + f" · {self._domain}"
                result = self._mind.ingest(
                    compacted,
                    tags=["qa", self._domain, "hermes"],
                    title=title,
                    metadata=metadata,
                )
                if isinstance(result, dict):
                    # obsidian-mind's POST /api/notes/:path returns the path
                    # we wrote to; use that as the stable entry id.
                    mem_id = (
                        result.get("path")
                        or result.get("id")
                        or (result.get("note") or {}).get("path")
                        or (result.get("note") or {}).get("id")
                    )
                    if mem_id:
                        _maybe_record(f"mind:{mem_id}")
            except Exception as exc:
                logger.debug("vault sync_turn mind write failed: %s", exc)

        threading.Thread(target=_write_primary, name="vault-write", daemon=True).start()
        threading.Thread(target=_write_mind, name="mind-write", daemon=True).start()

        # Phase 3 — Reflector hook.  Background critique of (Q, A) lands in
        # vault tag=reflection.  Gated by HERMES_REFLECTOR_ENABLED.
        try:
            import os as _os
            if _os.environ.get("HERMES_REFLECTOR_ENABLED", "true").lower() in ("1", "true", "yes", "on"):
                from .reflection import reflect_async
                reflect_async(
                    session_id=sid,
                    user_text=user_safe,
                    assistant_text=asst_safe,
                    domain=self._domain,
                    platform=self._platform,
                    vault_client=self._client,
                    mind_client=self._mind,
                )
        except Exception as _rexc:
            logger.debug("reflector hook skipped: %s", _rexc)

        # Phase 4 (voice) — background skill distiller. Decoupled from the planner
        # so STREAMING voice turns also learn reusable skills. Spawns its own daemon
        # thread AFTER the response, so it adds ZERO latency to the turn. Gated by
        # HERMES_DISTILLER_ENABLED + a procedural-answer check (chit-chat/refusals
        # never qualify).
        try:
            import os as _os2
            if _os2.environ.get("HERMES_DISTILLER_ENABLED", "true").lower() in ("1", "true", "yes", "on"):
                from .distiller import should_distill_turn, distill_turn_async
                if should_distill_turn(user_safe, asst_safe):
                    distill_turn_async(user_safe, asst_safe,
                                       domain=self._domain, mind_client=self._mind)
        except Exception as _dexc:
            logger.debug("voice distiller hook skipped: %s", _dexc)

    def last_ingest_id(self) -> Optional[str]:
        with self._last_ingest_lock:
            return self._last_ingest_id

    # -- Tools (explicit recall for power users) ---------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = [
            {
                "name": "vault_search",
                "description": (
                    "Search the Obsidian Vault for prior Q+A pairs, decisions, or learned facts. "
                    "Use when the user references something you've discussed before, or to confirm "
                    "a past decision before redoing analysis."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search keywords or question."},
                        "domain": {
                            "type": "string",
                            "description": "Optional tag filter (voice, code, chat, cron, general, crypto).",
                        },
                        "limit": {"type": "integer", "description": "Max results (default 8)."},
                    },
                    "required": ["query"],
                },
            },
        ]
        # Phase 6 — goal_tracker tool schemas (always created in __init__,
        # gated by HERMES_GOALS_ENABLED env var inside tool_schemas()).
        try:
            schemas.extend(self._goals.tool_schemas())
        except Exception as exc:
            logger.debug("goal_tracker tool_schemas failed: %s", exc)
        return schemas

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        import json as _json
        # Phase 6 — goal_tracker tools dispatched first.  Backends may be None
        # if VAULT_SECRET and OBSIDIAN_MIND_URL are both unset; GoalTracker
        # handlers return an error string in that case rather than raising.
        if tool_name.startswith("goal_"):
            return self._goals.handle(tool_name, args)
        if not (self._client or self._mind):
            return _json.dumps({"error": "vault not initialized"})
        if tool_name != "vault_search":
            return _json.dumps({"error": f"unknown tool {tool_name}"})

        query = args.get("query", "")
        if not query:
            return _json.dumps({"error": "query required"})
        domain = args.get("domain") or self._domain
        limit = int(args.get("limit", 8))

        # Prefer super-agent /memory when available; fall back to obsidian-mind.
        if self._client is not None:
            entries = self._client.export(tag=domain, limit=max(limit * 3, 24))
        elif self._mind is not None:
            mind_raw = self._mind.search(query, limit=max(limit * 2, 16))
            entries = mind_raw if isinstance(mind_raw, list) else []
        else:
            return _json.dumps({"results": [], "domain": domain})
        query_tokens = _tokenize(query)
        now = time.time()
        scored = []
        for e in entries:
            text = _entry_text(e)
            if not text:
                continue
            score = _score_entry(query_tokens, _tokenize(text), now - _entry_timestamp(e))
            if score > 0:
                scored.append({"score": round(score, 3), "content": text[:600]})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return _json.dumps({"results": scored[:limit], "domain": domain})

    def shutdown(self) -> None:
        for t in self._prefetch_threads:
            t.join(timeout=2.0)
        if self._client:
            self._client.close()


# ---------------------------------------------------------------------------
# Heuristic secret sniff — second line of defense behind redact_sensitive_text.
# ---------------------------------------------------------------------------

_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret|password|bearer|token|private[_-]?key|access[_-]?key)\s*[:=]",
    re.IGNORECASE,
)


def _looks_secret(text: str) -> bool:
    if not text:
        return False
    return bool(_SECRET_HINT_RE.search(text))
