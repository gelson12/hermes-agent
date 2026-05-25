"""Vault-aware model router.

Picks a cheap, mid, or big LLM for an incoming turn based on how confidently
the Obsidian Vault already covers the topic. Wired into
gateway.platforms.api_server._handle_chat_completions (and the /v1/responses
twin) so OpenJarvis, friday_jarvis2, and any other Hermes consumer
progressively use cheaper models as the vault matures.

Decision tree (industry-grade defaults from quant signal routing literature,
adapted for chat):

  maturity[domain].mature -> cheap   (cron-tuned, see Phase 4)
  confidence >= 0.60 and hits >= 1 -> cheap
  confidence >= 0.30              -> mid
  confidence < 0.30 or hits == 0  -> big   (no change from default)

The cheap/mid/big tiers are configurable via env vars so the user can wire
Groq/DeepSeek/Cerebras/Gemini-Flash etc. to taste without code changes:

  VAULT_ROUTE_CHEAP_MODEL    default: "groq/llama-3.3-70b-versatile"
  VAULT_ROUTE_MID_MODEL      default: "openrouter/google/gemini-2.5-flash"
  VAULT_ROUTE_BIG_MODEL      default: "" -> keep gateway default (Claude/etc.)

Maturity table is in-process state, populated by the n8n cron from Phase 4
via PUT /v1/routing/maturity. Survives restarts when persisted to
$HERMES_HOME/vault_routing.json.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier resolution
# ---------------------------------------------------------------------------

DEFAULT_CHEAP_MODEL = "groq/llama-3.3-70b-versatile"
DEFAULT_MID_MODEL = "openrouter/google/gemini-2.5-flash"
DEFAULT_BIG_MODEL = ""  # empty -> caller keeps its gateway default


def _tier_model(tier: str, fallback: str) -> str:
    env_key = {
        "cheap": "VAULT_ROUTE_CHEAP_MODEL",
        "mid": "VAULT_ROUTE_MID_MODEL",
        "big": "VAULT_ROUTE_BIG_MODEL",
    }.get(tier, "")
    override = os.environ.get(env_key, "").strip() if env_key else ""
    if override:
        return override
    defaults = {"cheap": DEFAULT_CHEAP_MODEL, "mid": DEFAULT_MID_MODEL, "big": DEFAULT_BIG_MODEL}
    return defaults.get(tier, "") or fallback


# ---------------------------------------------------------------------------
# Per-domain maturity table (Phase 4 cron updates this)
# ---------------------------------------------------------------------------

@dataclass
class DomainMaturity:
    """Per-(domain, model) success-rate snapshot updated by the n8n cron."""
    domain: str
    model: str = ""
    samples: int = 0
    success_rate: float = 0.0
    mature: bool = False
    last_update: float = 0.0


class MaturityTable:
    """Thread-safe maturity store, persisted to disk so restarts don't lose state."""

    def __init__(self, persist_path: Optional[Path] = None):
        self._lock = threading.Lock()
        self._table: Dict[str, DomainMaturity] = {}
        self._persist_path = persist_path
        self._load()

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for k, v in (raw or {}).items():
                self._table[k] = DomainMaturity(**v)
        except Exception as exc:
            logger.debug("vault_router maturity load failed: %s", exc)

    def _persist(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps({k: asdict(v) for k, v in self._table.items()}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("vault_router maturity persist failed: %s", exc)

    def get(self, domain: str) -> DomainMaturity:
        with self._lock:
            return self._table.get(domain, DomainMaturity(domain=domain))

    def update(self, domain: str, model: str, samples: int, success_rate: float, mature: bool) -> None:
        with self._lock:
            self._table[domain] = DomainMaturity(
                domain=domain,
                model=model,
                samples=samples,
                success_rate=success_rate,
                mature=mature,
                last_update=time.time(),
            )
            self._persist()

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: asdict(v) for k, v in self._table.items()}


_DEFAULT_TABLE: Optional[MaturityTable] = None
_TABLE_LOCK = threading.Lock()


def get_maturity_table() -> MaturityTable:
    """Singleton, persisted at $HERMES_HOME/vault_routing.json."""
    global _DEFAULT_TABLE
    with _TABLE_LOCK:
        if _DEFAULT_TABLE is None:
            try:
                from hermes_constants import get_hermes_home
                persist_path = get_hermes_home() / "vault_routing.json"
            except Exception:
                persist_path = None
            _DEFAULT_TABLE = MaturityTable(persist_path=persist_path)
        return _DEFAULT_TABLE


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    tier: str               # cheap | mid | big
    model: str              # actual model string (may be empty -> caller default)
    confidence: float       # 0..1 vault confidence
    hits: int               # vault hit count
    domain: str             # voice/code/chat/cron/general/...
    mature: bool            # whether the cron has tagged (domain,model) as mature
    reason: str             # short rationale for logs

    def as_log(self) -> Dict[str, Any]:
        return {
            "event": "hermes.routing.decision",
            **asdict(self),
        }


def select_model(
    *,
    query: str,
    vault_confidence: float,
    vault_hits: int,
    domain: str,
    default_model: str,
) -> RoutingDecision:
    """Pick a model for the incoming turn given vault state.

    Pure function — no I/O. ``default_model`` is what the gateway would use
    without any override (kept for the big tier when ``VAULT_ROUTE_BIG_MODEL``
    is unset).
    """
    table = get_maturity_table()
    maturity = table.get(domain)

    if maturity.mature and maturity.model:
        return RoutingDecision(
            tier="cheap",
            model=maturity.model,
            confidence=vault_confidence,
            hits=vault_hits,
            domain=domain,
            mature=True,
            reason=f"domain {domain} marked mature with {maturity.samples} samples @ {maturity.success_rate:.0%}",
        )

    if vault_confidence >= 0.60 and vault_hits >= 1:
        return RoutingDecision(
            tier="cheap",
            model=_tier_model("cheap", default_model),
            confidence=vault_confidence,
            hits=vault_hits,
            domain=domain,
            mature=False,
            reason=f"strong vault hit (conf={vault_confidence:.2f})",
        )

    if vault_confidence >= 0.30:
        return RoutingDecision(
            tier="mid",
            model=_tier_model("mid", default_model),
            confidence=vault_confidence,
            hits=vault_hits,
            domain=domain,
            mature=False,
            reason=f"moderate vault hit (conf={vault_confidence:.2f})",
        )

    return RoutingDecision(
        tier="big",
        model=_tier_model("big", default_model) or default_model,
        confidence=vault_confidence,
        hits=vault_hits,
        domain=domain,
        mature=False,
        reason="cold or weak vault — use default",
    )


# ---------------------------------------------------------------------------
# Stateless vault probe for gateway-level routing (no AIAgent needed)
# ---------------------------------------------------------------------------

# Reused from provider.py logic — kept here so the gateway can call this
# without importing the memory plugin (which only loads when memory.provider
# is set in config).  Keeps the routing layer working even if the user wants
# routing but not full memory injection.

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


def _entry_score(query_tokens: List[str], entry_text: str, entry_age_seconds: float) -> float:
    if not query_tokens or not entry_text:
        return 0.0
    entry_tokens = _tokenize(entry_text)
    if not entry_tokens:
        return 0.0
    freq: Dict[str, int] = {}
    for t in entry_tokens:
        freq[t] = freq.get(t, 0) + 1
    overlap = 0.0
    for qt in set(query_tokens):
        if qt in freq:
            overlap += 1.0 / math.log(1 + freq[qt] + 1)
    overlap_norm = overlap / max(len(set(query_tokens)), 1)
    age_days = max(entry_age_seconds, 0) / 86400.0
    recency = math.exp(-age_days / 30.0)
    return max(0.0, min(1.0, overlap_norm * recency))


def probe_vault(query: str, domain: str, limit: int = 24) -> Tuple[float, int]:
    """Stateless vault probe — returns (top_confidence, hit_count).

    Used by the gateway to make a routing decision when there's no active
    VaultProvider instance to ask (e.g. first call from a fresh client before
    the memory provider has prefetched). Safe to call from any thread; the
    underlying httpx client is per-call.
    """
    from datetime import datetime
    try:
        from plugins.memory.vault.client import from_env as _client
    except Exception:
        return 0.0, 0

    c = _client()
    if c is None:
        return 0.0, 0
    try:
        entries = c.export(tag=domain, limit=limit)
    finally:
        c.close()
    if not entries:
        return 0.0, 0

    qt = _tokenize(query)
    if not qt:
        return 0.0, 0

    now = time.time()
    best = 0.0
    hits = 0
    for e in entries:
        text = ""
        for k in ("content", "text", "body", "summary"):
            v = e.get(k) if isinstance(e, dict) else None
            if isinstance(v, str) and v.strip():
                text = v
                break
        if not text:
            continue
        age = now
        for k in ("created_at", "createdAt", "timestamp", "ts"):
            v = e.get(k) if isinstance(e, dict) else None
            if isinstance(v, (int, float)):
                age = now - float(v)
                break
            if isinstance(v, str):
                try:
                    age = now - datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
                    break
                except Exception:
                    continue
        score = _entry_score(qt, text, age)
        if score > 0:
            hits += 1
            best = max(best, score)
    return best, hits
