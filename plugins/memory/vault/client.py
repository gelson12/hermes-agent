"""HTTP client for the Obsidian Vault gateway on Railway.

API contract (inferred from existing n8n crypto outcome_tracker workflow):

  GET  /memory/export?tag=<str>&limit=<int>
    headers: X-Vault-Secret: <secret>
    returns: {"memories": [{"content": str, "tags": [...], "source": str, "created_at": str, ...}]}

  POST /memory/ingest
    headers: X-Vault-Secret: <secret>
    form:    content=<str>, tags=<json-array-string>, source=<str>

  PATCH /memory/<id>   (optional, used by /v1/feedback to attach success flag)
    headers: X-Vault-Secret: <secret>
    body:    {"metadata": {...}}

All calls are best-effort: vault unreachable -> log + return empty list / drop the
write. Vault is an enhancement layer, not a hard dependency.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://super-agent-production.up.railway.app"
DEFAULT_TIMEOUT = 4.0


class VaultClient:
    def __init__(self, secret: str, base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT):
        self._secret = secret
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        return {"X-Vault-Secret": self._secret}

    def export(self, tag: str = "", limit: int = 24, source: str = "") -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        if tag:
            params["tag"] = tag
        if source:
            params["source"] = source
        try:
            resp = self._client.get(
                f"{self._base_url}/memory/export",
                params=params,
                headers=self._headers(),
            )
            resp.raise_for_status()
            payload = resp.json() or {}
            memories = payload.get("memories") or payload.get("items") or []
            return memories if isinstance(memories, list) else []
        except Exception as exc:
            logger.debug("vault export failed (tag=%s): %s", tag, exc)
            return []

    def ingest(
        self,
        content: str,
        *,
        tags: Optional[List[str]] = None,
        source: str = "hermes-vault",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        data = {
            "content": content,
            "tags": json.dumps(tags or []),
            "source": source,
        }
        if metadata:
            data["metadata"] = json.dumps(metadata)
        try:
            resp = self._client.post(
                f"{self._base_url}/memory/ingest",
                data=data,
                headers=self._headers(),
            )
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {"ok": True}
        except Exception as exc:
            logger.debug("vault ingest failed: %s", exc)
            return None

    def patch_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> bool:
        """Attach metadata (e.g. success_flag from Phase 3 feedback) to an existing entry."""
        try:
            resp = self._client.patch(
                f"{self._base_url}/memory/{memory_id}",
                json={"metadata": metadata},
                headers=self._headers(),
            )
            return resp.is_success
        except Exception as exc:
            logger.debug("vault patch failed (id=%s): %s", memory_id, exc)
            return False

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


def from_env() -> Optional[VaultClient]:
    """Construct a VaultClient from env vars, or None if not configured.

    Env vars:
        VAULT_SECRET      (required)
        VAULT_BASE_URL    (optional, default super-agent-production.up.railway.app)
        VAULT_TIMEOUT     (optional, seconds, default 4)
    """
    secret = os.environ.get("VAULT_SECRET", "").strip()
    if not secret:
        return None
    base_url = os.environ.get("VAULT_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    try:
        timeout = float(os.environ.get("VAULT_TIMEOUT", DEFAULT_TIMEOUT))
    except ValueError:
        timeout = DEFAULT_TIMEOUT
    return VaultClient(secret=secret, base_url=base_url, timeout=timeout)
