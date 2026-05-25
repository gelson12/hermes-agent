"""Obsidian-Mind adapter — secondary recall + write target alongside the primary
super-agent /memory endpoint.

Obsidian-Mind (deployed at obsidian-mind-production.up.railway.app in
divine-contentment) is the user's enriched knowledge layer: notes, search,
indexing.  The primary super-agent /memory endpoint stays the raw event log;
Obsidian-Mind is the curated knowledge graph that benefits from semantic
search and structured notes.

This adapter is intentionally generic: every path, header, and auth scheme is
env-configurable so the same code works against whichever HTTP API
Obsidian-Mind actually exposes (we treat it as a black box).  If
OBSIDIAN_MIND_URL is unset, all methods no-op and the VaultProvider falls
back to its primary source — there is no hard dependency.

Env vars (all optional except URL):

  OBSIDIAN_MIND_URL              base URL, e.g. https://obsidian-mind-production.up.railway.app
  OBSIDIAN_MIND_AUTH_HEADER      full header line, e.g. "Authorization: Bearer xxx"
                                  or "X-API-Key: yyy".  Multiple headers separated by '||'.
  OBSIDIAN_MIND_SEARCH_PATH      default "/api/search"
  OBSIDIAN_MIND_SEARCH_PARAM     default "q"
  OBSIDIAN_MIND_INGEST_PATH      default "/api/notes"
  OBSIDIAN_MIND_INGEST_METHOD    default "POST"
  OBSIDIAN_MIND_INGEST_FIELD     default "content"        (form-field for the note body)
  OBSIDIAN_MIND_TIMEOUT          default 4 (seconds)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


def _parse_headers(raw: str) -> Dict[str, str]:
    """Parse 'K: V || K2: V2' into a dict.  Empty input -> {}."""
    headers: Dict[str, str] = {}
    if not raw:
        return headers
    for chunk in raw.split("||"):
        if ":" not in chunk:
            continue
        k, _, v = chunk.partition(":")
        k, v = k.strip(), v.strip()
        if k and v:
            headers[k] = v
    return headers


class ObsidianMindClient:
    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        *,
        search_path: str = "/api/search",
        search_param: str = "q",
        ingest_path: str = "/api/notes",
        ingest_method: str = "POST",
        ingest_field: str = "content",
        timeout: float = 4.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._headers = headers or {}
        self._search_path = search_path
        self._search_param = search_param
        self._ingest_path = ingest_path
        self._ingest_method = ingest_method.upper()
        self._ingest_field = ingest_field
        self._client = httpx.Client(timeout=timeout)

    def read_note(self, path: str) -> Optional[str]:
        """GET /api/notes/<path> — read raw note content. Returns text or None."""
        if not path:
            return None
        try:
            resp = self._client.get(
                f"{self._base_url}/api/notes/{path.lstrip('/')}",
                headers=self._headers,
            )
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict):
                for k in ("content", "text", "body"):
                    v = payload.get(k)
                    if isinstance(v, str):
                        return v
            if isinstance(payload, str):
                return payload
        except Exception as exc:
            logger.debug("obsidian-mind read_note failed (%s): %s", path, exc)
        return None

    def search(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        if not query:
            return []
        try:
            resp = self._client.get(
                f"{self._base_url}{self._search_path}",
                params={self._search_param: query, "limit": limit},
                headers=self._headers,
            )
            resp.raise_for_status()
            payload = resp.json() or {}
        except Exception as exc:
            logger.debug("obsidian-mind search failed: %s", exc)
            return []
        # Accept several shapes: {"results": [...]}, {"notes": [...]}, [...]
        if isinstance(payload, list):
            return payload[:limit]
        for k in ("results", "notes", "items", "matches"):
            v = payload.get(k) if isinstance(payload, dict) else None
            if isinstance(v, list):
                return v[:limit]
        return []

    def ingest(
        self,
        content: str,
        *,
        tags: Optional[List[str]] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        **extra: Any,
    ) -> Optional[Dict[str, Any]]:
        # `source` and other extra kwargs are accepted (for API parity with
        # the super-agent VaultClient) and folded into metadata.
        # Obsidian-mind exposes POST /api/notes/:path — the note path goes
        # in the URL.  When OBSIDIAN_MIND_INGEST_PATH ends with `/` (or
        # is the bare directory form) we synthesize a unique filename
        # under it so multiple writes don't overwrite each other.
        import time as _time, uuid as _uuid
        path = self._ingest_path
        if path.endswith("/") or "/notes" in path and not path.rstrip("/").rsplit("/", 1)[-1].endswith(".md"):
            stamp = _time.strftime("%Y%m%dT%H%M%S", _time.gmtime())
            short = _uuid.uuid4().hex[:8]
            tag_slug = (tags[1] if tags and len(tags) > 1 else "general").replace("/", "-")
            path = path.rstrip("/") + f"/hermes/{tag_slug}/{stamp}-{short}.md"

        # Fold source + any extra kwargs into the metadata dict so they survive.
        effective_meta: Dict[str, Any] = dict(metadata) if isinstance(metadata, dict) else {}
        if source:
            effective_meta.setdefault("source", source)
        for k, v in (extra or {}).items():
            effective_meta.setdefault(k, v)

        body: Dict[str, Any] = {self._ingest_field: content}
        if title:
            body["title"] = title
        if tags:
            body["tags"] = tags
        if effective_meta:
            body["metadata"] = effective_meta
        try:
            resp = self._client.request(
                self._ingest_method,
                f"{self._base_url}{path}",
                json=body,
                headers=self._headers,
            )
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {"ok": True}
        except Exception as exc:
            logger.debug("obsidian-mind ingest failed (path=%s): %s", path, exc)
            return None

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


def from_env() -> Optional[ObsidianMindClient]:
    """Construct from env, or None if OBSIDIAN_MIND_URL unset.

    Auth resolution: prefers OBSIDIAN_MIND_AUTH_HEADER; falls back to reusing
    VAULT_SECRET as X-Vault-Secret since obsidian-mind is deployed in the
    same divine-contentment project and shares the super-agent vault secret.
    """
    url = os.environ.get("OBSIDIAN_MIND_URL", "").strip()
    if not url:
        return None
    headers = _parse_headers(os.environ.get("OBSIDIAN_MIND_AUTH_HEADER", ""))
    if not headers:
        # Default: share the super-agent vault secret since obsidian-mind sits
        # behind the same divine-contentment auth wall.
        shared_secret = os.environ.get("VAULT_SECRET", "").strip()
        if shared_secret:
            headers = {"X-Vault-Secret": shared_secret}
    try:
        timeout = float(os.environ.get("OBSIDIAN_MIND_TIMEOUT", 4.0))
    except ValueError:
        timeout = 4.0
    return ObsidianMindClient(
        base_url=url,
        headers=headers,
        search_path=os.environ.get("OBSIDIAN_MIND_SEARCH_PATH", "/api/search"),
        search_param=os.environ.get("OBSIDIAN_MIND_SEARCH_PARAM", "q"),
        ingest_path=os.environ.get("OBSIDIAN_MIND_INGEST_PATH", "/api/notes"),
        ingest_method=os.environ.get("OBSIDIAN_MIND_INGEST_METHOD", "POST"),
        ingest_field=os.environ.get("OBSIDIAN_MIND_INGEST_FIELD", "content"),
        timeout=timeout,
    )
