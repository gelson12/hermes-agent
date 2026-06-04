"""email_remote — Gmail + Outlook(Graph) for Hermes server-side agents.

Mirrors the OpenJarvis-Avengers worker's proven ``email_direct`` approach
(refresh-token -> access-token -> REST), so a spawned Hermes agent can read,
send, and trash mail across the same four mailboxes in the background, with NO
dependency on n8n. Self-contained (sync httpx), fail-soft.

Accounts: personal | bridge | remote (Gmail) and outlook (Microsoft Graph).
Credentials are the same env vars the worker uses:
  Gmail:   GMAIL_OAUTH_APP_CLIENT_ID + GMAIL_OATH_CLIENT_SECRET
           (or GMAIL_Client_ID + GMAIL_Client_secret)
           GMAIL_REFRESH_TOKEN_{PERSONAL,BRIDGE,REMOTE} (or GMAIL_REFRESH_TOKEN)
  Outlook: OUTLOOK_CLIENT_ID + OUTLOOK_CLIENT_SECRET + OUTLOOK_REFRESH_TOKEN
           (+ OUTLOOK_Access_Token_URL)

Safety: email_search is read-only. email_send / email_trash mutate the mailbox,
so they declare requires_approval=True semantics by being gated through the
normal Hermes approval flow (the agent must get approval before they run).
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from tools.registry import registry

logger = logging.getLogger(__name__)
_TIMEOUT = 12.0


# ── credentials / routing ───────────────────────────────────────────────────
def _g_pairs() -> List[tuple]:
    pairs = []
    for cid_env, sec_env in (("GMAIL_Client_ID", "GMAIL_Client_secret"),
                             ("GMAIL_OAUTH_APP_CLIENT_ID", "GMAIL_OATH_CLIENT_SECRET")):
        cid = (os.environ.get(cid_env) or "").strip()
        sec = (os.environ.get(sec_env) or "").strip()
        if cid and sec and (cid, sec) not in pairs:
            pairs.append((cid, sec))
    return pairs


def _g_refresh(account: str) -> str:
    acct = (account or "").lower()
    env = {"bridge": "GMAIL_REFRESH_TOKEN_BRIDGE", "remote": "GMAIL_REFRESH_TOKEN_REMOTE",
           "personal": "GMAIL_REFRESH_TOKEN_PERSONAL"}.get(acct)
    if env and os.environ.get(env, "").strip():
        return os.environ[env].strip()
    if acct in ("all", "gmail", "google", ""):
        t = os.environ.get("GMAIL_REFRESH_TOKEN_PERSONAL", "").strip()
        if t:
            return t
    return os.environ.get("GMAIL_REFRESH_TOKEN", "").strip()


def _is_outlook(account: str) -> bool:
    return (account or "").lower() == "outlook"


def _graph_creds() -> tuple:
    cid = (os.environ.get("OUTLOOK_CLIENT_ID") or os.environ.get("MS_GRAPH_CLIENT_ID") or "").strip()
    sec = (os.environ.get("OUTLOOK_CLIENT_SECRET") or os.environ.get("MS_GRAPH_CLIENT_SECRET") or "").strip()
    rt = (os.environ.get("OUTLOOK_REFRESH_TOKEN") or os.environ.get("MS_GRAPH_REFRESH_TOKEN") or "").strip()
    tenant = (os.environ.get("MS_GRAPH_TENANT_ID") or "common").strip() or "common"
    url = (os.environ.get("OUTLOOK_Access_Token_URL")
           or f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token").strip()
    return cid, sec, rt, url


def email_remote_available(_args=None, **_kw) -> bool:
    """check_fn — expose the tools only when SOME mailbox is configured."""
    if _graph_creds()[0] and _graph_creds()[2]:
        return True
    return bool(_g_pairs() and (_g_refresh("personal") or _g_refresh("all")
                                or _g_refresh("bridge") or _g_refresh("remote")))


# ── token + HTTP ────────────────────────────────────────────────────────────
def _google_token(account: str) -> Optional[str]:
    rt = _g_refresh(account)
    if not rt:
        return None
    for cid, sec in _g_pairs():
        try:
            r = httpx.post("https://oauth2.googleapis.com/token", timeout=_TIMEOUT, data={
                "client_id": cid, "client_secret": sec,
                "refresh_token": rt, "grant_type": "refresh_token"})
            at = r.json().get("access_token") if r.status_code < 300 else None
            if at:
                return at
        except Exception as exc:  # noqa: BLE001
            logger.debug("google token failed: %s", exc)
    return None


def _graph_token() -> Optional[str]:
    cid, sec, rt, url = _graph_creds()
    if not (cid and rt):
        return None
    try:
        data = {"client_id": cid, "refresh_token": rt, "grant_type": "refresh_token"}
        if sec:
            data["client_secret"] = sec
        r = httpx.post(url, timeout=_TIMEOUT, data=data)
        return r.json().get("access_token") if r.status_code < 300 else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("graph token failed: %s", exc)
        return None


def _rfc822(to: str, subject: str, body: str) -> str:
    msg = (f"To: {to}\r\nSubject: {subject}\r\n"
           f"Content-Type: text/plain; charset=UTF-8\r\n\r\n{body}")
    return base64.urlsafe_b64encode(msg.encode("utf-8")).decode("ascii")


# ── handlers ────────────────────────────────────────────────────────────────
def handle_email_search(args: Dict[str, Any], **_kw) -> str:
    account = (args.get("account") or "all").lower()
    query = args.get("query") or ""
    folder = (args.get("folder") or "inbox").lower()
    try:
        if _is_outlook(account):
            tok = _graph_token()
            if not tok:
                return "Outlook is not configured."
            params = {"$top": "10", "$select": "from,subject,receivedDateTime,bodyPreview",
                      "$orderby": "receivedDateTime desc"}
            if query:
                params = {"$top": "10", "$search": f'"{query}"',
                          "$select": "from,subject,receivedDateTime,bodyPreview"}
            r = httpx.get("https://graph.microsoft.com/v1.0/me/messages", timeout=_TIMEOUT,
                          headers={"Authorization": f"Bearer {tok}"}, params=params)
            msgs = r.json().get("value", []) if r.status_code < 300 else []
            out = [{"id": m.get("id"),
                    "from": ((m.get("from") or {}).get("emailAddress") or {}).get("address", ""),
                    "subject": m.get("subject", ""), "snippet": (m.get("bodyPreview") or "")[:160]}
                   for m in msgs[:10]]
        else:
            tok = _google_token(account)
            if not tok:
                return f"The {account} Gmail account is not configured."
            label = "SPAM" if folder == "junk" else "INBOX"
            r = httpx.get("https://gmail.googleapis.com/gmail/v1/users/me/messages", timeout=_TIMEOUT,
                          headers={"Authorization": f"Bearer {tok}"},
                          params={"q": query, "maxResults": "10", "labelIds": label})
            ids = [m["id"] for m in r.json().get("messages", [])][:10] if r.status_code < 300 else []
            out = []
            for mid in ids:
                mr = httpx.get(f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{mid}",
                               timeout=_TIMEOUT, headers={"Authorization": f"Bearer {tok}"},
                               params={"format": "metadata",
                                       "metadataHeaders": ["From", "Subject", "Date"]})
                if mr.status_code >= 300:
                    continue
                h = {x["name"].lower(): x["value"]
                     for x in (mr.json().get("payload", {}) or {}).get("headers", [])}
                out.append({"id": mid, "from": h.get("from", ""),
                            "subject": h.get("subject", ""),
                            "snippet": (mr.json().get("snippet") or "")[:160]})
        if not out:
            return f"No matching messages in {account}."
        lines = [f"{len(out)} message(s) in {account}:"]
        for m in out:
            lines.append(f"- [{m['id']}] {m['from']} — {m['subject']}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("email_search failed: %s", exc)
        return f"Email search failed: {exc}"


def handle_email_send(args: Dict[str, Any], **_kw) -> str:
    account = (args.get("account") or "personal").lower()
    to, subject, body = args.get("to", ""), args.get("subject", ""), args.get("body", "")
    if not to or "@" not in to:
        return "A valid recipient address is required."
    try:
        if _is_outlook(account):
            tok = _graph_token()
            if not tok:
                return "Outlook is not configured."
            r = httpx.post("https://graph.microsoft.com/v1.0/me/sendMail", timeout=_TIMEOUT,
                           headers={"Authorization": f"Bearer {tok}"},
                           json={"message": {"subject": subject,
                                             "body": {"contentType": "Text", "content": body},
                                             "toRecipients": [{"emailAddress": {"address": to}}]},
                                 "saveToSentItems": True})
            return f"Sent from outlook to {to}." if r.status_code in (200, 202) else f"Send failed ({r.status_code})."
        tok = _google_token(account)
        if not tok:
            return f"The {account} Gmail account is not configured."
        r = httpx.post("https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                       timeout=_TIMEOUT, headers={"Authorization": f"Bearer {tok}"},
                       json={"raw": _rfc822(to, subject, body)})
        return f"Sent from {account} to {to}." if r.status_code == 200 else f"Send failed ({r.status_code})."
    except Exception as exc:  # noqa: BLE001
        logger.warning("email_send failed: %s", exc)
        return f"Send failed: {exc}"


def handle_email_trash(args: Dict[str, Any], **_kw) -> str:
    account = (args.get("account") or "all").lower()
    mid = args.get("message_id", "")
    if not mid:
        return "A message_id (from email_search) is required."
    try:
        if _is_outlook(account):
            tok = _graph_token()
            if not tok:
                return "Outlook is not configured."
            r = httpx.post(f"https://graph.microsoft.com/v1.0/me/messages/{mid}/move",
                           timeout=_TIMEOUT, headers={"Authorization": f"Bearer {tok}"},
                           json={"destinationId": "deleteditems"})
            return "Moved to Deleted Items." if r.status_code in (200, 201) else f"Trash failed ({r.status_code})."
        tok = _google_token(account)
        if not tok:
            return f"The {account} Gmail account is not configured."
        r = httpx.post(f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{mid}/trash",
                       timeout=_TIMEOUT, headers={"Authorization": f"Bearer {tok}"})
        return "Moved to Trash (recoverable)." if r.status_code == 200 else f"Trash failed ({r.status_code})."
    except Exception as exc:  # noqa: BLE001
        logger.warning("email_trash failed: %s", exc)
        return f"Trash failed: {exc}"


# ── schemas ─────────────────────────────────────────────────────────────────
_ACCOUNT_DESC = ("Which mailbox: 'personal', 'bridge', or 'remote' (Gmail), or "
                 "'outlook'. Defaults to 'personal'/'all'.")

registry.register(
    name="email_search", toolset="email-remote", is_async=False,
    check_fn=email_remote_available,
    schema={"name": "email_search",
            "description": "Search/list recent emails in a mailbox (read-only). Returns ids for email_trash.",
            "parameters": {"type": "object", "properties": {
                "account": {"type": "string", "description": _ACCOUNT_DESC},
                "query": {"type": "string", "description": "Gmail/Graph search query, e.g. 'is:unread', 'from:hsbc'. Empty for recent."},
                "folder": {"type": "string", "enum": ["inbox", "junk"], "description": "inbox (default) or junk."}}}},
    handler=lambda args, **kw: handle_email_search(args, **kw),
    description="Read/search a mailbox (personal/bridge/remote Gmail or Outlook).", emoji="📥")

registry.register(
    name="email_send", toolset="email-remote", is_async=False,
    check_fn=email_remote_available,
    schema={"name": "email_send",
            "description": "Send an email. MUTATES — requires approval. Reaches real recipients.",
            "parameters": {"type": "object", "properties": {
                "account": {"type": "string", "description": _ACCOUNT_DESC},
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string"}, "body": {"type": "string"}},
                "required": ["to"]}},
    handler=lambda args, **kw: handle_email_send(args, **kw),
    description="Send an email from a mailbox (outward-facing — gate behind approval).", emoji="📤")

registry.register(
    name="email_trash", toolset="email-remote", is_async=False,
    check_fn=email_remote_available,
    schema={"name": "email_trash",
            "description": "Move a message to Trash/Deleted (recoverable). MUTATES — requires approval. Use message_id from email_search.",
            "parameters": {"type": "object", "properties": {
                "account": {"type": "string", "description": _ACCOUNT_DESC},
                "message_id": {"type": "string", "description": "Id from a prior email_search."}},
                "required": ["message_id"]}},
    handler=lambda args, **kw: handle_email_trash(args, **kw),
    description="Trash a message (recoverable — gate behind approval).", emoji="🗑️")


__all__ = ["handle_email_search", "handle_email_send", "handle_email_trash",
           "email_remote_available"]
