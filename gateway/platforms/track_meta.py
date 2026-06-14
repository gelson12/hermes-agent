"""Viewer metadata for outreach tracking pings (the Hermes ``/t`` endpoint).

Pure, dependency-light helpers so the "X viewed their preview" Telegram ping can say
WHERE (coarse IP geolocation) and WITH WHAT (device), flag automated link-preview
bots, and flag the owner's own devices. Everything is fail-soft: any error yields a
neutral label, and nothing here ever touches the ``/t`` redirect/pixel response.

``aiohttp`` is imported lazily inside :func:`geo_lookup` so the rest of the module
imports with only the stdlib (keeps the pure helpers trivially unit-testable).
"""
from __future__ import annotations

import ipaddress
import os

# Known link-preview / crawler User-Agents (plus broad catch-alls that never appear
# in a normal browser UA). These fetch the page to build an unfurl card — not a human.
_PREVIEW_BOTS = (
    "facebookexternalhit", "facebot", "telegrambot", "whatsapp", "twitterbot",
    "slackbot", "slack-imgproxy", "linkedinbot", "discordbot", "skypeuripreview",
    "redditbot", "pinterest", "embedly", "vkshare", "applebot", "googlebot",
    "bingbot", "yandexbot", "duckduckbot", "metainspector", "google-inspectiontool",
    "bot", "crawler", "spider",
)


def client_ip(xff: str, remote: str = "") -> str:
    """Real client IP: first hop of ``X-Forwarded-For`` (Railway's edge proxy sets it),
    else the socket peer. ``""`` if neither is usable."""
    ip = (xff or "").split(",")[0].strip()
    if not ip:
        ip = (remote or "").strip()
    return ip


def is_private_ip(ip: str) -> bool:
    """True for private/loopback/link-local or unparseable IPs (not geolocatable)."""
    try:
        return ipaddress.ip_address(ip).is_private
    except Exception:  # noqa: BLE001
        return True


def is_preview_bot(ua: str) -> bool:
    u = (ua or "").lower()
    return bool(u) and any(b in u for b in _PREVIEW_BOTS)


def device_label(ua: str) -> str:
    """Coarse ``"platform · browser"`` from a User-Agent. ``""`` if unknown."""
    u = (ua or "").lower()
    if not u:
        return ""
    if "edg/" in u or "edga" in u:
        br = "Edge"
    elif "opr/" in u or "opera" in u:
        br = "Opera"
    elif "firefox" in u or "fxios" in u:
        br = "Firefox"
    elif "chrome" in u or "crios" in u:
        br = "Chrome"
    elif "safari" in u:
        br = "Safari"
    else:
        br = ""
    if "iphone" in u:
        plat = "iPhone"
    elif "ipad" in u:
        plat = "iPad"
    elif "android" in u:
        plat = "Android " + ("phone" if "mobile" in u else "tablet")
    elif "windows" in u:
        plat = "Windows"
    elif "macintosh" in u or "mac os" in u:
        plat = "Mac"
    elif "linux" in u:
        plat = "Linux"
    else:
        plat = "device"
    return f"{plat} · {br}" if br else plat


def is_self_ip(ip: str) -> bool:
    """True if ``ip`` is one of the owner's own devices (env ``BRIDGE_TRACK_SELF_IPS``,
    comma-separated) — so a self-view is flagged, not mistaken for a lead."""
    if not ip:
        return False
    sel = {x.strip() for x in (os.getenv("BRIDGE_TRACK_SELF_IPS") or "").split(",") if x.strip()}
    return ip in sel


def compose_ping(business: str, how: str, *, geo: str = "", device: str = "",
                 is_bot: bool = False, is_self: bool = False) -> str:
    """Build the Telegram ping text with whatever viewer context we have."""
    base = f"\U0001F525 {business} just {how} their website preview."
    if is_bot:
        return base + "\n⚠️ automated link-preview fetch (not a human view)"
    bits = []
    if is_self:
        bits.append("\U0001F464 your own device")
    if geo:
        bits.append("\U0001F4CD ~" + geo)
    if device:
        bits.append(device)
    return base + ("\n" + " · ".join(bits) if bits else "")


async def geo_lookup(ip: str, timeout: float = 3.0) -> str:
    """Coarse ``"City, Region"`` (falling back to country) from a public IP via a free,
    keyless service. ``""`` for private/unknown IPs or any error. Lazy-imports aiohttp."""
    if not ip or is_private_ip(ip):
        return ""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.get(f"https://ipwho.is/{ip}",
                                timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                d = await r.json(content_type=None)
        if not isinstance(d, dict) or not d.get("success"):
            return ""
        parts = [p for p in (d.get("city"), d.get("region"), d.get("country")) if p]
        return ", ".join(parts[:2]) if parts else ""
    except Exception:  # noqa: BLE001
        return ""
