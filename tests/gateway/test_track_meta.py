"""Unit tests for gateway.platforms.track_meta — viewer-metadata for /t pings.

Pure helpers (no network): IP extraction, private-IP guard, bot detection, device
labelling, owner-self-IP flag, and ping composition. geo_lookup is only checked for
its fail-soft behaviour on non-public IPs (no outbound call).
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gateway", "platforms"))

import track_meta as tm  # noqa: E402


def test_client_ip_prefers_forwarded_first_hop():
    assert tm.client_ip("203.0.113.7, 10.0.0.1", "10.0.0.9") == "203.0.113.7"
    assert tm.client_ip("", "198.51.100.5") == "198.51.100.5"
    assert tm.client_ip("", "") == ""


def test_is_private_ip():
    assert tm.is_private_ip("10.0.0.1") is True
    assert tm.is_private_ip("8.8.8.8") is False
    assert tm.is_private_ip("not-an-ip") is True


def test_is_preview_bot():
    assert tm.is_preview_bot("facebookexternalhit/1.1") is True
    assert tm.is_preview_bot("TelegramBot (like TwitterBot)") is True
    assert tm.is_preview_bot(
        "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 Chrome/120 Mobile Safari/537.36") is False
    assert tm.is_preview_bot(
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) AppleWebKit/605 Safari/604") is False


def test_device_label():
    assert tm.device_label(
        "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 Chrome/120 Mobile Safari/537.36"
    ) == "Android phone · Chrome"
    assert tm.device_label(
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) AppleWebKit/605 Version/17 Safari/604"
    ) == "iPhone · Safari"
    assert tm.device_label(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Edg/120"
    ) == "Windows · Edge"
    assert tm.device_label("") == ""


def test_is_self_ip(monkeypatch):
    monkeypatch.setenv("BRIDGE_TRACK_SELF_IPS", "203.0.113.7, 198.51.100.5")
    assert tm.is_self_ip("203.0.113.7") is True
    assert tm.is_self_ip("8.8.8.8") is False
    monkeypatch.delenv("BRIDGE_TRACK_SELF_IPS", raising=False)
    assert tm.is_self_ip("203.0.113.7") is False


def test_compose_ping():
    bot = tm.compose_ping("S&E Roofing", "viewed", is_bot=True)
    assert "automated link-preview" in bot and "S&E Roofing" in bot
    human = tm.compose_ping("S&E Roofing", "viewed", geo="Maidstone, Kent",
                            device="Android phone · Chrome")
    assert "Maidstone, Kent" in human and "Android phone · Chrome" in human
    selfv = tm.compose_ping("S&E Roofing", "viewed", is_self=True, geo="London")
    assert "your own device" in selfv
    bare = tm.compose_ping("Biz", "clicked through to")
    assert bare.endswith("their website preview.")


def test_geo_lookup_failsoft_on_non_public():
    assert asyncio.run(tm.geo_lookup("10.0.0.1")) == ""
    assert asyncio.run(tm.geo_lookup("")) == ""
