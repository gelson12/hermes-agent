"""Tests for the prompt-evolver bootstrap that makes the loop actually run:
seed an empty baseline, resolve the active variant, and self-drive maybe_propose
from record_outcome (the n8n cron that used to call it is offline).

Before this, record_outcome was never called anywhere and no variant was ever
seeded, so the evolver could never aggregate outcomes or propose anything.
"""
import time

import agent.prompt_evolver as ev


def _reset():
    ev._CACHE.clear()
    ev._SEEDED.clear()
    ev._OUTCOME_COUNTS.clear()


def test_ensure_seed_creates_empty_baseline_when_none(monkeypatch):
    _reset()
    monkeypatch.setenv("HERMES_PROMPT_EVOLVER_ENABLED", "true")
    monkeypatch.setattr(ev, "_search_prompts", lambda *a, **k: [])   # no active variant yet
    writes = []
    monkeypatch.setattr(ev, "_record_prompt_variant",
                        lambda domain, text, **k: writes.append((domain, text, k)) or "v-seed-1")

    vid = ev.ensure_seed("voice-jarvis")
    assert vid == "v-seed-1"
    assert writes and writes[0][1] == ""                 # baseline body is EMPTY (neutral)
    assert writes[0][2].get("status") == "active"
    # current_prompt now returns "" (empty body) → nothing appended to the system prompt
    assert ev.current_prompt("voice-jarvis") == ""
    # idempotent: second call doesn't write again
    assert ev.ensure_seed("voice-jarvis") == "v-seed-1"
    assert len(writes) == 1


def test_active_variant_id_from_existing(monkeypatch):
    _reset()
    monkeypatch.setenv("HERMES_PROMPT_EVOLVER_ENABLED", "true")
    variant = {"content": "PROMPT_VARIANT [v9] domain=d status=active\n\nBe concise.",
               "metadata": {"variant_id": "v9", "status": "active", "domain": "d"}}
    monkeypatch.setattr(ev, "_search_prompts",
                        lambda domain, status_filter=None, limit=20: [variant] if status_filter in (None, "active") else [])
    assert ev.active_variant_id("d") == "v9"
    assert ev.current_prompt("d") == "Be concise."        # body extracted + appended


def test_record_outcome_self_drives_propose(monkeypatch):
    _reset()
    monkeypatch.setenv("HERMES_PROMPT_EVOLVER_ENABLED", "true")
    monkeypatch.setattr(ev, "_PROPOSE_EVERY", 2)

    class _Backend:
        def ingest(self, *a, **k):
            return None
        def close(self):
            return None

    monkeypatch.setattr(ev, "_mind_client", lambda: _Backend())
    monkeypatch.setattr(ev, "_vault_client", lambda: None)
    proposals = []
    monkeypatch.setattr(ev, "maybe_propose", lambda domain: proposals.append(domain))

    ev.record_outcome("d", "v1", True)        # count 1 → no propose
    ev.record_outcome("d", "v1", False)       # count 2 → propose fires (every 2)
    for _ in range(50):
        if proposals:
            break
        time.sleep(0.02)
    assert proposals == ["d"]


def test_list_and_promote_candidate(monkeypatch):
    _reset()
    monkeypatch.setenv("HERMES_PROMPT_EVOLVER_ENABLED", "true")
    cand = {"content": "PROMPT_VARIANT [c7] domain=d status=candidate samples=40 rate=0.300\n\n"
                       "When unsure, ask one clarifying question before answering.",
            "metadata": {"variant_id": "c7", "status": "candidate", "domain": "d",
                         "samples": 40, "success_rate": 0.3}}
    active = {"content": "PROMPT_VARIANT [a1] domain=d status=active\n\nBe concise.",
              "metadata": {"variant_id": "a1", "status": "active", "domain": "d"}}

    def _search(domain, status_filter=None, limit=20):
        if status_filter == "candidate":
            return [cand]
        if status_filter == "active":
            return [active]
        return [cand, active]
    monkeypatch.setattr(ev, "_search_prompts", _search)
    # list shows only the candidate
    lst = ev.list_candidates("d")
    assert len(lst) == 1 and lst[0]["variant_id"] == "c7"
    assert "clarifying question" in lst[0]["text"]
    assert ev.candidate_count("d") == 1

    writes = []
    monkeypatch.setattr(ev, "_record_prompt_variant",
                        lambda domain, text, **k: writes.append((k.get("status"), k.get("variant_id"), text)) or k.get("variant_id"))
    res = ev.promote_candidate("d", "c7")
    assert res["ok"] is True and res["variant_id"] == "c7"
    # archived the old active, promoted the candidate to active
    statuses = {(s, vid) for (s, vid, _t) in writes}
    assert ("archived", "a1") in statuses
    assert ("active", "c7") in statuses
    # unknown candidate → graceful error
    assert ev.promote_candidate("d", "nope")["ok"] is False


def test_disabled_is_inert(monkeypatch):
    _reset()
    monkeypatch.setenv("HERMES_PROMPT_EVOLVER_ENABLED", "false")
    assert ev.ensure_seed("d") is None
    assert ev.active_variant_id("d") is None
    assert ev.current_prompt("d") == ""
    st = ev.evolver_state("d")
    assert st["enabled"] is False and st["chars"] == 0
