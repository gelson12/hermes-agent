"""Tests for the passthrough memory bridge.

The tool-calling passthrough bypasses the AIAgent (it must return OpenAI tool_calls),
so it also bypassed the vault self-improvement loop for ALL tool-bearing (voice)
traffic. This bridge re-attaches recall + write-back. These tests pin the pure
helpers and the provider orchestration (with a stub backend)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gateway.platforms import passthrough_memory as p  # noqa: E402


def test_last_user_text():
    assert p.last_user_text({"messages": [{"role": "system", "content": "s"},
                                          {"role": "user", "content": "hello"}]}) == "hello"
    assert p.last_user_text({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi"}, {"type": "image_url"}]}]}) == "hi"
    assert p.last_user_text({"messages": [{"role": "user", "content": "first"},
                                          {"role": "assistant", "content": "x"},
                                          {"role": "user", "content": "second"}]}) == "second"
    assert p.last_user_text({"messages": []}) == ""


def test_inject_recall_folds_into_system_without_mutating():
    msgs = [{"role": "system", "content": "base"}, {"role": "user", "content": "q"}]
    out = p.inject_recall(msgs, "[Vault Recall]\n- foo")
    assert out[0]["role"] == "system"
    assert "Vault Recall" in out[0]["content"] and "base" in out[0]["content"]
    assert msgs[0]["content"] == "base"          # input not mutated
    out2 = p.inject_recall([{"role": "user", "content": "q"}], "recallX")
    assert out2[0]["role"] == "system" and "recallX" in out2[0]["content"]
    assert p.inject_recall(msgs, "") is msgs       # empty recall → unchanged


def test_inject_context_folds_recall_and_goals():
    msgs = [{"role": "system", "content": "base"}, {"role": "user", "content": "q"}]
    out = p.inject_context(msgs, recall="[Vault Recall]\n- foo", goals="# Active Goals\n- [a1] (high) ship it")
    assert "Vault Recall" in out[0]["content"]
    assert "Active Goals" in out[0]["content"] and "ship it" in out[0]["content"]
    assert msgs[0]["content"] == "base"                    # not mutated
    assert p.inject_context(msgs) is msgs                  # neither → unchanged
    # goals only, no system message → a system message is created
    out2 = p.inject_context([{"role": "user", "content": "q"}], goals="# Active Goals\n- [b2] (normal) learn")
    assert out2[0]["role"] == "system" and "Active Goals" in out2[0]["content"]


def test_inject_context_includes_evolved_addendum():
    msgs = [{"role": "system", "content": "base"}, {"role": "user", "content": "q"}]
    out = p.inject_context(msgs, evolved="Be concise and ground answers in memory.")
    assert "Be concise and ground answers in memory." in out[0]["content"]
    assert "base" in out[0]["content"] and msgs[0]["content"] == "base"
    # recall + goals + evolved all fold in together
    out2 = p.inject_context(msgs, recall="R", goals="# Active Goals\n- [g] x", evolved="E-add")
    c = out2[0]["content"]
    assert "R" in c and "Active Goals" in c and "E-add" in c


def test_evolved_prompt_block_gated_off(monkeypatch):
    # evolver disabled → bridge returns "" regardless
    monkeypatch.setenv("HERMES_PASSTHROUGH_MEMORY", "1")
    monkeypatch.setenv("HERMES_PROMPT_EVOLVER_ENABLED", "false")
    assert p.evolved_prompt_block("voice-jarvis", "sid") == ""
    assert p.evolved_prompt_block("", "sid") == ""        # no scope key


def test_looks_like_goal_prefilter():
    assert p._looks_like_goal("my goal is to launch the app") is True
    assert p._looks_like_goal("remind me to renew the domain") is True
    assert p._looks_like_goal("I want to learn Spanish this year") is True
    assert p._looks_like_goal("what's the weather today?") is False
    assert p._looks_like_goal("open my email") is False
    assert p._looks_like_goal("ok") is False


def test_active_goal_count():
    block = "# Active Goals\n\n- [a1] (high) ship the app\n- [b2] (normal) renew domain\n\nKeep aligned."
    assert p.active_goal_count(block) == 2
    assert p.active_goal_count("") == 0


def test_maybe_track_goal_adds_then_dedupes(monkeypatch):
    monkeypatch.setenv("HERMES_VOICE_GOALS", "1")
    monkeypatch.setenv("HERMES_GOALS_ENABLED", "true")
    p._GOALS_COUNTS.update({"added": 0, "skip": 0, "dupe": 0, "fail": 0})
    p._RECENT_GOALS.clear()
    added = []

    class _Goals:
        def handle(self, name, args):
            added.append((name, args))
            return '{"ok": true, "goal": {"id": "abc"}}'

    class _Prov:
        _goals = _Goals()

    # LLM judge says "durable goal"
    monkeypatch.setattr(p, "_extract_goal_llm", lambda u, a: {"text": "launch the app by friday", "priority": "high"})
    p.maybe_track_goal("sk", _Prov(), "my goal is to launch the app by friday", "Noted, sir.")
    assert added and added[0][0] == "goal_add"
    assert "added=1" in p.goals_summary()
    # same goal again → deduped, not re-added
    p.maybe_track_goal("sk", _Prov(), "my goal is to launch the app by friday", "Noted, sir.")
    assert len(added) == 1

    # pre-filter miss → judge never called, nothing added
    monkeypatch.setattr(p, "_extract_goal_llm", lambda u, a: (_ for _ in ()).throw(AssertionError("should not run")))
    p.maybe_track_goal("sk", _Prov(), "what's the weather?", "It is sunny, sir.")
    assert len(added) == 1


def test_sse_accumulator_handles_splits_and_ignores_noise():
    a = p.SSEContentAccumulator()
    a.feed(b'data: {"choices":[{"delta":{"content":"Hel')   # split mid-line
    a.feed(b'lo"}}]}\n\ndata: {"choices":[{"delta":{"content":" world"}}]}\n\n')
    a.feed(b'data: [DONE]\n\n')
    assert a.text() == "Hello world"
    a2 = p.SSEContentAccumulator()
    a2.feed(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
    a2.feed(b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{}"}}]}}]}\n\n')
    assert a2.text() == ""                          # tool_calls/role deltas add nothing


def test_gating_and_missing_inputs(monkeypatch):
    monkeypatch.setenv("HERMES_PASSTHROUGH_MEMORY", "0")
    assert p.enabled() is False
    assert p.recall_block("scope", "sid", "q") == ""
    monkeypatch.setenv("HERMES_PASSTHROUGH_MEMORY", "1")
    assert p.enabled() is True
    assert p.recall_block("", "sid", "q") == ""      # no scope key
    assert p.recall_block("scope", "sid", "") == ""  # no user text


def test_is_substantive_gates_chitchat_but_keeps_facts():
    # trivial: bare acks / greetings → not learned
    assert p.is_substantive("hey jarvis", "Good evening, sir.") is False
    assert p.is_substantive("thanks", "You're welcome, sir.") is False
    assert p.is_substantive("are you there", "Yes, sir.") is False
    assert p.is_substantive("ok", "Acknowledged.") is False          # short reply
    # substantive: real facts / procedures → learned
    assert p.is_substantive(
        "remember my favourite colour is teal",
        "Noted, sir — I'll remember your favourite colour is teal going forward.") is True
    assert p.is_substantive(
        "how do I restart the worker",
        "Run `railway redeploy` on the service, then watch /health until it returns 200, "
        "which usually takes a few minutes on a cold boot.") is True


def test_writeback_counts_outcomes(monkeypatch):
    monkeypatch.setenv("HERMES_PASSTHROUGH_MEMORY", "1")
    p._WRITES.update({"ok": 0, "fail": 0, "skip": 0, "last_error": ""})

    class _OkProv:
        def sync_turn(self, u, a, *, session_id=""):
            return None

    class _BadProv:
        def sync_turn(self, u, a, *, session_id=""):
            raise RuntimeError("backend down")

    monkeypatch.setattr(p, "_get_provider", lambda sk, sid, plat: _OkProv())
    p.write_back("k", "s", "what is the plan for launch day", "Here is the full launch-day plan, sir.")
    monkeypatch.setattr(p, "_get_provider", lambda sk, sid, plat: _BadProv())
    p.write_back("k", "s", "another substantive question here", "Another substantive answer here, sir.")
    for _ in range(50):
        if "ok=1" in p.writes_summary() and "fail=1" in p.writes_summary():
            break
        time.sleep(0.02)
    assert "ok=1" in p.writes_summary()
    assert "fail=1" in p.writes_summary()
    assert p._WRITES["last_error"] == "backend down"


def test_writeback_skips_trivial_turn(monkeypatch):
    calls = []

    class _Stub:
        def sync_turn(self, u, a, *, session_id=""):
            calls.append((u, a))

    monkeypatch.setenv("HERMES_PASSTHROUGH_MEMORY", "1")
    monkeypatch.setattr(p, "_get_provider", lambda sk, sid, plat: _Stub())
    p.write_back("voice-jarvis", "s", "hey jarvis", "Good evening, sir.")
    time.sleep(0.2)
    assert calls == []                       # trivial turn never reached the provider


def test_goal_update_completes_active_goal(monkeypatch):
    monkeypatch.setenv("HERMES_VOICE_GOALS", "1")
    monkeypatch.setenv("HERMES_GOALS_ENABLED", "true")
    p._GOALS_COUNTS.update({"added": 0, "skip": 0, "dupe": 0, "fail": 0, "done": 0, "prog": 0})
    handled = []

    class _Goals:
        def _cached_active(self, n):
            return [{"id": "abc123", "text": "launch the Bridge app by Friday"}]
        def handle(self, name, args):
            handled.append((name, args))
            return '{"ok": true, "goal": {"id": "abc123"}}'

    class _Prov:
        _goals = _Goals()

    # judge says complete → goal_complete on the matching id
    monkeypatch.setattr(p, "_judge_goal_update", lambda u, a, b: {"action": "complete", "goal_id": "abc123"})
    p.maybe_update_goal("sk", _Prov(), "I've finished launching the Bridge app!", "Congratulations, sir.")
    assert ("goal_complete", {"id": "abc123"}) in handled
    assert "done=1" in p.goals_summary()

    # pre-filter miss → judge never runs, no handle
    handled.clear()
    monkeypatch.setattr(p, "_judge_goal_update", lambda u, a, b: (_ for _ in ()).throw(AssertionError("nope")))
    p.maybe_update_goal("sk", _Prov(), "what's on my plate today?", "You have three goals, sir.")
    assert handled == []

    # judge returns an id NOT in the active list → ignored (no invented ids)
    monkeypatch.setattr(p, "_judge_goal_update", lambda u, a, b: {"action": "complete", "goal_id": "zzz999"})
    p.maybe_update_goal("sk", _Prov(), "I finished it", "Great, sir.")
    assert handled == []


def test_status_reports_loop_state(monkeypatch):
    monkeypatch.setenv("HERMES_PASSTHROUGH_MEMORY", "0")
    assert p.status("scope", "sid") == "off"
    monkeypatch.setenv("HERMES_PASSTHROUGH_MEMORY", "1")
    assert p.status("", "sid") == "skip"                      # no scope key
    monkeypatch.setattr(p, "_get_provider", lambda sk, sid, plat: None)
    assert p.status("scope", "sid") == "skip"                 # no backend
    monkeypatch.setattr(p, "_get_provider", lambda sk, sid, plat: object())
    assert p.status("scope", "sid") == "attached"             # live loop

    def _boom(sk, sid, plat):
        raise RuntimeError("init blew up")
    monkeypatch.setattr(p, "_get_provider", _boom)
    assert p.status("scope", "sid") == "err"


def test_recall_and_writeback_drive_the_provider(monkeypatch):
    calls = {"prefetch": [], "queue": [], "sync": []}

    class _StubProvider:
        def prefetch(self, q, *, session_id=""):
            calls["prefetch"].append((q, session_id))
            return "[Vault Recall]\n- prior fact"

        def queue_prefetch(self, q, *, session_id=""):
            calls["queue"].append((q, session_id))

        def sync_turn(self, u, a, *, session_id=""):
            calls["sync"].append((u, a, session_id))

    monkeypatch.setenv("HERMES_PASSTHROUGH_MEMORY", "1")
    monkeypatch.setattr(p, "_get_provider", lambda sk, sid, plat: _StubProvider())

    rec = p.recall_block("voice-jarvis", "sid1", "what about teal")
    assert rec == "[Vault Recall]\n- prior fact"
    assert calls["prefetch"] == [("what about teal", "sid1")]
    assert calls["queue"] == [("what about teal", "sid1")]      # this turn queued for next

    # substantive turn (trivial chit-chat is gated out separately)
    _u, _a = "what is my favourite colour", "Your favourite colour is teal, sir — noted."
    p.write_back("voice-jarvis", "sid1", _u, _a)                # background thread
    for _ in range(50):
        if calls["sync"]:
            break
        time.sleep(0.02)
    assert calls["sync"] == [(_u, _a, "sid1")]
