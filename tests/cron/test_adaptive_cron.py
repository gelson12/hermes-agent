"""WS-C: outcome-adaptive cron cadence — the cron analog of ScheduleWakeup.

A recurring interval job created with ``adaptive=True`` tightens its interval
after a tick that has something to report ("changed") and backs off when ticks
are quiet ("idle"), staying within [min_s, max_s]. HERMES_ADAPTIVE_CRON=0 pins it
back to the fixed schedule.
"""
from datetime import datetime

import pytest

from cron.jobs import (
    create_job,
    get_job,
    mark_job_run,
    _normalize_adaptive,
    parse_schedule,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def _delta_s(job) -> float:
    """Seconds between a job's last_run_at and its scheduled next_run_at."""
    nxt = datetime.fromisoformat(job["next_run_at"])
    last = datetime.fromisoformat(job["last_run_at"])
    return (nxt - last).total_seconds()


# ── normalization ───────────────────────────────────────────────────────────

def test_normalize_adaptive_interval_defaults():
    sched = parse_schedule("every 60m")          # base 3600s
    a = _normalize_adaptive(sched, True)
    assert a["min_s"] == 900 and a["max_s"] == 14400 and a["backoff"] == 2.0
    assert a["current_s"] == 3600


def test_normalize_adaptive_rejects_non_interval():
    assert _normalize_adaptive(parse_schedule("30m"), True) is None        # one-shot
    assert _normalize_adaptive(parse_schedule("every 10m"), None) is None  # opt-out
    assert _normalize_adaptive(parse_schedule("every 10m"), False) is None
    from cron.jobs import HAS_CRONITER
    if HAS_CRONITER:  # cron-expr parsing requires croniter (a prod core dep)
        assert _normalize_adaptive(parse_schedule("0 9 * * *"), True) is None


def test_normalize_adaptive_custom_bounds():
    a = _normalize_adaptive(parse_schedule("every 10m"), {"min_s": 120, "max_s": 1200, "backoff": 3})
    assert a["min_s"] == 120 and a["max_s"] == 1200 and a["backoff"] == 3.0


# ── create stores it ────────────────────────────────────────────────────────

def test_create_job_stores_adaptive_on_interval(tmp_cron_dir):
    job = create_job(prompt="watch the price", schedule="every 30m", adaptive=True)
    stored = get_job(job["id"])
    assert stored["adaptive"]["current_s"] == 1800

    one_shot = create_job(prompt="ping later", schedule="30m", adaptive=True)
    assert get_job(one_shot["id"]).get("adaptive") is None


# ── adaptive rescheduling ───────────────────────────────────────────────────

def test_changed_tightens_idle_loosens(tmp_cron_dir):
    job = create_job(prompt="watch", schedule="every 60m", adaptive=True)  # base 3600
    jid = job["id"]

    mark_job_run(jid, success=True, signal="changed")
    assert _delta_s(get_job(jid)) == pytest.approx(1800, abs=5)   # 3600 → /2

    mark_job_run(jid, success=True, signal="changed")
    assert _delta_s(get_job(jid)) == pytest.approx(900, abs=5)    # floor min_s

    mark_job_run(jid, success=True, signal="changed")
    assert _delta_s(get_job(jid)) == pytest.approx(900, abs=5)    # clamped at min_s

    mark_job_run(jid, success=True, signal="idle")
    assert _delta_s(get_job(jid)) == pytest.approx(1800, abs=5)   # 900 → *2


def test_idle_backs_off_to_ceiling(tmp_cron_dir):
    job = create_job(prompt="watch", schedule="every 60m", adaptive=True)  # base 3600
    jid = job["id"]
    for _ in range(5):
        mark_job_run(jid, success=True, signal="idle")
    assert _delta_s(get_job(jid)) == pytest.approx(14400, abs=5)  # ceiling max_s (base*4)


def test_kill_switch_pins_to_schedule(tmp_cron_dir, monkeypatch):
    monkeypatch.setenv("HERMES_ADAPTIVE_CRON", "0")
    job = create_job(prompt="watch", schedule="every 60m", adaptive=True)
    jid = job["id"]
    mark_job_run(jid, success=True, signal="changed")
    # adaptive override is disabled → falls back to the fixed schedule (60m).
    assert _delta_s(get_job(jid)) == pytest.approx(3600, abs=5)
    assert get_job(jid)["adaptive"]["current_s"] == 3600  # untouched


def test_non_adaptive_job_unaffected_by_signal(tmp_cron_dir):
    job = create_job(prompt="plain", schedule="every 60m")  # no adaptive
    jid = job["id"]
    mark_job_run(jid, success=True, signal="changed")
    assert _delta_s(get_job(jid)) == pytest.approx(3600, abs=5)  # fixed schedule
    assert get_job(jid).get("adaptive") is None
