"""A completion whose owner process died is re-offered by processes that are ALREADY running (#97202).

Startup replay (``restore_undelivered_completions``) runs once per process, so before this a result
persisted by a process that then died (a desktop reload) waited for the next process start. These
tests produce the orphan with a real owner process against a real temp ``state.db``; time is injected
(``now=``) or written into the row, never raced against the wall clock.
"""

import json
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools import async_delegation as ad
from tools.process_registry import process_registry

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The owner dispatches, its child fails (the reload-interrupted child of the report lands as
# state='error'), the completion is persisted, and the process dies before anything delivers it.
_OWNER = r'''
import os, time
from tools import async_delegation as ad
def child():
    raise RuntimeError("interrupted: waiting for model response")
r = ad.dispatch_async_delegation(
    goal="adversarial review", context=None, toolsets=None, role="leaf", model="m",
    session_key="bot-chat", parent_session_id="bot-parent", runner=child)
deadline = time.time() + 10
while ad.active_count() and time.time() < deadline:
    time.sleep(.01)
print(r["delegation_id"], flush=True)
os._exit(0)
'''


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _orphan(home: Path) -> str:
    """Run a real owner process under ``home`` and return the id of the completion it left pending."""
    home.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": REPO}
    out = subprocess.run([sys.executable, "-c", _OWNER], cwd=REPO, env=env, text=True,
                         capture_output=True, timeout=60, check=True)
    return out.stdout.strip().splitlines()[-1]


def _live_owner_completion(home: Path) -> str:
    """Complete normally in this still-live process without consuming delivery."""
    home.mkdir(parents=True, exist_ok=True)
    with _Home(home):
        handle = ad.dispatch_async_delegation(
            goal="live owner", context=None, toolsets=None, role="leaf", model="m",
            session_key="live-parent", parent_session_id="parent-session",
            runner=lambda: {"status": "completed", "summary": "done"},
        )
        deadline = time.monotonic() + 10
        while ad.active_count() and time.monotonic() < deadline:
            time.sleep(0.01)
    return handle["delegation_id"]


def _row(home: Path, delegation_id: str) -> dict:
    conn = sqlite3.connect(home / "state.db")
    try:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone())
    finally:
        conn.close()


def _set(home: Path, delegation_id: str, **cols) -> None:
    conn = sqlite3.connect(home / "state.db")
    try:
        conn.execute(f"UPDATE async_delegations SET {', '.join(f'{k}=?' for k in cols)} WHERE delegation_id=?",
                     (*cols.values(), delegation_id))
        conn.commit()
    finally:
        conn.close()


class _Home:
    """Bind the profile home for code outside a turn, the way a delivery loop does."""

    def __init__(self, home: Path):
        self.home = home

    def __enter__(self):
        self._token = set_hermes_home_override(str(self.home))
        return self

    def __exit__(self, *exc):
        reset_hermes_home_override(self._token)


def _drain(q) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_orphaned_terminal_completion_is_offered_once_while_the_process_runs(tmp_path):
    home = tmp_path / "home"
    delegation_id = _orphan(home)
    row = _row(home, delegation_id)
    assert (row["state"], row["delivery_state"]) == ("error", "pending") and row["event_json"]
    q = queue.Queue()
    with _Home(home):
        # Freshly written: a live consumer may still be on it, so the sweep leaves it alone.
        assert ad.sweep_orphaned_completions(q, now=row["updated_at"] + 1) == 0
        later = row["updated_at"] + ad._ORPHAN_STALE_S + 1
        assert ad.sweep_orphaned_completions(q, now=later) == 1
        (evt,) = _drain(q)
        assert evt["delegation_id"] == delegation_id and evt["status"] == "error"
        assert evt["session_key"] == "bot-chat" and evt["restored"] is True
        # Offered once per process: the next sweep does not flood the queue with the same row.
        assert ad.sweep_orphaned_completions(q, now=later + 60) == 0
        assert q.empty()
        # Delivery still goes through the atomic claim: a second consumer (another process that
        # also offered the row) cannot claim it, and the winner's ack settles the row.
        claim = ad.claim_event_delivery(evt, "first")
        assert claim
        assert ad.claim_event_delivery(evt, "second") is None
        assert ad.complete_completion_delivery(delegation_id, claim)
    assert _row(home, delegation_id)["delivery_state"] == "delivered"


def test_aged_pending_completion_is_recovered_while_owner_remains_alive(tmp_path):
    home = tmp_path / "home"
    delegation_id = _live_owner_completion(home)
    row = _row(home, delegation_id)
    assert row["owner_pid"] == os.getpid() and row["delivery_state"] == "pending"
    q = queue.Queue()
    with _Home(home):
        # Preserve the normal immediate-delivery window.
        assert ad.sweep_orphaned_completions(q, now=row["completed_at"] + 1) == 0
        later = row["completed_at"] + ad._PENDING_DELIVERY_GRACE_S + 1
        assert ad.sweep_orphaned_completions(q, now=later) == 1
        (evt,) = _drain(q)
        assert evt["delegation_id"] == delegation_id
        assert evt["delivery_recovery_reason"] == "aged_pending_live_owner"
        claim = ad.claim_event_delivery(evt, "parent")
        assert claim and ad.complete_event_delivery(evt, claim) is None
    recovered = _row(home, delegation_id)
    assert recovered["delivery_state"] == "delivered"
    assert recovered["delivery_attempts"] == 1
    assert recovered["delivery_claimed_at"] is None
    assert recovered["last_delivery_claimed_at"] is not None
    assert recovered["delivered_at"] is not None
    assert recovered["delivery_recovery_reason"] == "aged_pending_live_owner"


def test_normal_delivery_before_grace_is_not_recovered(tmp_path):
    home = tmp_path / "home"
    delegation_id = _live_owner_completion(home)
    row = _row(home, delegation_id)
    evt = json.loads(row["event_json"])
    q = queue.Queue()
    with _Home(home):
        claim = ad.claim_event_delivery(evt, "normal")
        assert claim
        ad.complete_event_delivery(evt, claim)
        assert ad.sweep_orphaned_completions(
            q, now=row["completed_at"] + ad._PENDING_DELIVERY_GRACE_S + 1,
        ) == 0
    assert q.empty()
    assert _row(home, delegation_id)["delivery_recovery_reason"] is None


def test_two_aged_pending_recovery_workers_process_once(tmp_path):
    home = tmp_path / "home"
    delegation_id = _live_owner_completion(home)
    row = _row(home, delegation_id)
    later = row["completed_at"] + ad._PENDING_DELIVERY_GRACE_S + 1
    q = queue.Queue()
    counts = []

    def recover():
        with _Home(home):
            counts.append(ad.sweep_orphaned_completions(q, now=later))

    threads = [threading.Thread(target=recover) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with _Home(home):
        assert sum(counts) == 1
        (evt,) = _drain(q)
        first = ad.claim_event_delivery(evt, "first")
        assert first
        assert ad.claim_event_delivery(evt, "second") is None
        ad.complete_event_delivery(evt, first)
    assert _row(home, delegation_id)["delivery_attempts"] == 1


def test_stale_delivery_claim_is_reclaimed_for_aged_live_owner(tmp_path):
    home = tmp_path / "home"
    delegation_id = _live_owner_completion(home)
    row = _row(home, delegation_id)
    later = row["completed_at"] + ad._PENDING_DELIVERY_GRACE_S + 1
    _set(
        home, delegation_id,
        delivery_claim="stalled-consumer",
        delivery_claimed_at=time.time() - ad._CLAIM_LEASE_S - 1,
    )
    q = queue.Queue()
    with _Home(home):
        assert ad.sweep_orphaned_completions(q, now=later) == 1
        (evt,) = _drain(q)
        claim = ad.claim_event_delivery(evt, "recovery")
        assert claim
        ad.complete_event_delivery(evt, claim)
    assert _row(home, delegation_id)["delivery_state"] == "delivered"


def test_unavailable_parent_releases_aged_completion_for_later_retry(tmp_path):
    home = tmp_path / "home"
    delegation_id = _live_owner_completion(home)
    row = _row(home, delegation_id)
    later = row["completed_at"] + ad._PENDING_DELIVERY_GRACE_S + 1
    q = queue.Queue()
    with _Home(home):
        assert ad.sweep_orphaned_completions(q, now=later) == 1
        (evt,) = _drain(q)
        claim = ad.claim_event_delivery(evt, "unavailable-parent")
        assert claim
        assert ad.release_completion_delivery(delegation_id, claim)
        ad.return_completion_offer(evt)
        pending = _row(home, delegation_id)
        assert pending["delivery_state"] == "pending"
        assert pending["delivery_attempts"] == 1
        assert pending["event_json"] and pending["result_json"]
        retry_at = pending["updated_at"] + ad._PENDING_DELIVERY_GRACE_S + 1
        assert ad.sweep_orphaned_completions(q, now=retry_at) == 1
        (retry_evt,) = _drain(q)
        retry_claim = ad.claim_event_delivery(retry_evt, "parent-back")
        assert retry_claim
        ad.complete_event_delivery(retry_evt, retry_claim)
    assert _row(home, delegation_id)["delivery_state"] == "delivered"


def test_sweep_is_bound_to_the_profile_home_it_runs_under(tmp_path):
    """A→B→A: each profile's sweep reads only its own ledger and never adopts the other's row."""
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    id_a, id_b = _orphan(home_a), _orphan(home_b)
    later = max(_row(home_a, id_a)["updated_at"], _row(home_b, id_b)["updated_at"]) + ad._ORPHAN_STALE_S + 1
    q = queue.Queue()
    with _Home(home_a):
        ad.sweep_orphaned_completions(q, now=later)
    assert [e["delegation_id"] for e in _drain(q)] == [id_a]
    with _Home(home_b):
        ad.sweep_orphaned_completions(q, now=later)
    assert [e["delegation_id"] for e in _drain(q)] == [id_b]
    with _Home(home_a):
        assert ad.sweep_orphaned_completions(q, now=later + 60) == 0
    assert q.empty()
    assert _row(home_b, id_b)["delivery_state"] == "pending"


def test_restart_replay_and_sweep_never_offer_the_same_row_twice(tmp_path):
    home = tmp_path / "home"
    delegation_id = _orphan(home)
    later = _row(home, delegation_id)["updated_at"] + ad._ORPHAN_STALE_S + 1
    q = queue.Queue()
    with _Home(home):
        assert ad.restore_undelivered_completions(q) == 1
        assert ad.sweep_orphaned_completions(q, now=later) == 0
    assert [e["delegation_id"] for e in _drain(q)] == [delegation_id]


@pytest.mark.parametrize("exhausted", ["attempts", "age"])
def test_orphan_past_its_delivery_budget_converges_to_dropped(tmp_path, exhausted):
    home = tmp_path / "home"
    delegation_id = _orphan(home)
    row = _row(home, delegation_id)
    later = row["updated_at"] + ad._ORPHAN_STALE_S + 1
    if exhausted == "attempts":
        _set(home, delegation_id, delivery_attempts=ad._MAX_DELIVERY_ATTEMPTS)
    else:
        later = row["completed_at"] + ad._MAX_COMPLETION_REPLAY_AGE_S + 1
    q = queue.Queue()
    with _Home(home):
        assert ad.sweep_orphaned_completions(q, now=later) == 0
    assert q.empty()
    assert _row(home, delegation_id)["delivery_state"] == "dropped"


def test_in_flight_claim_is_left_to_its_holder(tmp_path):
    home = tmp_path / "home"
    delegation_id = _orphan(home)
    row = _row(home, delegation_id)
    later = row["updated_at"] + ad._ORPHAN_STALE_S + 1
    _set(home, delegation_id, delivery_claim="other-process", delivery_claimed_at=later - 1)
    q = queue.Queue()
    with _Home(home):
        assert ad.sweep_orphaned_completions(q, now=later) == 0
    assert q.empty()


def test_gateway_watcher_sweeps_each_served_ledger_in_its_own_scope(tmp_path, monkeypatch):
    """The gateway's delivery loop re-offers a secondary profile's orphan while it runs."""
    from gateway.run import GatewayRunner

    home_b = tmp_path / "b"
    delegation_id = _orphan(home_b)
    stale = time.time() - ad._ORPHAN_STALE_S - 60
    _set(home_b, delegation_id, updated_at=stale, completed_at=stale)
    isolated = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    runner = object.__new__(GatewayRunner)
    runner._primary_profile_name = "default"
    runner._served_profile_homes = {"default": get_hermes_home(), "b": home_b}
    runner._sweep_orphaned_completion_ledgers()
    assert [e["delegation_id"] for e in _drain(isolated)] == [delegation_id]


def test_tui_notification_poller_sweeps_under_its_session_profile(tmp_path, monkeypatch):
    """The desktop/TUI delivery loop runs the (throttled) sweep in the session's own profile scope."""
    from tui_gateway import server

    home_b = tmp_path / "b"
    home_b.mkdir()
    stop = threading.Event()
    seen = []

    def fake_sweep(target_queue):
        seen.append((str(get_hermes_home()), target_queue))
        stop.set()
        return 0

    monkeypatch.setattr(ad, "maybe_sweep_orphaned_completions", fake_sweep)
    isolated = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    session = {"profile_home": str(home_b), "_finalized": False}
    server._notification_poller_loop(stop, "sid-orphan-sweep", session)
    assert seen and seen[0][0] == str(home_b) and seen[0][1] is isolated


def _tui_session(session_key: str, home: Path) -> dict:
    return {"history_lock": threading.RLock(), "running": False, "history": [], "session_key": session_key,
            "profile_home": str(home), "_finalized": False, "_notification_emitted": set()}


def test_offer_dropped_by_a_session_that_cannot_own_it_is_re_offered_to_the_owner(tmp_path, monkeypatch):
    """Every TUI/Desktop poller drains one process-wide queue. A session that cannot prove it owns an
    async delegation drops its copy while the durable row stays pending, so the offer must not
    suppress the next sweep: the owning session, live later, still gets the row without a restart."""
    from tools.process_registry_notifications import format_process_notification
    from tui_gateway import server

    home = tmp_path / "home"
    delegation_id = _orphan(home)
    later = _row(home, delegation_id)["updated_at"] + ad._ORPHAN_STALE_S + 1
    q = queue.Queue()
    registry = type("Registry", (), {"completion_queue": q, "is_completion_consumed": lambda self, sid: False})()
    started = []
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda rid, sid, session, text, **kw: started.append(sid))

    def drain(sid, session):
        server._notif_handle_ready(sid, session, _drain(q), session["_notification_emitted"], registry,
                                   format_process_notification, None)

    other, owner = _tui_session("other-chat", home), _tui_session("bot-chat", home)
    with _Home(home):
        monkeypatch.setitem(server._sessions, "sid-other", other)
        assert ad.sweep_orphaned_completions(q, now=later) == 1
        drain("sid-other", other)  # the wrong session wins the dequeue while the owner is not live yet
        assert q.empty() and started == []
        assert _row(home, delegation_id)["delivery_state"] == "pending"

        monkeypatch.setitem(server._sessions, "sid-owner", owner)  # the owner resumes
        assert ad.sweep_orphaned_completions(q, now=later + ad.ORPHAN_SWEEP_INTERVAL_S) == 1
        drain("sid-owner", owner)
    assert started == ["sid-owner"]
    assert _row(home, delegation_id)["delivery_state"] == "delivered"
    with _Home(home):  # delivered rows are never offered again
        assert ad.sweep_orphaned_completions(q, now=later + 2 * ad.ORPHAN_SWEEP_INTERVAL_S) == 0


def test_offer_released_after_a_failed_tui_turn_is_re_offered(tmp_path, monkeypatch):
    """The TUI poller releases its claim and discards its copy when the turn cannot start; the row is
    pending again with one attempt spent, so the sweep offers it again instead of skipping it."""
    from tui_gateway import server

    home = tmp_path / "home"
    delegation_id = _orphan(home)
    later = _row(home, delegation_id)["updated_at"] + ad._ORPHAN_STALE_S + 1
    q = queue.Queue()
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no free worker")))
    owner = _tui_session("bot-chat", home)
    with _Home(home):
        assert ad.sweep_orphaned_completions(q, now=later) == 1
        (evt,) = _drain(q)
        assert server._notif_claim_turn(owner)
        server._notif_dispatch_event("sid-owner", owner, evt, "text")
        row = _row(home, delegation_id)
        assert (row["delivery_state"], row["delivery_claim"], row["delivery_attempts"]) == ("pending", None, 1)
        assert ad.sweep_orphaned_completions(q, now=row["updated_at"] + ad._ORPHAN_STALE_S + 1) == 1
    assert [e["delegation_id"] for e in _drain(q)] == [delegation_id]


def test_throttle_runs_at_most_one_sweep_per_home_per_interval(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ad, "sweep_orphaned_completions", lambda q, **kw: calls.append(str(get_hermes_home())) or 0)
    q = queue.Queue()
    with _Home(tmp_path / "a"):
        ad.maybe_sweep_orphaned_completions(q, now=100.0)
        ad.maybe_sweep_orphaned_completions(q, now=101.0)
        with _Home(tmp_path / "b"):  # another profile has its own clock
            ad.maybe_sweep_orphaned_completions(q, now=101.0)
        ad.maybe_sweep_orphaned_completions(q, now=100.0 + ad.ORPHAN_SWEEP_INTERVAL_S + 1)
    assert calls == [str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "a")]
