"""A direction pause retains its execution, including after Worker process loss."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_db_notify as notify
from hermes_cli import kanban_db_workspace as workspace_db
from hermes_cli import kanban_direction as direction


QUESTION = {
    "question": "Which interface owns the compatibility mapping?",
    "why_it_matters": "Changing its owner changes the published migration contract.",
    "evidence": {"references": ["canon/interfaces.md:42", "src/mapping.py:19"],
                 "precedent_status": "conflicting"},
    "options": ["Retain the existing adapter", "Migrate the public API"],
    "decision_needed": "migration_or_public_interface",
}
ANSWER = {
    "decision": "Retain the existing adapter.",
    "rationale": "Preserve the published compatibility contract.",
    "resume_instruction": "Continue this activity and verify the adapter contract.",
}


@pytest.fixture
def worker(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD",
                "HERMES_KANBAN_TASK", "HERMES_KANBAN_WORKSPACES_ROOT"):
        monkeypatch.delenv(key, raising=False)
    # A real pre-existing branch/worktree with unfinished local activity. No
    # production repository, gateway, or Worker process participates.
    repo, workspace = tmp_path / "repo", tmp_path / "existing-worktree"
    for argv in (
        ["git", "init", "--initial-branch=main", str(repo)],
        ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.org",
         "commit", "--allow-empty", "-m", "fixture"],
        ["git", "-C", str(repo), "worktree", "add", "-b", "worker/existing", str(workspace)],
    ):
        subprocess.run(argv, check=True, capture_output=True)
    (workspace / "activity.txt").write_text("existing activity and unfinished changes")
    gitdir = Path((workspace / ".git").read_text().removeprefix("gitdir: ").strip())
    from hermes_state import SessionDB

    worker_home = home / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    (worker_home / "SOUL.md").write_text("Fixture Worker")
    session_db = SessionDB(db_path=worker_home / "state.db")
    try:
        session_db.create_session(session_id="original-worker-session", source="kanban",
                                  cwd=str(workspace))
        session_db.append_message(session_id="original-worker-session", role="user",
                                  content="Continue the existing activity in this workspace.")
    finally:
        session_db.close()
    clock = [2_000_000_000]
    monkeypatch.setattr(direction.time, "time", lambda: clock[0])
    monkeypatch.setattr(dispatch, "_process_fingerprint", lambda pid: f"worker-{pid}")
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "normal")
    kb.init_db()
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="Keep this activity", assignee="worker", created_by="coordinator",
            session_id="owner-session", workspace_kind="worktree", workspace_path=str(workspace),
            branch_name="worker/existing", max_runtime_seconds=120,
        )
        notify.add_notify_sub(
            conn, task_id=task_id, platform="discord", chat_id="owner-thread",
            chat_type="group", user_id="owner-user", notifier_profile="coordinator",
            delivery_mode="notify+wake", delivery_metadata={"scope_id": "owner-guild"},
        )
        task = kb.claim_task(conn, task_id)
        dispatch._set_worker_pid(conn, task_id, 987654321)
        metadata = {"activity": {"id": "existing-activity", "correction_cycles": 1},
                    "opaque_worker_state": ["keep", {"budget": 2}]}
        conn.execute("UPDATE tasks SET consecutive_failures = 1, block_recurrences = 1 WHERE id = ?",
                     (task_id,))
        conn.execute("UPDATE task_runs SET metadata = ? WHERE id = ?",
                     (json.dumps(metadata), task.current_run_id))
        conn.commit()
        task = kb.get_task(conn, task_id)
        run = dict(conn.execute("SELECT * FROM task_runs WHERE id = ?", (task.current_run_id,)).fetchone())
    return SimpleNamespace(task=task, run=run, workspace=workspace, gitdir=gitdir, clock=clock,
                           worker_home=worker_home)


def _assert_same_execution(conn, worker, status):
    task = kb.get_task(conn, worker.task.id)
    assert task.status == status
    for field in ("id", "current_run_id", "assignee", "workspace_kind", "workspace_path",
                  "branch_name", "claim_lock", "started_at", "session_id", "consecutive_failures",
                  "block_recurrences", "max_retries", "goal_max_turns"):
        assert getattr(task, field) == getattr(worker.task, field), field
    runs = conn.execute("SELECT * FROM task_runs WHERE task_id = ?", (task.id,)).fetchall()
    assert len(runs) == 1
    run = dict(runs[0])
    assert run["id"] == worker.run["id"] == task.current_run_id
    assert run["status"] == status
    for field in ("metadata", "started_at", "claim_lock", "outcome", "terminal_result", "ended_at"):
        assert run[field] == worker.run[field], field
    assert run["outcome"] is run["terminal_result"] is run["ended_at"] is None
    assert len(kb.list_tasks(conn)) == 1
    assert (worker.workspace / "activity.txt").read_text() == "existing activity and unfinished changes"
    assert (worker.gitdir / "HEAD").read_text().strip() == "ref: refs/heads/worker/existing"
    return run


def _ask(conn, worker, question=QUESTION):
    return direction.request_direction(
        conn, worker.task.id, expected_run_id=worker.task.current_run_id,
        worker_session_id="original-worker-session", question=question,
    )


def _answer(conn, worker, question):
    return direction.answer_direction(
        conn, worker.task.id, question["id"], responder_profile="coordinator", response=ANSWER,
        responder_session={**question["owner_session"], "session_id": "owner-session"},
    )


@pytest.mark.parametrize("empty_session", [False, True])
def test_unavailable_worker_history_stays_paused_without_replacement(
        worker, monkeypatch, all_assignees_spawnable, empty_session):
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    db = SessionDB(db_path=get_hermes_home() / "profiles" / "worker" / "state.db")
    try:
        db.delete_session("original-worker-session")
        if empty_session:
            db.create_session(session_id="original-worker-session", source="kanban")
    finally:
        db.close()
    monkeypatch.setattr(dispatch, "_worker_alive", lambda *_args: False)
    launch = Mock(side_effect=AssertionError("missing history must never start a fresh Worker"))
    monkeypatch.setattr(subprocess, "Popen", launch)
    with kbc.connect_closing() as conn:
        question = _ask(conn, worker)
        _answer(conn, worker, question)
        assert dispatch.dispatch_once(conn).spawned == []
        _assert_same_execution(conn, worker, "needs_direction")
        durable = direction.latest_direction(conn, worker.task.id)
        assert durable["response"] == ANSWER
        assert durable["resumed_at"] is durable["resume_token"] is None
    launch.assert_not_called()


def _admit(launch):
    from hermes_cli.kanban_direction_admission import (
        admit_prepared_direction_resume, prepare_direction_resume_from_env,
    )
    from hermes_state import SessionDB

    with patch.dict(os.environ, launch["env"], clear=True):
        admission = prepare_direction_resume_from_env(worker_session_id="original-worker-session")
        if admission is False:
            return False
        db = SessionDB(Path(launch["env"]["HERMES_HOME"]) / "state.db", read_only=True)
        try:
            history = db.get_messages_as_conversation("original-worker-session", repair_alternation=True)
        finally:
            db.close()
        return admit_prepared_direction_resume(
            SimpleNamespace(_direction_resume_admission=admission,
                            session_id="original-worker-session", agent=object()),
            conversation_history=history,
        )


def test_two_durable_pauses_exclude_waiting_from_watchdogs_and_runtime(
        worker, monkeypatch, all_assignees_spawnable):
    monkeypatch.setattr(dispatch, "_worker_alive", lambda *_args: True)
    spawn = Mock(side_effect=AssertionError("a waiting Worker must not be spawned"))
    signal = Mock(side_effect=AssertionError("a waiting Worker must not be signalled"))
    total_paused = 0
    question_ids = []
    for cycle in range(2):
        worker.clock[0] += 10  # actual work between decisions counts toward the runtime cap
        payload = {**QUESTION, "question": f"{QUESTION['question']} Decision {cycle + 1}."}
        with kbc.connect_closing() as conn:
            question = _ask(conn, worker, payload)
            assert _ask(conn, worker, payload)["id"] == question["id"]
        # Re-open the board: the pause and all session/run bindings must survive.
        with kbc.connect_closing() as conn:
            durable = direction.latest_direction(conn, worker.task.id)
            assert durable["question"] == payload
            assert (durable["task_id"], durable["run_id"], durable["worker_session_id"],
                    durable["owner_profile"], durable["owner_session_id"]) == (
                        worker.task.id, worker.task.current_run_id, "original-worker-session",
                        "coordinator", "owner-session")
            assert durable["response"] is None
            question_ids.append(durable["id"])
            # No answer, even months after every claim/heartbeat/runtime deadline.
            pause_seconds = 180 * 86400 + cycle
            worker.clock[0] += pause_seconds
            total_paused += pause_seconds
            assert kb.release_stale_claims(conn, signal_fn=signal) == 0
            assert dispatch.detect_stale_running(conn, stale_timeout_seconds=1, signal_fn=signal) == []
            assert dispatch.enforce_max_runtime(conn, signal_fn=signal) == []
            assert dispatch._record_task_failure(
                conn, worker.task.id, "late Worker deadline", outcome="timed_out", end_run=True,
            ) is False
            assert kb.reclaim_task(conn, worker.task.id, reason="ordinary reclaim", signal_fn=signal) is False
            with pytest.raises(RuntimeError, match="claimed"):
                kb.assign_task(conn, worker.task.id, "replacement-worker")
            tick = dispatch.dispatch_once(conn, spawn_fn=spawn, stale_timeout_seconds=1)
            assert not (tick.spawned or tick.crashed or tick.stale or tick.timed_out or tick.reclaimed)
            _assert_same_execution(conn, worker, "needs_direction")
            assert direction.consume_direction(
                conn, worker.task.id, durable["id"], expected_run_id=worker.task.current_run_id,
                claim_lock=worker.task.claim_lock,
            ) is None
            _answer(conn, worker, durable)
        with kbc.connect_closing() as conn:
            durable = direction.latest_direction(conn, worker.task.id)
            assert durable["response"] == ANSWER and durable["answered_at"] == worker.clock[0]
            _assert_same_execution(conn, worker, "needs_direction")
            resumed = direction.consume_direction(
                conn, worker.task.id, durable["id"], expected_run_id=worker.task.current_run_id,
                claim_lock=worker.task.claim_lock,
            )
            assert resumed["response"] == ANSWER
            run = _assert_same_execution(conn, worker, "running")
            assert run["direction_paused_seconds"] == total_paused
            assert run["last_heartbeat_at"] == run["last_activity_at"] == worker.clock[0]
            assert run["claim_expires"] > worker.clock[0]
            assert direction.consume_direction(
                conn, worker.task.id, durable["id"], expected_run_id=worker.task.current_run_id,
                claim_lock=worker.task.claim_lock,
            ) is None
            assert _answer(conn, worker, durable)["resumed_at"] == worker.clock[0]
            assert dispatch.enforce_max_runtime(conn, signal_fn=signal) == []
    assert len(set(question_ids)) == 2
    spawn.assert_not_called()
    signal.assert_not_called()
    with kbc.connect_closing() as conn:
        events = kb.list_events(conn, worker.task.id)
        for kind in ("developer_question", "developer_response", "direction_resumed"):
            assert sum(e.kind == kind for e in events) == 2
        assert worker.clock[0] - worker.run["started_at"] - total_paused == 20
        # The 120s cap still works on active time; it was extended, not disabled.
        worker.clock[0] += 99
        assert dispatch.enforce_max_runtime(conn, signal_fn=signal) == []
        worker.clock[0] += 2
        monkeypatch.setattr(dispatch, "_worker_alive", lambda *_args: False)
        assert dispatch.enforce_max_runtime(conn, signal_fn=lambda *_args: None) == [worker.task.id]


def test_lost_paused_process_restores_session_without_replacing_run_or_workspace(
        worker, monkeypatch, all_assignees_spawnable):
    original_pid, restored_pid = worker.task.worker_pid, 987654322
    monkeypatch.setattr(dispatch, "_worker_alive", lambda pid, _fingerprint: pid == restored_pid)
    provision = Mock(side_effect=AssertionError("resume must reuse the existing worktree"))
    monkeypatch.setattr(workspace_db, "resolve_workspace", provision)
    monkeypatch.setattr(workspace_db, "_resolve_worktree_workspace", provision)
    spawn = Mock(wraps=dispatch._default_spawn)
    argv = Mock(wraps=dispatch._worker_argv)
    monkeypatch.setattr(dispatch, "_default_spawn", spawn)
    monkeypatch.setattr(dispatch, "_worker_argv", argv)
    monkeypatch.setattr(dispatch, "_restart_safe_worker_argv", lambda _task, command: command)
    launches = []

    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return SimpleNamespace(pid=restored_pid)

    monkeypatch.setattr(subprocess, "Popen", popen)
    with kbc.connect_closing() as conn:
        question = _ask(conn, worker)
        worker.clock[0] += 365 * 86400
        # Lost original process stays paused until an actual owner answer exists.
        assert dispatch.dispatch_once(conn, stale_timeout_seconds=1).spawned == []
        _assert_same_execution(conn, worker, "needs_direction")
        assert kb.get_task(conn, worker.task.id).worker_pid == original_pid
        _answer(conn, worker, question)
    with kbc.connect_closing() as conn:
        result = dispatch.dispatch_once(conn, stale_timeout_seconds=1)
        assert result.spawned == [(worker.task.id, worker.task.assignee, str(worker.workspace))]
        assert not (result.crashed or result.stale or result.timed_out or result.reclaimed)
        run = _assert_same_execution(conn, worker, "needs_direction")
        assert run["worker_pid"] == kb.get_task(conn, worker.task.id).worker_pid == restored_pid
        waiting = direction.latest_direction(conn, worker.task.id)
        assert waiting["resumed_at"] is waiting["resume_admitted_at"] is None
        assert waiting["resume_token"] == launches[0][1]["env"]["HERMES_KANBAN_DIRECTION_TOKEN"]
        # Booting the resumed interpreter remains paused too. Only the child
        # admitted against the committed launch token may restart execution.
        worker.clock[0] += 600
        assert dispatch.dispatch_once(conn, stale_timeout_seconds=1).spawned == []
        _assert_same_execution(conn, worker, "needs_direction")
        assert _admit(launches[0][1]) is True
        assert _admit(launches[0][1]) is False
        run = _assert_same_execution(conn, worker, "running")
        assert run["worker_pid"] == kb.get_task(conn, worker.task.id).worker_pid == restored_pid
        assert run["direction_paused_seconds"] == 365 * 86400 + 600
        for _ in range(3):
            _answer(conn, worker, question)
            assert dispatch.dispatch_once(conn, stale_timeout_seconds=1).spawned == []
        _assert_same_execution(conn, worker, "running")
        events = kb.list_events(conn, worker.task.id)
        assert sum(e.kind == "claimed" for e in events) == 1
        for kind in ("developer_response", "direction_resumed", "direction_resume_launched"):
            assert sum(e.kind == kind for e in events) == 1
    provision.assert_not_called()
    spawn.assert_called_once()
    argv.assert_called_once()
    assert len(launches) == 1
    command, launch = launches[0]
    assert command[command.index("--resume") + 1] == "original-worker-session"
    assert "--no-restore-cwd" in command
    prompt = command[command.index("-q") + 1]
    assert ANSWER["resume_instruction"] in prompt and QUESTION["question"] in prompt
    assert launch["cwd"] == str(worker.workspace)
    assert launch["env"]["HERMES_KANBAN_RUN_ID"] == str(worker.task.current_run_id)
    assert launch["env"]["HERMES_KANBAN_TASK"] == worker.task.id
    assert launch["env"]["HERMES_KANBAN_WORKSPACE"] == str(worker.workspace)
    assert launch["env"]["HERMES_KANBAN_BRANCH"] == worker.task.branch_name
    assert launch["env"]["HERMES_KANBAN_CLAIM_LOCK"] == worker.task.claim_lock


def test_rolled_back_resume_launch_cannot_execute_or_replace_original_run(
        worker, monkeypatch, all_assignees_spawnable):
    launches, live_children = [], set()
    monkeypatch.setattr(dispatch, "_worker_alive", lambda pid, _fingerprint: pid in live_children)
    monkeypatch.setattr(dispatch, "_restart_safe_worker_argv", lambda _task, command: command)

    def popen(command, **kwargs):
        child_pid = 987654322 + len(launches)
        launches.append((command, kwargs))
        live_children.add(child_pid)
        return SimpleNamespace(pid=child_pid)

    monkeypatch.setattr(subprocess, "Popen", popen)
    with kbc.connect_closing() as conn:
        question = _ask(conn, worker)
        _answer(conn, worker, question)
        # The OS child exists when the durable PID write fails. A rollback
        # must not leave that orphan able to continue the Worker activity.
        conn.execute("CREATE TRIGGER fail_resume_pid BEFORE UPDATE OF worker_pid ON tasks "
                     "WHEN NEW.worker_pid != OLD.worker_pid "
                     "BEGIN SELECT RAISE(ABORT, 'injected PID persistence failure'); END")
        conn.commit()
        assert dispatch.dispatch_once(conn).spawned == []
        assert len(launches) == 1
        _assert_same_execution(conn, worker, "needs_direction")
        durable = direction.latest_direction(conn, worker.task.id)
        assert durable["response"] == ANSWER
        assert durable["resume_token"] is durable["resumed_at"] is durable["resume_admitted_at"] is None
        assert _admit(launches[0][1]) is False
        _assert_same_execution(conn, worker, "needs_direction")
        conn.execute("DROP TRIGGER fail_resume_pid")
        conn.commit()
        result = dispatch.dispatch_once(conn)
        assert result.spawned == [(worker.task.id, worker.task.assignee, str(worker.workspace))]
        assert len(launches) == 2
        first_token = launches[0][1]["env"]["HERMES_KANBAN_DIRECTION_TOKEN"]
        second_token = launches[1][1]["env"]["HERMES_KANBAN_DIRECTION_TOKEN"]
        assert first_token != second_token
        assert _admit(launches[0][1]) is False
        assert _admit(launches[1][1]) is True
        assert _admit(launches[1][1]) is False
        _assert_same_execution(conn, worker, "running")
        _answer(conn, worker, question)
        assert dispatch.dispatch_once(conn).spawned == []
        assert len(launches) == 2
        events = kb.list_events(conn, worker.task.id)
        for kind in ("claimed", "developer_response", "direction_resumed", "direction_resume_launched"):
            assert sum(e.kind == kind for e in events) == 1
