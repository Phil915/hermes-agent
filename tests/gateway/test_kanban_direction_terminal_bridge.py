"""Cached coordinator turns retain authority across the real terminal/Python bridge."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from agent.agent_init import _publish_session_id
from gateway.config import GatewayConfig, Platform
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource, SessionStore, build_session_context
from gateway.session_identity import restore_identity
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch, kanban_db_notify as notify
from hermes_cli import kanban_direction as direction
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from tools.environments.local import LocalEnvironment

from tests.hermes_cli.test_kanban_direction import ANSWER, QUESTION


OWNER = "strategic-industrial-intelligence"
REPO = Path(__file__).resolve().parents[2]


class RecordingAdapter:
    supports_async_delivery = True

    def __init__(self):
        self.handled = []
        self.homes = []

    async def send(self, *_args, **_kwargs):
        pass

    async def handle_message(self, event):
        self.handled.append(event)
        self.homes.append(get_hermes_home())
        event._gateway_accepted = True


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    import hermes_state

    home = tmp_path / ".hermes"
    owner_home, worker_home = (home / "profiles" / name for name in (OWNER, "worker"))
    for directory in (home, owner_home, worker_home):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text(
            "toolsets: [kanban]\nterminal:\n  auto_source_bashrc: false\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    # Restore runtime profile resolution after the suite's fixed test-home override.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    for name in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK",
                 "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME", "HERMES_KANBAN_GOAL_MODE",
                 "HERMES_KANBAN_DIRECTION_TOKEN", "HERMES_DELEGATED_CHILD_CONTEXT", "HERMES_BIN"):
        monkeypatch.delenv(name, raising=False)
    source = SessionSource(platform=Platform.DISCORD, chat_id="fixture-channel", chat_type="group",
                           user_id="fixture-user", scope_id="fixture-guild", profile=OWNER)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: RecordingAdapter()}
    runner._profile_adapters = {OWNER: {}}
    runner._primary_profile_name = runner._kanban_notifier_profile = "default"
    runner._kanban_dispatcher_lock_handle = object()
    runner.config = GatewayConfig(multiplex_profiles=True, profile_routes=parse_profile_routes([
        dict(platform="discord", guild_id=source.scope_id, chat_id=source.chat_id, profile=OWNER),
    ]))
    restore_identity(source, runner=runner, transport_profile="default")
    runner.session_store = SessionStore(home / "sessions", runner.config)
    with _profile_runtime_scope(owner_home):
        entry = runner.session_store.get_or_create_session(source)
        terminal = LocalEnvironment(cwd=str(REPO), timeout=20)
    # The gateway's persisted routing entry, not a caller's tool argument, owns the raw id.
    assert entry.transport_profile == "default"
    owner_state = SessionDB(owner_home / "state.db", read_only=True)
    try:
        assert owner_state.get_session(entry.session_id) is not None
    finally:
        owner_state.close()
    context = build_session_context(source, runner.config, entry)
    agent = SimpleNamespace(session_id=entry.session_id)
    runner._agent_cache = {entry.session_key: (agent, "fixture-config", 0, entry.session_id)}
    runner._agent_cache_lock = threading.Lock()
    repo, workspace = tmp_path / "fixture-repo", tmp_path / "same-worktree"
    for argv in (
        ["git", "init", "--initial-branch=main", str(repo)],
        ["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.org",
         "commit", "--allow-empty", "-m", "fixture"],
        ["git", "-C", str(repo), "worktree", "add", "-b", "worker/same-activity", str(workspace)],
    ):
        subprocess.run(argv, check=True, capture_output=True)
    (workspace / "activity.txt").write_text("Preserved activity")
    worker_session = "fixture-worker-session"
    state = SessionDB(worker_home / "state.db")
    try:
        state.create_session(worker_session, source="kanban", cwd=str(workspace))
        state.append_message(worker_session, role="user", content="Continue the same fixture activity.")
    finally:
        state.close()
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Bridge recovery fixture", assignee="worker",
                                 created_by=OWNER, session_id=entry.session_id,
                                 workspace_kind="worktree", workspace_path=str(workspace),
                                 branch_name="worker/same-activity", max_runtime_seconds=300)
        notify.add_notify_sub(conn, task_id=task_id, platform="discord", chat_id=source.chat_id,
                              chat_type=source.chat_type, user_id=source.user_id,
                              notifier_profile=OWNER, delivery_mode="notify+wake",
                              delivery_metadata={"scope_id": source.scope_id})
        task = kb.claim_task(conn, task_id)
        conn.execute("UPDATE task_runs SET metadata = ? WHERE id = ?",
                     (json.dumps({"activity": {"id": "same-activity", "correction_cycles": 1}}),
                      task.current_run_id))
        conn.commit()
        original_run = dict(conn.execute("SELECT * FROM task_runs WHERE id = ?",
                                        (task.current_run_id,)).fetchone())
    case = SimpleNamespace(runner=runner, context=context, entry=entry, terminal=terminal,
                           home=home, owner_home=owner_home, worker_home=worker_home,
                           worker_session=worker_session, task=task, run=original_run,
                           workspace=workspace, children=[])
    try:
        yield case
    finally:
        for child in case.children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)
        terminal.cleanup()
        runner.session_store.close_all_db_handles()
        runner._clear_session_env([])


def _terminal_answer(case, args, *, context=None, fresh=False):
    """Production bridge: terminal -> Python -> hermes_bootstrap -> existing handler."""
    context = context or case.context
    script = (
        "import hermes_bootstrap; import json; "
        "from gateway.session_context import get_session_env; "
        "from tools.kanban_tools import _handle_answer_direction; "
        f"args = json.loads({json.dumps(args)!r}); "
        "print(json.dumps({'session_id': get_session_env('HERMES_SESSION_ID', ''), "
        "'answer': json.loads(_handle_answer_direction(args)) if args else None}))"
    )
    command = f"HERMES_PROFILE={shlex.quote(OWNER)} " + shlex.join([sys.executable, "-c", script])
    with _profile_runtime_scope(case.owner_home):
        tokens = case.runner._set_session_env(context)
        try:
            if fresh:
                # A new AIAgent publishes this after the gateway binds the turn.
                _publish_session_id(case.entry.session_id)
            else:
                turn = TurnRunner(case.runner, SimpleNamespace(
                    session_key=case.entry.session_key, session_id=case.entry.session_id,
                    _interrupt_depth=0))
                found = turn._lookup_cached_agent(
                    "fixture-config", case.runner._agent_cache_lock, case.runner._agent_cache,
                    10, case.entry.session_id, False, 0)
                assert found.reused and found.agent.session_id == case.entry.session_id
            result = case.terminal.execute(command)
        finally:
            case.runner._clear_session_env(tokens)
    assert result["returncode"] == 0, result
    return json.loads(result["output"].splitlines()[-1])


def _pause_worker(case):
    env = dict(os.environ, HERMES_HOME=str(case.worker_home), HERMES_PROFILE="worker",
               HERMES_KANBAN_TASK=case.task.id, HERMES_KANBAN_RUN_ID=str(case.task.current_run_id),
               HERMES_KANBAN_CLAIM_LOCK=case.task.claim_lock, HERMES_SESSION_ID=case.worker_session)
    script = (
        "import hermes_bootstrap; import json; from tools.kanban_tools import _handle_needs_direction; "
        f"print(_handle_needs_direction(json.loads({json.dumps({'developer_question': QUESTION})!r})), flush=True)"
    )
    child = subprocess.Popen([sys.executable, "-c", script], cwd=REPO, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    case.children.append(child)
    with kbc.connect_closing() as conn:
        dispatch._set_worker_pid(conn, case.task.id, child.pid)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with kbc.connect_closing() as conn:
            question = direction.latest_direction(conn, case.task.id)
        if question is not None:
            assert child.poll() is None, child.communicate(timeout=5)
            return child, question
        assert child.poll() is None, child.communicate(timeout=5)
        time.sleep(0.02)
    pytest.fail("Worker did not persist its question")


def _assert_same_run(case, status):
    with kbc.connect_closing() as conn:
        current = kb.get_task(conn, case.task.id)
        for field in ("id", "current_run_id", "assignee", "workspace_kind", "workspace_path",
                      "branch_name", "claim_lock", "consecutive_failures", "block_recurrences",
                      "max_retries", "goal_max_turns"):
            assert getattr(current, field) == getattr(case.task, field), field
        assert current.status == status
        runs = conn.execute("SELECT * FROM task_runs WHERE task_id = ?", (case.task.id,)).fetchall()
        assert len(runs) == 1 and runs[0]["id"] == case.run["id"] == case.task.current_run_id
        for field in ("metadata", "started_at", "outcome", "ended_at", "terminal_result"):
            assert runs[0][field] == case.run[field], field
        assert len(kb.list_tasks(conn)) == 1
    assert (case.workspace / "activity.txt").read_text() == "Preserved activity"
    gitdir = Path((case.workspace / ".git").read_text().removeprefix("gitdir: ").strip())
    assert (gitdir / "HEAD").read_text().strip() == "ref: refs/heads/worker/same-activity"


def test_process_loss_answer_uses_persisted_cached_coordinator_identity(bridge, monkeypatch):
    case = bridge
    assert _terminal_answer(case, {}, fresh=True)["session_id"] == case.entry.session_id
    child, question = _pause_worker(case)
    assert question["owner_session_id"] == case.entry.session_id
    assert question["worker_session_id"] == case.worker_session
    _assert_same_run(case, "needs_direction")
    child.kill()
    child.communicate(timeout=10)
    assert child.poll() is not None
    with kbc.connect_closing() as conn:
        fingerprint = conn.execute("SELECT worker_started_at FROM task_runs WHERE id = ?",
                                   (case.task.current_run_id,)).fetchone()[0]
        assert not dispatch._worker_alive(child.pid, fingerprint)
        assert dispatch.dispatch_once(conn).spawned == []  # Missing answer is safely paused.
    rows = _notifier_collect(case.runner, kb, notifier_profile="default", gc_due=False, gc_retention_days=30)
    assert len(rows) == 1

    async def wake():
        await _KanbanNotification(case.runner, rows[0], platform_cls=Platform, sub_fail_counts={}).deliver()

    asyncio.run(wake())
    adapter = case.runner.adapters[Platform.DISCORD]
    assert len(adapter.handled) == 1 and adapter.homes == [case.owner_home]
    event = adapter.handled[0]
    assert event.source.profile == OWNER and question["id"] in event.text
    with _profile_runtime_scope(case.owner_home):
        persisted = case.runner.session_store.get_or_create_session(event.source, touch_activity=False)
    assert persisted.session_id == case.entry.session_id
    context = build_session_context(event.source, case.runner.config, persisted)
    args = dict(task_id=case.task.id, question_id=question["id"], response=ANSWER)
    result = _terminal_answer(case, args, context=context)
    assert result["session_id"] == persisted.session_id
    assert result["answer"]["ok"], result
    with kbc.connect_closing() as conn:
        assert direction.latest_direction(conn, case.task.id)["response"] == ANSWER
    _assert_same_run(case, "needs_direction")

    launches = []

    def capture_launch(command, **kwargs):
        launches.append((command, kwargs))
        # Only the restored model process is intercepted. Its live PID keeps
        # subsequent dispatcher sweeps from treating the fixture as another loss.
        return SimpleNamespace(pid=os.getpid())

    with monkeypatch.context() as launch_patch:
        launch_patch.setattr(subprocess, "Popen", capture_launch)
        launch_patch.setattr(dispatch, "_restart_safe_worker_argv", lambda _task, argv: argv)
        with kbc.connect_closing() as conn:
            resumed = dispatch.dispatch_once(conn)
    assert resumed.spawned == [(case.task.id, "worker", str(case.workspace))]
    assert len(launches) == 1
    command, launch = launches[0]
    assert command[command.index("--resume") + 1] == case.worker_session
    assert ANSWER["resume_instruction"] in command[command.index("-q") + 1]
    assert launch["cwd"] == str(case.workspace)
    assert launch["env"]["HERMES_KANBAN_RUN_ID"] == str(case.task.current_run_id)
    assert launch["env"]["HERMES_KANBAN_BRANCH"] == case.task.branch_name
    from hermes_cli.kanban_direction_admission import (
        admit_prepared_direction_resume, prepare_direction_resume_from_env,
    )
    with monkeypatch.context() as admission_env:
        for name in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK",
                     "HERMES_KANBAN_DIRECTION_TOKEN"):
            admission_env.setenv(name, launch["env"][name])
        admission = prepare_direction_resume_from_env(worker_session_id=case.worker_session)
        assert admission is not None and admission is not False
        state = SessionDB(case.worker_home / "state.db", read_only=True)
        try:
            history = state.get_messages(case.worker_session)
        finally:
            state.close()
        shell = SimpleNamespace(session_id=case.worker_session, agent=object(),
                                _direction_resume_admission=admission)
        assert admit_prepared_direction_resume(shell, conversation_history=history)
        shell._direction_resume_admission = admission
        assert not admit_prepared_direction_resume(shell, conversation_history=history)
    _assert_same_run(case, "running")
    for _ in range(2):
        replay = _terminal_answer(case, args, context=context)
        assert replay["answer"]["ok"] and replay["answer"]["resumed"]
        with kbc.connect_closing() as conn:
            assert dispatch.dispatch_once(conn).spawned == []
    _assert_same_run(case, "running")
    with kbc.connect_closing() as conn:
        events = kb.list_events(conn, case.task.id)
        for kind in ("claimed", "developer_question", "developer_response", "direction_resume_launched", "direction_resumed"):
            assert sum(event.kind == kind for event in events) == 1, kind


@pytest.mark.parametrize("responder_id", ["", "unrelated-coordinator-session"])
def test_terminal_bridge_cannot_borrow_or_supply_owner_session(bridge, monkeypatch, responder_id):
    case = bridge
    child, question = _pause_worker(case)
    child.kill()
    child.communicate(timeout=10)
    monkeypatch.setenv("HERMES_SESSION_ID", case.entry.session_id)  # Stale process-global mirror.
    args = dict(task_id=case.task.id, question_id=question["id"], response=ANSWER)
    result = _terminal_answer(case, args, context=replace(case.context, session_id=responder_id))
    assert result["session_id"] == responder_id
    assert "only the owning coordinator session" in result["answer"]["error"]
    supplied = _terminal_answer(case, {**args, "session_id": case.entry.session_id},
                                context=replace(case.context, session_id=responder_id))
    assert "error" in supplied["answer"]
    with kbc.connect_closing() as conn:
        assert direction.latest_direction(conn, case.task.id)["response"] is None
    _assert_same_run(case, "needs_direction")
