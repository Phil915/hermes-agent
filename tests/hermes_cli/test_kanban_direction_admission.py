"""An uncommitted direction launch cannot reach the model or claim its session."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli.kanban_direction_admission import prepare_direction_resume_from_env
from tests.hermes_cli.test_kanban_direction import _answer, _ask, _assert_same_execution, worker


@pytest.mark.parametrize("quiet", [True, False])
def test_rejected_resume_exits_before_any_single_query_execution(monkeypatch, quiet):
    import cli
    from hermes_cli.cli_single_query import _run_single_query_mode

    admit = Mock(return_value=False)
    route = Mock()
    worker = SimpleNamespace(session_id="same-worker-session", _claim_active_session=Mock())
    monkeypatch.setattr("hermes_cli.kanban_direction_admission.prepare_direction_resume_from_env", admit)
    monkeypatch.setattr(cli, "_should_seed_interactive", route)
    with pytest.raises(SystemExit) as exited:
        _run_single_query_mode(worker, "Continue after the decision", None, quiet, True)
    assert exited.value.code == 0
    admit.assert_called_once_with(worker_session_id="same-worker-session")
    route.assert_not_called()
    worker._claim_active_session.assert_not_called()


def test_ordinary_query_without_resume_token_does_not_open_board(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_DIRECTION_TOKEN", raising=False)
    connect = Mock(side_effect=AssertionError("ordinary chat must not open a Kanban board"))
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect_closing", connect)
    assert prepare_direction_resume_from_env(worker_session_id="ordinary-session") is None
    connect.assert_not_called()


def test_invalid_resume_identity_removes_nonce_before_exit(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DIRECTION_TOKEN", "launch-nonce")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "not-a-run")
    assert prepare_direction_resume_from_env(worker_session_id="worker-session") is False
    import os
    assert "HERMES_KANBAN_DIRECTION_TOKEN" not in os.environ


def _pending_restore(worker, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        question = _ask(conn, worker)
        _answer(conn, worker, question)
        with kb.write_txn(conn):
            conn.execute("UPDATE developer_questions SET resume_token = ? WHERE id = ?",
                         ("committed-launch", question["id"]))
    monkeypatch.setenv("HERMES_KANBAN_TASK", worker.task.id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(worker.task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", worker.task.claim_lock)
    monkeypatch.setenv("HERMES_KANBAN_DIRECTION_TOKEN", "committed-launch")
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    return question


@pytest.mark.parametrize("quiet", [False, True])
@pytest.mark.parametrize("refused", ["session_cap", "credentials", "agent_init"])
def test_startup_refusal_keeps_answer_and_original_run_paused(worker, monkeypatch, quiet, refused):
    import cli
    from hermes_cli import kanban_db_connect as kbc, kanban_db_dispatch as dispatch, kanban_direction
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    from hermes_cli.cli_single_query import _run_single_query_mode

    _pending_restore(worker, monkeypatch)
    model = Mock(side_effect=AssertionError("startup rejection must not call the model"))
    shell = SimpleNamespace(
        session_id="original-worker-session", agent=SimpleNamespace(run_conversation=model),
        _claim_active_session=Mock(return_value=refused != "session_cap"),
        _ensure_runtime_credentials=Mock(return_value=refused != "credentials"),
        _init_agent=Mock(return_value=False), _active_agent_route_signature="same-route",
        _resolve_turn_agent_config=lambda _: dict(signature="same-route", model="fixture", runtime={}),
        _secret_capture_callback=None, console=Mock(), _show_security_advisories=Mock(),
        _print_exit_summary=Mock(), _last_turn_result=None, _release_active_session=Mock(),
    )
    shell.chat = lambda query, images=None: CLIChatTurnMixin.chat(shell, query, images)
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *_: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda query, _: (query, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda _: [])
    monkeypatch.setattr(cli, "_route_single_query_images", lambda _cli, _query, query, *_: query)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda current: current._release_active_session())
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: SimpleNamespace())
    with pytest.raises(SystemExit):
        _run_single_query_mode(shell, "Resume with the durable answer", None, quiet, True)
    model.assert_not_called()
    assert shell._release_active_session.call_count == int(refused != "session_cap")
    with kbc.connect_closing() as conn:
        _assert_same_execution(conn, worker, "needs_direction")
        durable = kanban_direction.latest_direction(conn, worker.task.id)
        assert durable["response"] is not None
        assert durable["resumed_at"] is durable["resume_admitted_at"] is None
        # Process startup refusal is not a failed attempt or permission to retry
        # the card; the normal crash detector must leave its open run alone.
        monkeypatch.setattr(dispatch, "_worker_alive", lambda *_: False)
        assert dispatch.detect_crashed_workers(conn) == []
        _assert_same_execution(conn, worker, "needs_direction")


@pytest.mark.parametrize("quiet", [False, True])
def test_ready_restore_loads_worker_history_before_consuming_once(worker, monkeypatch, quiet):
    import os
    import cli
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_direction
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    from hermes_cli.cli_single_query import _run_quiet_single_query
    from hermes_cli.kanban_direction_admission import prepare_direction_resume_from_env
    from hermes_state import SessionDB

    question = _pending_restore(worker, monkeypatch)
    board_path = str(kb.kanban_db_path())
    admission = prepare_direction_resume_from_env(worker_session_id="original-worker-session")
    assert admission is not None and admission is not False
    assert "HERMES_KANBAN_DIRECTION_TOKEN" not in os.environ
    monkeypatch.setenv("HERMES_KANBAN_DB", board_path)
    monkeypatch.setenv("HERMES_HOME", str(worker.worker_home))
    db = SessionDB(worker.worker_home / "state.db")
    db.end_session("original-worker-session", "compression")
    db.create_session("worker-tip", source="kanban", parent_session_id="original-worker-session")
    db.append_message("worker-tip", role="user", content="The same Worker activity after compression.")
    shell = cli.HermesCLI.__new__(cli.HermesCLI)
    shell.session_id = "original-worker-session"
    shell._session_db = db
    shell.conversation_history = []
    shell.tool_progress_mode = "off"
    shell._restore_session_state = Mock()
    shell._direction_resume_admission = admission
    prompt = "Resume this same Worker activity. " + question["id"] + " Keep the existing adapter."
    calls = []

    class ReachedModel(BaseException):
        pass

    def run_conversation(**kwargs):
        calls.append(kwargs)
        with kbc.connect_closing() as conn:
            assert kb.get_task(conn, worker.task.id).status == "running"
            assert kanban_direction.latest_direction(conn, worker.task.id)["resumed_at"] is not None
        raise ReachedModel

    try:
        # Exercise the real --resume loader, including its compression lineage
        # resolution and stored-history reconstruction, before final admission.
        assert shell._load_resumed_history_late()
        assert shell.session_id == "worker-tip"
        history = list(shell.conversation_history)
        shell.agent = SimpleNamespace(run_conversation=run_conversation)
        for name in ("_sudo_password_callback", "_approval_callback", "_secret_capture_callback",
                     "_vault_unlock_callback", "_vault_save_login_callback", "_vault_code_callback"):
            setattr(shell, name, None)
        shell._flush_credit_notices = Mock()
        turn = SimpleNamespace(voice_prefix="", stream_callback=None)
        monkeypatch.setattr("hermes_cli.quiet_single_query.adopt_unanswered_turn", lambda *_: None)

        def execute():
            if quiet:
                _run_quiet_single_query(shell, prompt)
            else:
                shell.conversation_history = history + [{"role": "user", "content": prompt}]
                CLIChatTurnMixin._chat_run_agent(shell, turn, prompt)

        with pytest.raises(ReachedModel):
            execute()
        assert calls[0]["conversation_history"] == history
        assert calls[0]["user_message"] == prompt
        # A second already-prepared copy of the same launch cannot consume or
        # invoke the model again, even though it passed early verification.
        shell._direction_resume_admission = admission
        if quiet:
            with pytest.raises(SystemExit) as exited:
                execute()
            assert exited.value.code == 0
        else:
            execute()
            assert turn.result["completed"] is False
            # A copy can have passed early verification before the winning
            # process consumed the nonce. Top-level -q must then stop before
            # the goal judge can act on that other process's running run.
            from hermes_cli.cli_single_query import _run_single_query_mode
            shell._claim_active_session = Mock(return_value=True)
            shell._show_security_advisories = Mock()
            shell.console = Mock()
            shell.chat = lambda query, images=None: CLIChatTurnMixin._chat_run_agent(shell, turn, query)
            goal_loop = Mock(side_effect=AssertionError("rejected launch cannot judge another process's run"))
            monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
            monkeypatch.setattr("hermes_cli.kanban_direction_admission.prepare_direction_resume_from_env",
                                lambda **_: admission)
            monkeypatch.setattr(cli, "_should_seed_interactive", lambda *_: False)
            monkeypatch.setattr(cli, "_collect_query_images", lambda query, _: (query, []))
            monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda _: [])
            monkeypatch.setattr(cli, "_finalize_single_query", lambda _: None)
            monkeypatch.setattr(cli, "_run_kanban_goal_loop_chat", goal_loop)
            monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: SimpleNamespace())
            with pytest.raises(SystemExit) as exited:
                _run_single_query_mode(shell, prompt, None, False, True)
            assert exited.value.code == 0
            goal_loop.assert_not_called()
        assert len(calls) == 1
        with kbc.connect_closing() as conn:
            _assert_same_execution(conn, worker, "running")
            assert len([e for e in kb.list_events(conn, worker.task.id) if e.kind == "direction_resumed"]) == 1
    finally:
        db.close()
