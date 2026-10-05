"""An uncommitted direction launch cannot reach the model or claim its session."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli.kanban_direction_admission import admit_direction_resume_from_env


@pytest.mark.parametrize("quiet", [True, False])
def test_rejected_resume_exits_before_any_single_query_execution(monkeypatch, quiet):
    import cli
    from hermes_cli.cli_single_query import _run_single_query_mode

    admit = Mock(return_value=False)
    route = Mock()
    worker = SimpleNamespace(session_id="same-worker-session", _claim_active_session=Mock())
    monkeypatch.setattr("hermes_cli.kanban_direction_admission.admit_direction_resume_from_env", admit)
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
    assert admit_direction_resume_from_env(worker_session_id="ordinary-session") is True
    connect.assert_not_called()


def test_invalid_resume_identity_removes_nonce_before_exit(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DIRECTION_TOKEN", "launch-nonce")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "not-a-run")
    assert admit_direction_resume_from_env(worker_session_id="worker-session") is False
    import os
    assert "HERMES_KANBAN_DIRECTION_TOKEN" not in os.environ
