"""A direction process restore continues the already charged Kanban goal turn."""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import goals, kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch, kanban_direction as direction
from hermes_cli.cli_single_query import _run_kanban_goal_loop_q
from tests.hermes_cli.test_kanban_direction import _admit, _answer, _ask, worker


class _PausedProcessLost(BaseException):
    """A killed process cannot unwind the goal loop as an ordinary turn error."""


@pytest.mark.parametrize("finalize_nudge", [False, True])
def test_process_loss_preserves_goal_budget_and_finalize_nudge(
        worker, monkeypatch, all_assignees_spawnable, finalize_nudge):
    task_id, run_id = worker.task.id, worker.task.current_run_id
    max_turns = 5 if finalize_nudge else 3
    with kbc.connect_closing() as conn:
        conn.execute("UPDATE tasks SET goal_mode = 1, goal_max_turns = ? WHERE id = ?",
                     (max_turns, task_id))
        conn.commit()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    verdicts = iter(["continue", "done" if finalize_nudge else "continue",
                     "done" if finalize_nudge else "continue"])
    monkeypatch.setattr(goals, "judge_goal", lambda *_args, **_kwargs:
                        (next(verdicts), "Unfinished activity", False, None, False))
    turns, questions = [], []

    def initial_process_turn(prompt):
        turns.append(prompt)
        with kbc.connect_closing() as conn:
            progress = direction.goal_progress(conn, task_id, run_id)
            # The continuation is charged BEFORE its model/tool work starts.
            assert progress["turns_used"] == len(turns) + 1
            if len(turns) == 2:
                questions.append(_ask(conn, worker))
                raise _PausedProcessLost
        return "Unfinished activity"

    with pytest.raises(_PausedProcessLost):
        _run_kanban_goal_loop_q(SimpleNamespace(), "First turn", run_turn=initial_process_turn)
    expected_progress = {"turns_used": 3, "nudged_to_finalize": finalize_nudge}
    expected_metadata = {**json.loads(worker.run["metadata"]), "kanban_goal_progress": expected_progress}
    original_pid, restored_pid = worker.task.worker_pid, worker.task.worker_pid + 1
    monkeypatch.setattr(dispatch, "_worker_alive", lambda pid, _fp: pid == restored_pid)
    monkeypatch.setattr(dispatch, "_restart_safe_worker_argv", lambda _task, command: command)
    launches = []

    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return SimpleNamespace(pid=restored_pid)

    monkeypatch.setattr(subprocess, "Popen", popen)
    with kbc.connect_closing() as conn:
        paused = kb.get_task(conn, task_id)
        assert paused.status == "needs_direction" and paused.worker_pid == original_pid
        assert direction.goal_progress(conn, task_id, run_id) == expected_progress
        assert dispatch.dispatch_once(conn).spawned == []  # process loss alone cannot resume
        _answer(conn, worker, questions[0])
        assert dispatch.dispatch_once(conn).spawned == [(task_id, "worker", str(worker.workspace))]
    # The real launch admission consumes the durable answer on the original run.
    assert len(launches) == 1
    command, launch = launches[0]
    assert command[command.index("--resume") + 1] == "original-worker-session"
    assert launch["env"]["HERMES_KANBAN_RUN_ID"] == str(run_id)
    assert _admit(launch) is True
    assert _admit(launch) is False
    with kbc.connect_closing() as conn:
        resumed = kb.get_task(conn, task_id)
        assert resumed.status == "running" and resumed.current_run_id == run_id
        for field in ("id", "assignee", "workspace_path", "branch_name", "consecutive_failures",
                      "block_recurrences", "max_retries"):
            assert getattr(resumed, field) == getattr(worker.task, field), field
        runs = kb.list_runs(conn, task_id)
        assert len(runs) == 1 and runs[0].id == run_id
        assert runs[0].metadata == expected_metadata
        _answer(conn, worker, questions[0])  # answer replay cannot advance the goal counter
        assert dispatch.dispatch_once(conn).spawned == []
        assert direction.goal_progress(conn, task_id, run_id) == expected_progress

    # The restored CLI's initial turn finishes the interrupted third turn. The
    # loop must not reset to turn one, or issue a second finalize nudge.
    extra_turn = Mock(side_effect=AssertionError("direction recovery must not grant extra goal turns"))
    _run_kanban_goal_loop_q(SimpleNamespace(), "Restored third turn", run_turn=extra_turn)
    extra_turn.assert_not_called()
    with kbc.connect_closing() as conn:
        runs = kb.list_runs(conn, task_id)
        assert len(runs) == 1 and runs[0].id == run_id
        assert runs[0].outcome == "blocked"
        assert ("after a finalize nudge" if finalize_nudge else "(3/3)") in runs[0].summary
        assert runs[0].metadata == expected_metadata
        assert kb.get_task(conn, task_id).consecutive_failures == worker.task.consecutive_failures
    assert (worker.workspace / "activity.txt").read_text() == "existing activity and unfinished changes"
    assert (worker.gitdir / "HEAD").read_text().strip() == "ref: refs/heads/worker/existing"
