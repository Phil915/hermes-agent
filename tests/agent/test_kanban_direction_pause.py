"""Decision pauses preserve the running agent and cannot run pre-answer tool batches."""

import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.iteration_budget import IterationBudget
from agent.turn_tool_round import run_tool_round
from tests.agent.test_sequential_tool_timeout import _make_agent


def _call(name, call_id):
    return SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(name=name, arguments="{}"),
    )


def _round(agent, calls, messages):
    return run_tool_round(
        agent, assistant_message=SimpleNamespace(content="", tool_calls=calls, reasoning=None),
        finish_reason="tool_calls", messages=messages, conversation_history=[], api_call_count=1,
        effective_task_id="worker-session", user_message="Implement the card", system_message=None,
        active_system_prompt=None, compression_attempts=0, max_compression_attempts=3,
        final_response=None, failed=False, _turn_exit_reason="unknown",
        truncated_tool_call_retries=0, current_turn_user_idx=0,
    )


@pytest.mark.parametrize("direction_first", [True, False])
@pytest.mark.parametrize("deferred", [True, False])
def test_direction_must_be_alone_before_any_tool_side_effect(tmp_path, monkeypatch, direction_first, deferred):
    agent = _make_agent(tmp_path)
    agent.valid_tool_names.update({"kanban_needs_direction", "write_file", "tool_call"})
    calls = [_call("kanban_needs_direction", "ask"), _call("write_file", "write")]
    if deferred:
        monkeypatch.setattr("tools.tool_search.load_config_readonly", lambda: SimpleNamespace(
            effective_defer_tools=frozenset({"kanban_needs_direction"})))
        calls[0].function.name = "tool_call"
        calls[0].function.arguments = json.dumps({"name": "kanban_needs_direction", "arguments": {}})
    if not direction_first:
        calls.reverse()
    dispatch = Mock()
    monkeypatch.setattr(agent, "_execute_tool_calls", dispatch)
    messages = [{"role": "user", "content": "Implement the card"}]
    try:
        verdict = _round(agent, calls, messages)
        assert verdict.action == "continue"
        dispatch.assert_not_called()
        assert [row["role"] for row in messages] == ["user", "assistant", "tool", "tool"]
        assert [row["tool_call_id"] for row in messages[2:]] == [call.id for call in calls]
        assert all("No calls in this batch ran" in row["content"] for row in messages[2:])
        agent._flush_messages_to_session_db.assert_called()
    finally:
        if agent._session_db is not None:
            agent._session_db.close()


@pytest.mark.parametrize("answered", [True, False])
def test_only_answered_direction_refunds_the_decision_iteration(tmp_path, monkeypatch, answered):
    from agent.turn_preflight import PostToolCompressionVerdict

    agent = _make_agent(tmp_path)
    agent.valid_tool_names.add("kanban_needs_direction")
    agent.iteration_budget = IterationBudget(1)
    agent.max_iterations = 1
    assert agent.iteration_budget.consume()
    result = (
        {"ok": True, "status": "running", "question_id": "question-1", "response": {
            "decision": "Keep the current interface", "rationale": "Canon already defines it",
            "resume_instruction": "Continue on the same card",
        }} if answered else {"error": "Routine implementation details do not need upstream direction"}
    )

    def execute(assistant, messages, *_):
        messages.append({"role": "tool", "name": "kanban_needs_direction",
                         "tool_call_id": assistant.tool_calls[0].id,
                         "content": json.dumps(result) + "\n[iteration budget checkpoint]"})

    def unchanged_compression(_agent, **kw):
        return PostToolCompressionVerdict(
            end_turn=False, **{key: kw[key] for key in (
                "messages", "active_system_prompt", "conversation_history", "compression_attempts",
                "final_response", "turn_exit_reason", "current_turn_user_idx",
            )},
        )

    monkeypatch.setattr(agent, "_execute_tool_calls", execute)
    monkeypatch.setattr("agent.turn_tool_round.compress_after_tool_results", unchanged_compression)
    messages = [{"role": "user", "content": "Implement the card"}]
    try:
        verdict = _round(agent, [_call("kanban_needs_direction", "ask")], messages)
        assert verdict.action == "continue"
        assert agent.iteration_budget.remaining == int(answered)
        assert agent.max_iterations == (2 if answered else 1)
        assert not hasattr(verdict, "api_call_count")  # the real request count remains unchanged
        assert json.loads(messages[-1]["content"].split("\n")[0]) == result
    finally:
        if agent._session_db is not None:
            agent._session_db.close()


def test_direction_wait_exceeds_tool_deadline_without_spending_run_time(tmp_path, monkeypatch):
    from agent.tool_executor import execute_tool_calls_sequential

    agent = _make_agent(tmp_path)
    agent.valid_tool_names.add("kanban_needs_direction")
    started = agent._run_budget_started_at = time.time()
    release = threading.Event()

    def dispatch(*_, **__):
        threading.Timer(0.05, release.set).start()
        assert release.wait(timeout=2)
        return json.dumps({"ok": True, "status": "running", "question_id": "question-1", "response": {}})

    monkeypatch.setattr("agent.tool_executor._resolve_sequential_tool_timeout", lambda: 0.001)
    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    messages = []
    try:
        execute_tool_calls_sequential(
            agent, SimpleNamespace(tool_calls=[_call("kanban_needs_direction", "ask")]),
            messages, "worker-session",
        )
        assert json.loads(messages[-1]["content"])["question_id"] == "question-1"
        assert agent._run_budget_started_at - started >= 0.05
    finally:
        release.set()
        if agent._session_db is not None:
            agent._session_db.close()


def test_exhausted_interrupted_wait_does_not_fail_the_paused_card(tmp_path, monkeypatch):
    from agent.turn_finalizer import _record_kanban_budget_exhausted
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Choose an upstream interface", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'needs_direction' WHERE id = ?", (task_id,))
        run_id = claimed.current_run_id

    _record_kanban_budget_exhausted(task_id, 1, 1, logging.getLogger(__name__))

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "needs_direction"
        assert task.current_run_id == run_id
        assert task.consecutive_failures == 0
        assert kb.latest_run(conn, task_id).ended_at is None
