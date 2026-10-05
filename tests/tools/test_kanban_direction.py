"""Direction is a durable pause inside the same Worker tool/conversation."""
from __future__ import annotations

from copy import deepcopy
import json
import sqlite3

import pytest


QUESTION = {
    "question": "Which published interface owns the compatibility mapping?",
    "why_it_matters": "Choosing incorrectly would require a public migration.",
    "evidence": {"references": ["canon/interfaces.md:42", "src/mapping.py:19"],
                 "precedent_status": "conflicting"},
    "options": ["Keep the mapping in the existing adapter", "Move it into the public API"],
    "worker_recommendation": "Keep the mapping in the adapter for compatibility.",
    "decision_needed": "migration_or_public_interface",
}
RESPONSE = {
    "decision": "Keep the mapping in the existing adapter.",
    "rationale": "Preserves the published compatibility contract.",
    "scope_note": "No migration or reviewer routing changes.",
    "resume_instruction": "Continue this activity and verify the adapter contract.",
}
OWNER = dict(platform="discord", chat_id="project-thread", chat_type="group",
             user_id="project-owner", scope_id="project-guild", profile="coordinator",
             session_id="coordinator-session")


@pytest.fixture
def direction_worker(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kn
    from tools import kanban_tools  # register real handlers

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD",
                "HERMES_KANBAN_TASK", "HERMES_PROFILE_NAME", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(key, raising=False)
    workspace = tmp_path / "same-worktree"
    workspace.mkdir()
    (workspace / "activity.txt").write_text("active activity remains here")
    kb.init_db()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="Preserve existing Worker activity", assignee="worker",
                             created_by="coordinator", session_id=OWNER["session_id"],
                             workspace_kind="dir", workspace_path=str(workspace))
        kn.add_notify_sub(conn, task_id=tid, platform=OWNER["platform"],
                          chat_id=OWNER["chat_id"], chat_type=OWNER["chat_type"],
                          user_id=OWNER["user_id"], notifier_profile=OWNER["profile"],
                          delivery_mode="notify+wake",
                          delivery_metadata={"scope_id": OWNER["scope_id"], "chat_type": "group"})
        task = kb.claim_task(conn, tid, claimer="same-worker-claim")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", task.claim_lock)
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setenv("HERMES_SESSION_ID", "same-worker-session")
    return task, workspace


def _dispatch(name, args):
    from tools.registry import registry
    return json.loads(registry.dispatch(name, args))


def _answer_as_owner(monkeypatch, args, *, profile="coordinator", chat_id="project-thread",
                     session_id="coordinator-session"):
    from gateway.session_context import clear_session_vars, set_session_vars
    with monkeypatch.context() as context:
        context.delenv("HERMES_KANBAN_TASK")
        context.setenv("HERMES_PROFILE", profile)
        tokens = set_session_vars(**{**OWNER, "profile": profile, "chat_id": chat_id,
                                     "session_id": session_id})
        try:
            return _dispatch("kanban_answer_direction", args)
        finally:
            clear_session_vars(tokens)


def test_registered_direction_tools_resume_same_worker_twice_and_reject_replays(
        direction_worker, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_direction as kd
    from tools import kanban_tools as kt

    original, workspace = direction_worker
    tid, run_id = original.id, original.current_run_id
    resumed_context = []
    question_ids = []
    consume = kd.consume_direction
    transient_failures = []

    def temporarily_busy(conn, task_id, question_id, **kwargs):
        assert not conn.in_transaction
        if not transient_failures:
            transient_failures.append(question_id)
            raise sqlite3.OperationalError("database is locked")
        return consume(conn, task_id, question_id, **kwargs)

    monkeypatch.setattr(kd, "consume_direction", temporarily_busy)

    def developer_turn(_seconds):
        with kbc.connect_closing() as conn:
            question = kd.latest_direction(conn, tid)
            paused = kb.get_task(conn, tid)
            assert paused.status == "needs_direction"
            assert (paused.current_run_id, paused.assignee, paused.workspace_path) == (
                run_id, original.assignee, original.workspace_path)
            assert paused.consecutive_failures == original.consecutive_failures
            assert paused.block_recurrences == original.block_recurrences
            assert len(kb.list_runs(conn, tid)) == 1
            assert len(kb.list_tasks(conn)) == 1
            assert question["worker_session_id"] == "same-worker-session"
        question_ids.append(question["id"])
        args = {"task_id": tid, "question_id": question["id"], "response": RESPONSE}
        assert "error" in _dispatch("kanban_answer_direction", args), "Workers cannot answer themselves"
        assert "error" in _answer_as_owner(monkeypatch, args, profile="other-developer")
        assert "error" in _answer_as_owner(monkeypatch, args, chat_id="another-thread")
        assert "error" in _answer_as_owner(monkeypatch, args, session_id="new-session-same-channel")
        first = _answer_as_owner(monkeypatch, args)
        assert first["ok"], first
        replay = _answer_as_owner(monkeypatch, args)
        assert replay["ok"], replay
        changed = {**args, "response": {**RESPONSE, "decision": "Change the public API instead."}}
        assert "error" in _answer_as_owner(monkeypatch, changed)

    monkeypatch.setattr(kt.time, "sleep", developer_turn)
    for question_text in (QUESTION["question"], "Which compatibility version should the adapter retain?"):
        result = _dispatch("kanban_needs_direction", {
            "developer_question": {**QUESTION, "question": question_text}})
        assert result["ok"] and result["status"] == "running", result
        assert (result["task_id"], result["run_id"]) == (tid, run_id)
        # The ordinary tool result is the bounded context added to this conversation.
        resumed_context.append(result["response"])
        with kbc.connect_closing() as conn:
            durable = kd.latest_direction(conn, tid)
            assert durable["response"] == RESPONSE
            assert kd.consume_direction(conn, tid, durable["id"], expected_run_id=run_id,
                                        claim_lock=original.claim_lock) is None
            assert kb.get_task(conn, tid).status == "running"
    assert len(set(question_ids)) == 2
    assert transient_failures == question_ids[:1]
    assert resumed_context == [RESPONSE, RESPONSE]
    assert (workspace / "activity.txt").read_text() == "active activity remains here"
    shown = _dispatch("kanban_show", {})
    assert shown["developer_question"]["id"] == question_ids[-1]
    assert shown["developer_question"]["response"] == RESPONSE


def test_direction_policy_and_ownership_fail_closed_and_unanswered_stays_paused(
        direction_worker, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_direction as kd
    from tools import interrupt, kanban_tools as kt
    from tools.registry import registry

    original, _ = direction_worker
    candidates = {"kanban_needs_direction", "kanban_answer_direction"}
    visible = {d["function"]["name"] for d in registry.get_definitions(candidates, quiet=True)}
    assert visible == {"kanban_needs_direction"}
    for bad in (dict(QUESTION, decision_needed="naming"),
                dict(QUESTION, evidence={**QUESTION["evidence"], "precedent_status": "resolved"})):
        assert "error" in _dispatch("kanban_needs_direction", {"developer_question": bad})
    for missing in ("HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK", "HERMES_SESSION_ID"):
        with monkeypatch.context() as context:
            context.delenv(missing)
            assert "error" in _dispatch("kanban_needs_direction", {"developer_question": QUESTION})
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, original.id).status == "running"
        assert kd.latest_direction(conn, original.id) is None

    unanswered_polls = []

    def remain_unanswered(_seconds):
        with kbc.connect_closing() as conn:
            task = kb.get_task(conn, original.id)
            question = kd.latest_direction(conn, original.id)
            assert task.status == "needs_direction" and question["response"] is None
            assert task.current_run_id == original.current_run_id
            assert task.consecutive_failures == original.consecutive_failures
        unanswered_polls.append(question["id"])
        if len(unanswered_polls) == 3:
            interrupt.set_interrupt(True)

    monkeypatch.setattr(kt.time, "sleep", remain_unanswered)
    try:
        result = _dispatch("kanban_needs_direction", {"developer_question": deepcopy(QUESTION)})
    finally:
        interrupt.set_interrupt(False)
    assert result["ok"] and result["paused"] and result["status"] == "needs_direction"
    assert len(unanswered_polls) == 3 and len(set(unanswered_polls)) == 1
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, original.id).status == "needs_direction"
        assert len(kb.list_runs(conn, original.id)) == 1
