"""A persisted Discord guild anchor and its reconstructed scope identify one owner."""

from pathlib import Path

import pytest

from gateway.kanban_watchers_notifier import _event_for_subscription
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
from hermes_cli.kanban_direction import answer_direction, request_direction
from tools.kanban_tools import _resolve_notify_target


@pytest.mark.parametrize(("platform", "owner_session_id", "error"), [
    ("discord", None, "coordinator session identity"),
    ("discord", "", "coordinator session identity"),
    ("discord", "   ", "coordinator session identity"),
    ("tui", "owner-session", "TUI subscriptions cannot wake"),
])
def test_unbound_or_unsupported_owner_cannot_pause_worker(
        tmp_path, monkeypatch, platform, owner_session_id, error):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Existing activity", assignee="worker",
                                 session_id=owner_session_id)
        kbn.add_notify_sub(conn, task_id=task_id, platform=platform, chat_id="owner-chat",
                           notifier_profile="developer", delivery_mode="wake")
        task = kb.claim_task(conn, task_id)
        task_before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
        run_before = dict(conn.execute("SELECT * FROM task_runs WHERE id = ?",
                                       (task.current_run_id,)).fetchone())
        events_before = kb.list_events(conn, task_id)
        with pytest.raises(ValueError, match=error):
            request_direction(conn, task_id, expected_run_id=task.current_run_id,
                              worker_session_id="worker-session", question={
                                  "question": "Which conflicting canon provision governs the migration?",
                                  "why_it_matters": "The public interface depends on the decision.",
                                  "evidence": {"references": ["canon/v1", "canon/v2"],
                                               "precedent_status": "conflicting"},
                                  "options": ["Keep v1", "Approve v2"],
                                  "decision_needed": "canon_interpretation"})
        assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == task_before
        assert dict(conn.execute("SELECT * FROM task_runs WHERE id = ?",
                                 (task.current_run_id,)).fetchone()) == run_before
        assert kb.list_events(conn, task_id) == events_before
        assert conn.execute("SELECT COUNT(*) FROM developer_questions").fetchone()[0] == 0


def test_restored_owner_scope_alias_accepts_its_response_only(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    question = {
        "question": "Which conflicting canon provision governs the migration?",
        "why_it_matters": "The public interface depends on the authoritative provision.",
        "evidence": {"references": ["canon/v1", "canon/v2"], "precedent_status": "conflicting"},
        "options": ["Keep v1", "Approve v2"], "decision_needed": "canon_interpretation",
    }
    response = {"decision": "Keep v1", "rationale": "v1 remains authoritative.",
                "resume_instruction": "Continue this activity against v1."}
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="same card", assignee="worker", session_id="owner-session")
        kbn.add_notify_sub(conn, task_id=task_id, platform="discord", chat_id="post",
                           thread_id="post", chat_type="thread", user_id="owner-user",
                           notifier_profile="developer", delivery_mode="notify+wake",
                           delivery_metadata={"guild_id": "guild", "parent_chat_id": "channel"})
        task = kb.claim_task(conn, task_id)
        record = request_direction(conn, task_id, expected_run_id=task.current_run_id,
                                   worker_session_id="worker-session", question=question)
        event = next(event for event in kb.list_events(conn, task_id) if event.kind == "developer_question")
        sub = kbn.list_notify_subs(conn, task_id)[0]
        assert _event_for_subscription(event, sub)

        # Source reconstruction on a wake exports scope_id, even though the
        # explicit CLI subscription used --guild-id for the same Discord guild.
        tokens = set_session_vars(platform="discord", chat_id="post", thread_id="post",
                                  chat_type="thread", user_id="owner-user", profile="developer",
                                  scope_id="guild", parent_chat_id="channel", session_id="owner-session")
        try:
            responder = _resolve_notify_target()
        finally:
            clear_session_vars(tokens)
        responder["session_id"] = "owner-session"
        assert _event_for_subscription(event, responder)
        for mutation in ({"user_id": "another-user"}, {"chat_type": "group"},
                         {"delivery_metadata": {"scope_id": "another-guild", "parent_chat_id": "channel"}},
                         {"delivery_metadata": {"scope_id": "guild", "parent_chat_id": "another-channel"}}):
            wrong = {**responder, **mutation}
            assert not _event_for_subscription(event, wrong)
            with pytest.raises(ValueError, match="owning coordinator session"):
                answer_direction(conn, task_id, record["id"], responder_profile="developer",
                                 responder_session=wrong, response=response)
        # Neither a missing responder identity nor an older unbound question
        # grants authority to a replacement session on the same route.
        for missing in (None, "", "   "):
            with pytest.raises(ValueError, match="owning coordinator session"):
                answer_direction(conn, task_id, record["id"], responder_profile="developer",
                                 responder_session={**responder, "session_id": missing}, response=response)
            with kb.write_txn(conn):
                conn.execute("UPDATE developer_questions SET owner_session_id = ? WHERE id = ?",
                             (missing, record["id"]))
            with pytest.raises(ValueError, match="owning coordinator session"):
                answer_direction(conn, task_id, record["id"], responder_profile="developer",
                                 responder_session={**responder, "session_id": "replacement-session"},
                                 response=response)
            with kb.write_txn(conn):
                conn.execute("UPDATE developer_questions SET owner_session_id = ? WHERE id = ?",
                             ("owner-session", record["id"]))
        answered = answer_direction(conn, task_id, record["id"], responder_profile="developer",
                                    responder_session=responder, response=response)
        assert answered["response"] == response
        assert answered["resumed_at"] is None
        assert kb.get_task(conn, task_id).status == "needs_direction"


@pytest.mark.parametrize("platform", ["discord", "api_server"])
def test_owner_compression_continues_authority_but_other_sessions_do_not(tmp_path, monkeypatch, platform):
    from gateway.run import _profile_runtime_scope
    from hermes_state import SessionDB

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    owner_home, other_home = home / "profiles" / "developer", home / "profiles" / "other"
    owner_home.mkdir(parents=True)
    other_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    db = SessionDB(owner_home / "state.db")
    try:
        db.create_session("owner-root", source=platform)
        db.append_message("owner-root", role="user", content="Coordinate the existing activity.")
        db.end_session("owner-root", "compression")
        db.create_session("owner-tip", source=platform, parent_session_id="owner-root")
        db.append_message("owner-tip", role="user", content="Continue coordinating the existing activity.")
        db.create_session("sibling", source=platform, parent_session_id="owner-root",
                          model_config={"_branched_from": "owner-root"})
        db.append_message("sibling", role="user", content="A different branch of the conversation.")
        db.create_session("unrelated", source=platform)
        db.append_message("unrelated", role="user", content="Another conversation.")
    finally:
        db.close()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="same card", assignee="worker",
                                 session_id="worker-provenance" if platform == "api_server" else "owner-root")
        kbn.add_notify_sub(conn, task_id=task_id, platform=platform,
                           chat_id="owner-root" if platform == "api_server" else "channel",
                           chat_type="dm", user_id=None if platform == "api_server" else "owner",
                           notifier_profile="developer", delivery_mode="wake")
        task = kb.claim_task(conn, task_id)
        record = request_direction(conn, task_id, expected_run_id=task.current_run_id,
                                   worker_session_id="worker-session", question={
                                       "question": "Which authority owns this public-interface decision?",
                                       "why_it_matters": "It determines who may approve the migration.",
                                       "evidence": {"references": ["canon/authority"], "precedent_status": "unresolved"},
                                       "options": ["Product Owner", "Developer"], "decision_needed": "authority_boundary"})
        assert record["owner_session_id"] == "owner-root"
        response = {"decision": "Developer", "rationale": "The current scope grants authority.",
                    "resume_instruction": "Resume this activity within its existing scope."}
        if platform == "api_server":
            # The actual API wake resolves the tip before binding chat_id and
            # session_id; it supplies no chat_type, unlike persisted subs (dm).
            tokens = set_session_vars(platform="api_server", chat_id="owner-tip",
                                      session_id="owner-tip", profile="developer")
            try:
                responder = _resolve_notify_target()
            finally:
                clear_session_vars(tokens)
            responder["session_id"] = "owner-tip"
        else:
            responder = {**record["owner_session"], "session_id": "owner-tip"}
        with _profile_runtime_scope(owner_home):
            for wrong_id in ("sibling", "unrelated", "unknown"):
                with pytest.raises(ValueError, match="owning coordinator session"):
                    answer_direction(conn, task_id, record["id"], responder_profile="developer", response=response,
                                     responder_session={**responder, "session_id": wrong_id})
        # A lineage that exists only in the Developer profile cannot authorize
        # an answer from a different runtime store (A -> B -> A).
        with _profile_runtime_scope(other_home), pytest.raises(ValueError, match="owning coordinator session"):
            answer_direction(conn, task_id, record["id"], responder_profile="developer", response=response,
                             responder_session=responder)
        with _profile_runtime_scope(owner_home):
            answered = answer_direction(conn, task_id, record["id"], responder_profile="developer", response=response,
                                        responder_session=responder)
        assert answered["response"] == response
        assert kb.get_task(conn, task_id).status == "needs_direction"
