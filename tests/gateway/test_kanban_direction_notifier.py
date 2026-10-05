"""Developer questions reuse durable Kanban notification routing and cursors."""

import asyncio
from pathlib import Path

from gateway.config import GatewayConfig, Platform
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
from hermes_constants import get_hermes_home


OWNER = "strategic-industrial-intelligence"
CHAT = "1549847541200457771"
USER = "1520991676020035585"
SCOPE = "1520992154644910250"


class RecordingAdapter:
    supports_async_delivery = True

    def __init__(self):
        self.sent = []
        self.handled = []
        self.homes = []

    async def send(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))

    async def handle_message(self, event):
        self.handled.append(event)
        self.homes.append(get_hermes_home())
        event._gateway_accepted = True


def _runner(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    for name in (OWNER, "observer"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: RecordingAdapter()}
    runner._profile_adapters = {OWNER: {}, "observer": {Platform.DISCORD: RecordingAdapter()}}
    runner._primary_profile_name = "default"
    runner._kanban_notifier_profile = "default"
    runner._kanban_dispatcher_lock_handle = object()
    runner.config = GatewayConfig(multiplex_profiles=True, profile_routes=parse_profile_routes([
        dict(platform="discord", guild_id=SCOPE, chat_id=CHAT, profile=OWNER),
    ]))
    return runner, home


def _collect(runner):
    return _notifier_collect(runner, kb, notifier_profile="default", gc_due=False, gc_retention_days=30)


async def _deliver(runner, rows):
    for row in rows:
        await _KanbanNotification(runner, row, platform_cls=Platform, sub_fail_counts={}).deliver()


def test_developer_questions_wake_only_the_owner_once_per_question(tmp_path, monkeypatch):
    runner, home = _runner(tmp_path, monkeypatch)
    owner_adapter = runner.adapters[Platform.DISCORD]
    observer_adapter = runner._profile_adapters["observer"][Platform.DISCORD]
    question = {
        "question": "Which of the conflicting migration requirements governs this activity?",
        "why_it_matters": "The public migration contract changes depending on the decision.",
        "evidence": {"references": ["docs/canon.md#v1", "docs/migrations.md#v2"],
                     "precedent_status": "conflicting"},
        "options": ["Preserve the v1 contract", "Approve the v2 migration"],
        "decision_needed": "migration_or_public_interface",
        "worker_recommendation": "Preserve v1 unless the Developer approves the migration.",
        "work_that_can_continue": "None on the affected interface.",
    }
    owner_session = {"platform": "discord", "chat_id": CHAT, "thread_id": "",
                     "notifier_profile": OWNER, "user_id": USER, "chat_type": "group",
                     "delivery_metadata": {"scope_id": SCOPE}}
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title="existing activity", assignee="worker",
            session_id=f"agent:{OWNER}:discord:group:{CHAT}:{USER}")
        # The owner has one active wake route and one explicitly passive route;
        # another session and profile also watch this card. Only the owning wake route
        # may receive the decision request.
        for profile, thread, mode in ((OWNER, "", "notify+wake"),
                                      (OWNER, "passive", "notify"),
                                      (OWNER, "other-session", "notify+wake"),
                                      ("observer", "observer", "notify+wake")):
            kbn.add_notify_sub(
                conn, task_id=task_id, platform="discord", chat_id=CHAT,
                thread_id=thread, chat_type="group", user_id=USER,
                notifier_profile=profile, delivery_mode=mode,
                delivery_metadata={"scope_id": SCOPE})
        conn.execute("UPDATE tasks SET status = 'needs_direction' WHERE id = ?", (task_id,))
        kb._append_event(conn, task_id, kind="developer_question", payload={
            "question_id": 41, "owner_profile": OWNER, "question": question,
            "owner_session": owner_session,
            "worker_session_id": "same-worker-session"})

    rows = _collect(runner)
    assert len(rows) == 1
    assert rows[0]["sub"]["notifier_profile"] == OWNER
    assert rows[0]["sub"]["thread_id"] == ""
    asyncio.run(_deliver(runner, rows))

    assert len(owner_adapter.sent) == len(owner_adapter.handled) == 1
    assert observer_adapter.sent == observer_adapter.handled == []
    wake = owner_adapter.handled[0]
    assert (wake.source.profile, wake.source.chat_id, wake.source.user_id, wake.source.scope_id) == (
        OWNER, CHAT, USER, SCOPE)
    assert owner_adapter.homes == [home / "profiles" / OWNER]
    assert "NEEDS_DIRECTION" in wake.text
    assert question["question"] in wake.text
    assert "docs/canon.md#v1" in wake.text
    assert f"kanban_answer_direction(task_id='{task_id}', question_id=41" in wake.text
    assert "resume_instruction" in wake.text
    assert "Without an answer the Worker remains paused" in wake.text
    assert not _collect(runner)

    # The persisted cursor deduplicates later notifier instances without
    # consuming the subscription needed by a legitimate second decision.
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "needs_direction"
        assert len(kbn.list_notify_subs(conn, task_id)) == 4
        kb._append_event(conn, task_id, kind="developer_question", payload={
            "question_id": 42, "owner_profile": OWNER,
            "owner_session": owner_session,
            "question": {**question, "question": "Who owns the public-interface approval?"}})
    asyncio.run(_deliver(runner, _collect(runner)))
    assert len(owner_adapter.sent) == len(owner_adapter.handled) == 2
    assert "question_id=42" in owner_adapter.handled[-1].text
    assert observer_adapter.sent == observer_adapter.handled == []
    assert not _collect(runner)
