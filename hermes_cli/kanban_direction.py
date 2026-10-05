"""Non-terminal, run-bound Worker questions. All transitions share the board DB.

No retry, handoff or workspace operation belongs here. Answers stay paused until
the owning Worker consumes them; a dispatcher may restore a lost process on the
same run using its persisted session and workspace.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS developer_questions (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, run_id INTEGER NOT NULL,
    worker_session_id TEXT NOT NULL, owner_profile TEXT NOT NULL,
    owner_session TEXT NOT NULL, owner_session_id TEXT,
    question TEXT NOT NULL, response TEXT,
    requested_at INTEGER NOT NULL, answered_at INTEGER, resumed_at INTEGER,
    resume_token TEXT, resume_admitted_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_direction_task ON developer_questions(task_id, requested_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_direction_open ON developer_questions(task_id)
    WHERE resumed_at IS NULL;
"""

CATEGORIES = (
    "canon_interpretation", "architecture", "scope_expansion",
    "migration_or_public_interface", "conflicting_requirements",
    "authority_boundary", "meaningful_rework",
)
_SESSION_FIELDS = (
    "platform", "chat_id", "thread_id", "user_id", "user_id_alt", "chat_type",
    "notifier_profile",
)
_SCOPE_FIELDS = ("scope_id", "guild_id", "slack_team_id", "team_id", "parent_chat_id")


def _text(value: Any, name: str, limit: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be nonblank text of at most {limit} characters")
    return value.strip()


def _object(value: Any, required: set[str], optional: set[str], limit: int) -> dict:
    if not isinstance(value, dict) or required - value.keys() or value.keys() - required - optional:
        raise ValueError(f"expected fields {sorted(required)}; optional {sorted(optional)}")
    if len(json.dumps(value, ensure_ascii=False)) > limit:
        raise ValueError(f"payload exceeds {limit} characters")
    return dict(value)


def validate_question(value: Any) -> dict:
    q = _object(value, {"question", "why_it_matters", "evidence", "options", "decision_needed"},
                {"worker_recommendation", "work_that_can_continue"}, 8000)
    if q["decision_needed"] not in CATEGORIES:
        raise ValueError("decision_needed must be an upstream decision category; routine implementation is Worker-owned")
    for name in ("question", "why_it_matters", "worker_recommendation", "work_that_can_continue"):
        if name in q:
            q[name] = _text(q[name], name)
    evidence = _object(q["evidence"], {"references", "precedent_status"}, set(), 3000)
    if evidence["precedent_status"] not in {"unresolved", "conflicting"}:
        raise ValueError("check canon/code precedent first; already-resolved decisions cannot pause")
    for container, key, cap in ((q, "options", 3), (evidence, "references", 5)):
        values = container[key]
        if not isinstance(values, list) or not 1 <= len(values) <= cap:
            raise ValueError(f"{key} must contain 1–{cap} bounded entries")
        container[key] = [_text(v, key, 600) for v in values]
    q["evidence"] = evidence
    return q


def validate_response(value: Any) -> dict:
    response = _object(value, {"decision", "rationale", "resume_instruction"}, {"scope_note"}, 4000)
    return {key: _text(val, key, 2000 if key == "resume_instruction" else 1000)
            for key, val in response.items()}


def session_identity(session: dict) -> dict:
    """Compare the existing subscription's routing identity, excluding delivery cursors."""
    identity = {key: str(session.get(key) or "") for key in _SESSION_FIELDS}
    if identity["platform"] == "api_server":
        # API routes key by raw session ID, not chat type. Subscriptions default
        # to dm while API request contexts deliberately omit this field.
        identity["chat_type"] = ""
    metadata = session.get("delivery_metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    # A CLI subscription persists guild_id; the notifier reconstructs it as
    # SessionSource.scope_id and the answering turn exposes only scope_id.
    # Use the notifier's precedence so aliases cannot split one conversation.
    scope = next((metadata.get(key) for key in _SCOPE_FIELDS[:-1] if metadata.get(key)), "")
    identity["delivery_metadata"] = {
        "scope_id": str(scope), "parent_chat_id": str(metadata.get("parent_chat_id") or ""),
    }
    return identity


def _decode(row) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    for name in ("question", "response", "owner_session"):
        if result.get(name):
            result[name] = json.loads(result[name])
    return result


def latest_direction(conn, task_id: str) -> dict | None:
    return _decode(conn.execute(
        "SELECT * FROM developer_questions WHERE task_id = ? ORDER BY rowid DESC LIMIT 1",
        (task_id,),
    ).fetchone())


def request_direction(conn, task_id: str, *, expected_run_id: int,
                      worker_session_id: str, question: dict) -> dict:
    from hermes_cli import kanban_db as kb, kanban_db_notify as notify
    question = validate_question(question)
    worker_session_id = _text(worker_session_id, "worker_session_id", 200)
    with kb.write_txn(conn):
        task = kb.get_task(conn, task_id)
        if not task or expected_run_id is None or task.current_run_id != expected_run_id:
            raise ValueError("question requires the current Worker run")
        existing = latest_direction(conn, task_id)
        if task.status == "needs_direction" and existing and existing["resumed_at"] is None:
            if existing["question"] == question and existing["worker_session_id"] == worker_session_id:
                return existing
            raise ValueError("the current question must be answered before another is asked")
        run = conn.execute("SELECT ended_at FROM task_runs WHERE id = ?", (expected_run_id,)).fetchone()
        if task.status != "running" or not run or run["ended_at"] is not None:
            raise ValueError("only an open running Worker may ask for direction")
        owners = [sub for sub in notify.list_notify_subs(conn, task_id)
                  if sub.get("delivery_mode") in {"wake", "notify+wake"}
                  and sub.get("notifier_profile") and sub.get("chat_id")]
        if len(owners) != 1:
            raise ValueError("direction requires exactly one existing wake-capable coordinator subscription")
        owner = session_identity(owners[0])
        # Stateless API subscriptions name the owning raw session directly;
        # child cards can retain a different creator/Worker session as provenance.
        owner_session_id = owner["chat_id"] if owner["platform"] == "api_server" else task.session_id
        now, question_id = int(time.time()), "dir_" + uuid.uuid4().hex[:16]
        conn.execute(
            "INSERT INTO developer_questions (id, task_id, run_id, worker_session_id, owner_profile, "
            "owner_session, owner_session_id, question, requested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (question_id, task_id, expected_run_id, worker_session_id, owner["notifier_profile"],
             json.dumps(owner), owner_session_id, json.dumps(question), now),
        )
        conn.execute("UPDATE tasks SET status = 'needs_direction' WHERE id = ?", (task_id,))
        conn.execute("UPDATE task_runs SET status = 'needs_direction' WHERE id = ?", (expected_run_id,))
        kb._append_event(conn, task_id, "developer_question", {
            "question_id": question_id, "question": question, "owner_profile": owner["notifier_profile"],
            "owner_session": owner, "owner_session_id": owner_session_id,
            "worker_session_id": worker_session_id,
        }, run_id=expected_run_id)
        result = latest_direction(conn, task_id)
    kb.notify_task_updated(conn, task_id, ("status",))
    return result


def _owner_session_matches(owner_id: str | None, responder_id: str | None) -> bool:
    """Compression continues the same coordinator; a branch or new chat does not."""
    if not owner_id or owner_id == responder_id:
        return True
    if not responder_id:
        return False
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    db = None
    try:
        # Called in the authenticated responder's runtime scope, never the
        # gateway's launch profile. Exact IDs do not require a store lookup.
        db = SessionDB(get_hermes_home() / "state.db", read_only=True)
        return bool(db.get_session(owner_id) and db.get_session(responder_id)
                    and db.resolve_resume_session_id(owner_id) == responder_id)
    except (OSError, sqlite3.Error):
        return False
    finally:
        if db is not None:
            db.close()


def answer_direction(conn, task_id: str, question_id: str, *, responder_profile: str,
                     response: dict, responder_session: dict | None = None) -> dict:
    from hermes_cli import kanban_db as kb
    response = validate_response(response)
    with kb.write_txn(conn):
        row = _decode(conn.execute("SELECT * FROM developer_questions WHERE id = ? AND task_id = ?",
                                   (question_id, task_id)).fetchone())
        if not row:
            raise ValueError("unknown direction question")
        responder = session_identity(responder_session or {})
        same_session = bool(responder_session and responder_profile == row["owner_profile"]
                            and _owner_session_matches(row["owner_session_id"], responder_session.get("session_id")))
        if (same_session and row["owner_session"]["platform"] == "api_server"
                and responder["chat_id"] == responder_session.get("session_id")):
            # API wakes bind the resolved continuation as both chat_id and
            # session_id. Only the verified owner lineage may replace this key.
            responder["chat_id"] = row["owner_session"]["chat_id"]
        if not same_session or responder != row["owner_session"]:
            raise ValueError("only the owning coordinator session may answer this question")
        if row["response"] is not None:
            if row["response"] != response:
                raise ValueError("question already has a different durable answer")
            return row
        task = kb.get_task(conn, task_id)
        if not task or task.status != "needs_direction" or task.current_run_id != row["run_id"]:
            raise ValueError("question no longer belongs to the paused run")
        conn.execute("UPDATE developer_questions SET response = ?, answered_at = ? WHERE id = ?",
                     (json.dumps(response), int(time.time()), question_id))
        kb._append_event(conn, task_id, "developer_response",
                         {"question_id": question_id, "response": response,
                          "owner_profile": responder_profile}, run_id=row["run_id"])
        return _decode(conn.execute("SELECT * FROM developer_questions WHERE id = ?", (question_id,)).fetchone())


def _consume(conn, task, row: dict) -> dict:
    """Called under the board transaction, by the waiter or process-loss restore."""
    from hermes_cli import kanban_db as kb
    now = int(time.time())
    elapsed = max(0, now - row["requested_at"])
    expires = now + kb._resolve_claim_ttl_seconds()
    conn.execute("UPDATE developer_questions SET resumed_at = ? WHERE id = ? AND resumed_at IS NULL",
                 (now, row["id"]))
    conn.execute("UPDATE tasks SET status = 'running', claim_expires = ?, last_heartbeat_at = ? WHERE id = ?",
                 (expires, now, task.id))
    conn.execute("UPDATE task_runs SET status = 'running', claim_expires = ?, last_heartbeat_at = ?, "
                 "last_activity_at = ?, direction_paused_seconds = direction_paused_seconds + ? WHERE id = ?",
                 (expires, now, now, elapsed, task.current_run_id))
    kb._append_event(conn, task.id, "direction_resumed", {"question_id": row["id"], "paused_seconds": elapsed},
                     run_id=task.current_run_id)
    return {**row, "resumed_at": now}


def consume_direction(conn, task_id: str, question_id: str, *, expected_run_id: int,
                      claim_lock: str) -> dict | None:
    from hermes_cli import kanban_db as kb
    with kb.write_txn(conn):
        task = kb.get_task(conn, task_id)
        row = latest_direction(conn, task_id)
        if (not task or task.status != "needs_direction" or task.current_run_id != expected_run_id
                or not claim_lock or task.claim_lock != claim_lock or not row
                or row["id"] != question_id or row["run_id"] != expected_run_id
                or row["response"] is None or row["resumed_at"] is not None):
            return None
        result = _consume(conn, task, row)
    kb.notify_task_updated(conn, task_id, ("status",))
    return result


def direction_context(conn, task_id: str) -> str:
    row = latest_direction(conn, task_id)
    if not row:
        return ""
    return "\n\n## Developer direction (same card/run)\n" + json.dumps({
        "question_id": row["id"], "question": row["question"], "response": row["response"],
    }, ensure_ascii=False)


def paused_worker_counts(conn) -> dict[str, int]:
    """A live waiting process still occupies its existing host/profile slot."""
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    counts: dict[str, int] = {}
    for row in conn.execute("SELECT assignee, claim_lock, worker_pid, worker_started_at FROM tasks "
                            "WHERE status = 'needs_direction'"):
        remote = not str(row["claim_lock"] or "").startswith(kb._host_prefix())
        if remote or dispatch._worker_alive(row["worker_pid"], row["worker_started_at"]):
            counts[row["assignee"]] = counts.get(row["assignee"], 0) + 1
    return counts


def validate_resume_session(task, profile_home: str | None) -> None:
    """Fail paused if the promised Worker history is gone; never start fresh."""
    if not task.direction_resume_session_id:
        return
    from pathlib import Path
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    home = Path(profile_home) if profile_home else get_hermes_home()
    db = SessionDB(db_path=home / "state.db", read_only=True)
    try:
        if not db.get_session(task.direction_resume_session_id):
            raise ValueError("persisted Worker session is unavailable; direction remains paused")
        tip = db.resolve_resume_session_id(task.direction_resume_session_id)
        if not db.get_messages(tip, limit=1):
            raise ValueError("persisted Worker history is unavailable; direction remains paused")
    finally:
        db.close()


def resume_answered_workers(conn, result, *, spawn_fn, board, dry_run, spawn_budget,
                           per_profile_cap, per_profile_running) -> int:
    """Restore only answered pauses whose original host-local process is gone.

    Called inside the existing dispatcher lock and capacity budget. Deliberately
    skips claim_task, routing resolution, workspace creation, and failure/retry
    accounting. The session launcher receives the original run and workspace.
    """
    from dataclasses import replace
    from pathlib import Path
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch

    rows = conn.execute(
        "SELECT q.* FROM developer_questions q JOIN tasks t ON t.id = q.task_id "
        "WHERE t.status = 'needs_direction' AND t.current_run_id = q.run_id "
        "AND q.response IS NOT NULL AND q.resumed_at IS NULL ORDER BY q.answered_at, q.rowid",
    ).fetchall()
    spawned = 0
    for raw in rows:
        if spawn_budget is not None and spawned >= spawn_budget:
            break
        row = _decode(raw)
        task = kb.get_task(conn, row["task_id"])
        allowed = dispatch._profile_exists_fn()
        if not task.assignee or (allowed is not None and not allowed(task.assignee)):
            continue
        if per_profile_cap is not None and per_profile_running.get(task.assignee, 0) >= per_profile_cap:
            continue
        if not task.workspace_path or not Path(task.workspace_path).is_dir():
            continue  # Never manufacture a replacement workspace.
        run = conn.execute("SELECT * FROM task_runs WHERE id = ?", (row["run_id"],)).fetchone()
        if (not run or run["ended_at"] is not None
                or not str(task.claim_lock or "").startswith(kb._host_prefix())
                or dispatch._worker_alive(run["worker_pid"], run["worker_started_at"])):
            continue
        if dry_run:
            result.spawned.append((task.id, task.assignee, task.workspace_path))
            spawned += 1
            continue
        route = kb._json_dict(run["metadata"]).get("dispatch_route") or {}
        resumed = replace(
            task, direction_resume_session_id=row["worker_session_id"],
            direction_resume_token=uuid.uuid4().hex,
            direction_resume_prompt=("Resume this same Worker activity after the coordinator's answer. "
                                     "Keep this card, run, branch and workspace.\n" + direction_context(conn, task.id)),
            model_override=(route.get("model") if route.get("model") not in {None, "profile-default"}
                            else task.model_override),
            provider_override=(route.get("provider") if route.get("provider") not in {None, "profile-default"}
                               else task.provider_override),
        )
        try:
            # Startup takes the same write lock and admits this launch token only
            # after PID/token commit. A rolled-back or orphan launch cannot
            # make model calls, even though WAL readers can see the old pause.
            with kb.write_txn(conn):
                current = kb.get_task(conn, task.id)
                fresh = latest_direction(conn, task.id)
                if (current.status != "needs_direction" or current.current_run_id != row["run_id"]
                        or fresh["id"] != row["id"] or fresh["resumed_at"] is not None):
                    continue
                conn.execute("UPDATE developer_questions SET resume_token = ? WHERE id = ?",
                             (resumed.direction_resume_token, row["id"]))
                pid = dispatch._call_spawn_fn(spawn_fn or dispatch._default_spawn,
                                              resumed, task.workspace_path, board)
                if not pid:
                    raise RuntimeError("session resume launcher returned no Worker PID")
                fingerprint = dispatch._process_fingerprint(int(pid)) or dispatch.UNVERIFIED_WORKER_FINGERPRINT
                conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = ? WHERE id = ?",
                             (int(pid), fingerprint, task.id))
                conn.execute("UPDATE task_runs SET worker_pid = ?, worker_started_at = ? WHERE id = ?",
                             (int(pid), fingerprint, row["run_id"]))
                kb._append_event(conn, task.id, "direction_resume_launched", {
                    "question_id": row["id"], "worker_session_id": row["worker_session_id"], "pid": int(pid),
                }, run_id=row["run_id"])
        except Exception as exc:
            # Leave the answer unconsumed and the same run paused. Host launch
            # failure is not a Worker attempt or permission to replace its work.
            kb._log.warning("kanban direction resume deferred for %s: %s", task.id, exc)
            continue
        kb.notify_task_updated(conn, task.id, ("status", "worker_pid"), board=board)
        result.spawned.append((task.id, task.assignee, task.workspace_path))
        per_profile_running[task.assignee] = per_profile_running.get(task.assignee, 0) + 1
        spawned += 1
    return spawned
