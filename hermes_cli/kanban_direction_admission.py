"""Admit a restored Worker only after its launching board transaction commits."""

from __future__ import annotations

import logging
import os
import time


_TOKEN_ENV = "HERMES_KANBAN_DIRECTION_TOKEN"
logger = logging.getLogger(__name__)


def admit_direction_resume_from_env(*, worker_session_id: str) -> bool:
    """Consume this launch's nonce once, before any resumed model/tool execution.

    The launcher spawns under a board write transaction. Taking a writer lock
    here waits for that transaction's commit; a rollback or launcher death
    leaves no matching nonce, so the untracked child exits without doing work.
    """
    token = os.environ.pop(_TOKEN_ENV, None)
    if token is None:
        return True
    task_id = os.environ.get("HERMES_KANBAN_TASK")
    claim_lock = os.environ.get("HERMES_KANBAN_CLAIM_LOCK")
    try:
        run_id = int(os.environ.get("HERMES_KANBAN_RUN_ID", ""))
    except ValueError:
        return False
    if not token or not task_id or not claim_lock or not worker_session_id:
        return False

    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli.kanban_direction import _consume, _decode

    try:
        with kbc.connect_closing() as conn:
            with kb.write_txn(conn):
                row = conn.execute(
                    "SELECT q.* FROM developer_questions q "
                    "JOIN tasks t ON t.id = q.task_id "
                    "JOIN task_runs r ON r.id = q.run_id AND r.task_id = t.id "
                    "WHERE t.id = ? AND t.status = 'needs_direction' AND t.current_run_id = ? "
                    "AND t.claim_lock = ? AND r.ended_at IS NULL "
                    "AND q.run_id = ? AND q.worker_session_id = ? AND q.resume_token = ? "
                    "AND q.response IS NOT NULL AND q.resumed_at IS NULL AND q.resume_admitted_at IS NULL",
                    (task_id, run_id, claim_lock, run_id, worker_session_id, token),
                ).fetchone()
                if row is None:
                    return False
                _consume(conn, kb.get_task(conn, task_id), _decode(row))
                conn.execute(
                    "UPDATE developer_questions SET resume_admitted_at = ? WHERE id = ?",
                    (int(time.time()), row["id"]),
                )
            kb.notify_task_updated(conn, task_id, ("status",))
        return True
    except Exception:
        logger.warning("Kanban direction resume admission failed; Worker execution was not started", exc_info=True)
        return False
