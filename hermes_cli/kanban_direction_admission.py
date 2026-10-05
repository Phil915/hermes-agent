"""Fence process-loss restores before startup and admit them only when ready to run."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import time


_TOKEN_ENV = "HERMES_KANBAN_DIRECTION_TOKEN"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DirectionResume:
    task_id: str
    run_id: int
    claim_lock: str
    worker_session_id: str
    token: str


def _pending_launch(conn, launch: DirectionResume):
    return conn.execute(
        "SELECT q.* FROM developer_questions q "
        "JOIN tasks t ON t.id = q.task_id "
        "JOIN task_runs r ON r.id = q.run_id AND r.task_id = t.id "
        "WHERE t.id = ? AND t.status = 'needs_direction' AND t.current_run_id = ? "
        "AND t.claim_lock = ? AND r.ended_at IS NULL "
        "AND q.run_id = ? AND q.worker_session_id = ? AND q.resume_token = ? "
        "AND q.response IS NOT NULL AND q.resumed_at IS NULL AND q.resume_admitted_at IS NULL",
        (launch.task_id, launch.run_id, launch.claim_lock, launch.run_id,
         launch.worker_session_id, launch.token),
    ).fetchone()


def prepare_direction_resume_from_env(*, worker_session_id: str) -> DirectionResume | None | bool:
    """Validate a committed launch without consuming its answer during CLI startup.

    Taking the board writer lock waits for the launch transaction to commit.
    Pop the nonce now so startup subprocesses cannot inherit it. Credentials,
    session limits and history loading may still refuse; the run stays paused.
    """
    token = os.environ.pop(_TOKEN_ENV, None)
    if token is None:
        return None
    task_id = os.environ.get("HERMES_KANBAN_TASK")
    claim_lock = os.environ.get("HERMES_KANBAN_CLAIM_LOCK")
    try:
        run_id = int(os.environ.get("HERMES_KANBAN_RUN_ID", ""))
    except ValueError:
        return False
    if not token or not task_id or not claim_lock or not worker_session_id:
        return False
    launch = DirectionResume(task_id, run_id, claim_lock, worker_session_id, token)
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    try:
        with kbc.connect_closing() as conn, kb.write_txn(conn):
            return launch if _pending_launch(conn, launch) is not None else False
    except Exception:
        logger.warning("Kanban direction launch verification failed; Worker remains paused", exc_info=True)
        return False


def _admit_direction_resume(launch: DirectionResume, *, worker_session_id: str) -> bool:
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli.kanban_direction import _consume, _decode, _owner_session_matches

    # --resume follows the canonical compression tip while restoring history.
    # The same lineage is valid; a sibling or unrelated session cannot consume.
    if not _owner_session_matches(launch.worker_session_id, worker_session_id):
        return False
    try:
        with kbc.connect_closing() as conn:
            with kb.write_txn(conn):
                row = _pending_launch(conn, launch)
                if row is None:
                    return False
                _consume(conn, kb.get_task(conn, launch.task_id), _decode(row))
                conn.execute(
                    "UPDATE developer_questions SET resume_admitted_at = ? WHERE id = ?",
                    (int(time.time()), row["id"]),
                )
            kb.notify_task_updated(conn, launch.task_id, ("status",))
        return True
    except Exception:
        logger.warning("Kanban direction resume admission failed; Worker execution was not started", exc_info=True)
        return False


def admit_prepared_direction_resume(cli, *, conversation_history) -> bool:
    """Consume once, immediately before the ready CLI starts the restored turn."""
    launch = getattr(cli, "_direction_resume_admission", None)
    if launch is None:
        return True
    if not conversation_history or not getattr(cli, "agent", None):
        return False
    if not _admit_direction_resume(launch, worker_session_id=cli.session_id):
        return False
    cli._direction_resume_admission = None  # later goal turns are ordinary continuations
    return True
