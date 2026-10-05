"""A direction pause stays visible and cannot be resumed by a generic drag."""

import importlib.util
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc


def test_direction_pause_has_a_column_and_requires_its_response_path(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    plugin = Path(__file__).resolve().parents[2] / "plugins/kanban/dashboard/plugin_api.py"
    spec = importlib.util.spec_from_file_location("kanban_direction_status_test", plugin)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/kanban")
    client = TestClient(app)
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="same activity", assignee="worker")
        conn.execute("UPDATE tasks SET status = 'needs_direction' WHERE id = ?", (task_id,))
        ready_id = kb.create_task(conn, title="ordinary task", assignee="other-worker")

    board = client.get("/api/plugins/kanban/board").json()
    columns = {column["name"]: column["tasks"] for column in board["columns"]}
    assert [task["id"] for task in columns["needs_direction"]] == [task_id]
    assert not any(task["id"] == task_id for task in columns["todo"] + columns["blocked"])

    for status in ("ready", "running", "todo", "triage", "done", "blocked", "review", "archived"):
        response = client.patch(f"/api/plugins/kanban/tasks/{task_id}", json={"status": status})
        assert response.status_code == 400, response.text
        assert "kanban_answer_direction" in response.json()["detail"]
    for patch in ({"status": "ready"}, {"archive": True}):
        response = client.post("/api/plugins/kanban/tasks/bulk", json={"ids": [task_id], **patch})
        result = response.json()["results"][0]
        assert result["ok"] is False
        assert "kanban_answer_direction" in result["error"]
    response = client.patch(f"/api/plugins/kanban/tasks/{ready_id}", json={"status": "needs_direction"})
    assert response.status_code == 400
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "needs_direction"
        assert kb.get_task(conn, task_id).assignee == "worker"
        assert kb.get_task(conn, ready_id).status == "ready"
