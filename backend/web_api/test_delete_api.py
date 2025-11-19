import os
from pathlib import Path
from fastapi.testclient import TestClient
from backend.web_api.api_server import app, UPLOADS_DIR

client = TestClient(app)

def test_delete_nonexistent():
    p = UPLOADS_DIR / "not_exists.txt"
    r = client.delete(f"/api/v1/documents/{str(p)}")
    assert r.status_code == 404

def test_delete_permission_error(monkeypatch, tmp_path):
    p = UPLOADS_DIR / "perm.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    def fake_send2trash(_):
        from builtins import PermissionError as PE
        raise PE()
    monkeypatch.setattr("backend.web_api.api_server.send2trash", fake_send2trash)
    r = client.delete(f"/api/v1/documents/{str(p)}")
    assert r.status_code == 403

def test_delete_success(monkeypatch, tmp_path):
    p = UPLOADS_DIR / "ok.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    calls = {"count": 0}
    def fake_send2trash(_):
        calls["count"] += 1
    class VS:
        def delete_by_source(self, s):
            return 1
    from backend.web_api import api_server as srv
    monkeypatch.setattr("backend.web_api.api_server.send2trash", fake_send2trash)
    srv.rag.vector_store = VS()
    r = client.delete(f"/api/v1/documents/{str(p)}")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert calls["count"] == 1

def test_batch_delete_success(monkeypatch, tmp_path):
    p1 = UPLOADS_DIR / "a.txt"
    p2 = UPLOADS_DIR / "b.txt"
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_text("x")
    p2.write_text("y")
    calls = {"count": 0}
    def fake_send2trash(_):
        calls["count"] += 1
    class VS:
        def delete_by_source(self, s):
            return 1
    from backend.web_api import api_server as srv
    monkeypatch.setattr("backend.web_api.api_server.send2trash", fake_send2trash)
    srv.rag.vector_store = VS()
    r = client.post("/api/v1/documents/batch_delete", json={"sources": [str(p1), str(p2)]})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["count"] == 2
    assert calls["count"] >= 2