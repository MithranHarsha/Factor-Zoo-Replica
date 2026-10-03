"""Offline test for the dashboard's demo-snapshot bootstrap: a fresh
deploy with no persistent volume (Streamlit Community
Cloud) has no point-in-time store at all, so app.py copies the committed
80-company demo snapshot into place on first run. Local development,
where DB_PATH already exists from running the CLI pull commands, must be
a no-op.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    """Imports dashboard.app with DB_PATH and DEMO_SNAPSHOT_PATH patched
    to temp-directory locations, so this test never touches the real
    project database or the real committed demo file."""
    from factorzoo.dashboard import app

    fake_db_path = tmp_path / "live" / "factorzoo.duckdb"
    fake_demo_path = tmp_path / "demo" / "factorzoo_demo.duckdb"
    fake_demo_path.parent.mkdir(parents=True)
    fake_demo_path.write_bytes(b"not a real duckdb file, just bytes to copy")

    monkeypatch.setattr(app, "DB_PATH", fake_db_path)
    monkeypatch.setattr(app, "DEMO_SNAPSHOT_PATH", fake_demo_path)
    return app, fake_db_path, fake_demo_path


class TestBootstrapFromDemoSnapshot:
    def test_copies_demo_snapshot_when_no_live_db_exists(self, app_module):
        app, fake_db_path, fake_demo_path = app_module
        assert not fake_db_path.exists()

        app._bootstrap_from_demo_snapshot_if_needed()

        assert fake_db_path.exists()
        assert fake_db_path.read_bytes() == fake_demo_path.read_bytes()

    def test_does_nothing_when_live_db_already_exists(self, app_module):
        app, fake_db_path, _fake_demo_path = app_module
        fake_db_path.parent.mkdir(parents=True, exist_ok=True)
        fake_db_path.write_bytes(b"real local data, must not be overwritten")

        app._bootstrap_from_demo_snapshot_if_needed()

        assert fake_db_path.read_bytes() == b"real local data, must not be overwritten"

    def test_does_nothing_when_no_demo_snapshot_is_present_either(self, app_module):
        app, fake_db_path, fake_demo_path = app_module
        fake_demo_path.unlink()

        app._bootstrap_from_demo_snapshot_if_needed()  # must not raise

        assert not fake_db_path.exists()
