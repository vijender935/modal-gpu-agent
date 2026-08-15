import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
import app  # noqa: E402


def test_sandbox_runs_code_and_returns_files():
    result = app._execute_python_sandbox(
        "from pathlib import Path\nPath('result.txt').write_text('ok')\nprint('hello')"
    )
    assert result["success"] is True
    assert "hello" in result["stdout"]
    assert result["files"][0]["path"] == "result.txt"


def test_sandbox_scrubs_sensitive_environment(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_TOKEN_JSON", "must-not-leak")
    result = app._execute_python_sandbox("import os; print(os.getenv('GOOGLE_OAUTH_TOKEN_JSON'))")
    assert result["success"] is True
    assert "must-not-leak" not in result["stdout"]
    assert "None" in result["stdout"]


def test_sandbox_timeout_is_bounded():
    result = app._execute_python_sandbox("while True: pass", timeout_seconds=1)
    assert result["timed_out"] is True
    assert result["success"] is False


def test_sandbox_rejects_invalid_request():
    with pytest.raises(ValueError):
        app._execute_python_sandbox("print('x')", timeout_seconds=app.MAX_SANDBOX_TIMEOUT + 1)
    with pytest.raises(ValueError):
        app._execute_python_sandbox("x" * (app.MAX_SANDBOX_CODE_LENGTH + 1))
