import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import server  # noqa: E402


def test_validate_dimensions_rejects_unsafe_values():
    with pytest.raises(ValueError):
        server._validate_dimensions(255, 1024)
    with pytest.raises(ValueError):
        server._validate_dimensions(1025, 1024)
    with pytest.raises(ValueError):
        server._validate_dimensions(1024, 1025)


def test_validate_dimensions_accepts_bounded_multiple():
    server._validate_dimensions(1024, 1024)


def test_validate_prompt_is_bounded():
    with pytest.raises(ValueError):
        server._validate_prompt("")
    with pytest.raises(ValueError):
        server._validate_prompt("x" * (server.MAX_PROMPT_LENGTH + 1))


def test_modal_auth_is_fail_closed(monkeypatch):
    monkeypatch.delenv("MODAL_ENDPOINT_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        server._modal_headers()


def test_modal_auth_header_is_bearer(monkeypatch):
    monkeypatch.setenv("MODAL_ENDPOINT_TOKEN", "test-token")
    assert server._modal_headers() == {"Authorization": "Bearer test-token"}


def test_health_reports_degraded_when_required_tokens_are_missing(monkeypatch):
    monkeypatch.delenv("MCP_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("MODAL_ENDPOINT_TOKEN", raising=False)
    response = asyncio.run(server.health_check(None))
    assert response.status_code == 503


def test_async_drive_tool_rejects_invalid_file_id():
    result = asyncio.run(server.start_drive_processing.fn(file_id=""))
    assert result.startswith("Error:")
    assert "file_id" in result


def test_safe_error_does_not_dump_large_body():
    response = httpx.Response(500, text="x" * 10_000)
    message = server._safe_error(response)
    assert len(message) < 600


def test_unsafe_code_tool_is_not_registered():
    assert not hasattr(server, "_run_gpu_code")
    assert not hasattr(server.mcp, "run_gpu_code")


def test_generate_image_returns_image_content(monkeypatch):
    class FakeResponse:
        status_code = 200
        content = b"fake-png"

    async def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(server, "_post", fake_post)
    monkeypatch.setenv("MODAL_ENDPOINT_TOKEN", "test-token")
    result = asyncio.run(server.generate_image.fn("a test", 1024, 1024))
    assert result.__class__.__name__ == "Image"
    assert result.data == b"fake-png"


def test_post_includes_request_id_header(monkeypatch):
    seen = {}

    class FakeResponse:
        status_code = 200

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, endpoint, json, headers):
            seen.update(headers)
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    monkeypatch.setenv("MODAL_ENDPOINT_TOKEN", "test-token")
    asyncio.run(server._post("https://example.test", {}, timeout=1))
    assert seen["Authorization"] == "Bearer test-token"
    assert len(seen["X-Request-ID"]) == 12
