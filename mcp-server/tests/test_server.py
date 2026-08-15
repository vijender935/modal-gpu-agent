import asyncio
import inspect
import sys
from pathlib import Path

import httpx
import pytest
from fastmcp.server.middleware.rate_limiting import (
    RateLimitError,
    RateLimitingMiddleware,
)

sys.path.insert(0, str(Path(__file__).parents[1]))
import server


async def _invoke_tool(tool, **arguments):
    if inspect.iscoroutinefunction(tool):
        return await tool(**arguments)
    function = getattr(tool, "fn", None) or getattr(tool, "__wrapped__", None)
    if function is not None:
        return await function(**arguments)
    result = await tool.run(arguments)
    structured = getattr(result, "structured_content", None)
    if structured:
        return structured
    content = getattr(result, "content", None)
    if content and hasattr(content[0], "text"):
        return content[0].text
    return result


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


def test_health_does_not_require_gateway_token(monkeypatch):
    monkeypatch.delenv("MCP_GATEWAY_TOKEN", raising=False)
    monkeypatch.setenv("MODAL_ENDPOINT_TOKEN", "test-token")
    response = asyncio.run(server.health_check(None))
    assert response.status_code == 200
    assert b"mcp_gateway_token" not in response.body


def test_rate_limit_middleware_rejects_burst_overflow():
    middleware = RateLimitingMiddleware(
        max_requests_per_second=0.01,
        burst_capacity=1,
        global_limit=True,
    )

    async def call_next(_context):
        return "ok"

    assert asyncio.run(middleware.on_request(None, call_next)) == "ok"
    with pytest.raises(RateLimitError, match="Global rate limit exceeded"):
        asyncio.run(middleware.on_request(None, call_next))


def test_modal_auth_is_fail_closed(monkeypatch):
    monkeypatch.delenv("MODAL_ENDPOINT_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        server._modal_headers()


def test_modal_auth_header_is_bearer(monkeypatch):
    monkeypatch.setenv("MODAL_ENDPOINT_TOKEN", "test-token")
    assert server._modal_headers() == {"Authorization": "Bearer test-token"}


def test_health_reports_degraded_when_modal_token_is_missing(monkeypatch):
    monkeypatch.delenv("MODAL_ENDPOINT_TOKEN", raising=False)
    response = asyncio.run(server.health_check(None))
    assert response.status_code == 503


def test_async_drive_tool_rejects_invalid_file_id():
    result = asyncio.run(_invoke_tool(server.start_drive_processing, file_id=""))
    assert result.startswith("Error:")
    assert "file_id" in result


def test_sync_drive_tool_rejects_invalid_inputs():
    invalid_file_id = asyncio.run(_invoke_tool(server.process_images_from_drive, file_id=""))
    invalid_force = asyncio.run(
        _invoke_tool(server.process_images_from_drive, force_reprocess="yes")
    )
    assert invalid_file_id.startswith("Error:")
    assert "file_id" in invalid_file_id
    assert invalid_force.startswith("Error:")
    assert "force_reprocess" in invalid_force


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
    result = asyncio.run(
        _invoke_tool(server.generate_image, prompt="a test", width=1024, height=1024)
    )
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
