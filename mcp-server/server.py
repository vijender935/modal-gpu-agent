"""
Authenticated MCP gateway for the Modal GPU Agent.

The gateway is intentionally fail-closed in production:
- MCP_GATEWAY_TOKEN protects the public MCP endpoint.
- MODAL_ENDPOINT_TOKEN authenticates calls to Modal web functions.
- Arbitrary code execution is not exposed by this production service.
"""

from __future__ import annotations

import hmac
import os
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import McpError
from fastmcp.utilities.types import Image
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from starlette.responses import JSONResponse

IMAGE_ENDPOINT = os.getenv(
    "IMAGE_ENDPOINT",
    "https://vijender935--gpu-agent-generate-image-endpoint.modal.run",
)
GPU_ENDPOINT = os.getenv(
    "GPU_ENDPOINT",
    "https://vijender935--gpu-agent-check-gpu-endpoint.modal.run",
)
SANDBOX_ENDPOINT = os.getenv(
    "SANDBOX_ENDPOINT",
    "https://vijender935--gpu-agent-run-python-sandbox-endpoint.modal.run",
)
PROCESS_ENDPOINT = os.getenv(
    "PROCESS_ENDPOINT",
    "https://vijender935--gpu-agent-process-drive-endpoint.modal.run",
)

MAX_PROMPT_LENGTH = 2_000
MAX_SANDBOX_CODE_LENGTH = 32_000
MAX_SANDBOX_TIMEOUT = 120
MAX_RETRIES = 2


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _validate_dimensions(width: int, height: int, *, max_side: int = 4096) -> None:
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("width and height must be integers")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("width and height must be integers")
    if not (256 <= width <= max_side and 256 <= height <= max_side):
        raise ValueError(f"width and height must be between 256 and {max_side}")
    if width % 8 or height % 8:
        raise ValueError("width and height must be multiples of 8")
    if width * height > 16_000_000:
        raise ValueError("requested image is too large")


def _validate_prompt(prompt: str) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(f"prompt must be at most {MAX_PROMPT_LENGTH} characters")


def _gateway_token() -> str | None:
    return os.getenv("MCP_GATEWAY_TOKEN")


class BearerAuthMiddleware(Middleware):
    """Fail-closed bearer authentication for all MCP requests."""

    async def on_request(self, context: MiddlewareContext, call_next):
        if _env_flag("ALLOW_INSECURE_DEV"):
            return await call_next(context)

        expected = _gateway_token()
        if not expected:
            raise McpError(code=-32001, message="MCP gateway authentication is not configured")

        headers = get_http_headers()
        authorization = headers.get("authorization", "")
        scheme, _, provided = authorization.partition(" ")
        if scheme.lower() != "bearer" or not provided:
            raise McpError(code=-32001, message="Unauthorized")
        if not hmac.compare_digest(provided, expected):
            raise McpError(code=-32001, message="Unauthorized")
        return await call_next(context)


mcp = FastMCP("Modal GPU Agent", strict_input_validation=True)
mcp.add_middleware(BearerAuthMiddleware())
mcp.add_middleware(
    RateLimitingMiddleware(
        max_requests_per_second=float(os.getenv("MCP_REQUESTS_PER_SECOND", "0.2")),
        burst_capacity=int(os.getenv("MCP_BURST_CAPACITY", "5")),
    )
)
mcp.add_middleware(LoggingMiddleware(include_payloads=False))


@mcp.custom_route("/health", methods=["GET"])
async def health_check(_request):
    return JSONResponse({"status": "healthy", "service": "modal-gpu-agent-mcp"})


def _modal_headers() -> dict[str, str]:
    token = os.getenv("MODAL_ENDPOINT_TOKEN")
    if not token:
        raise RuntimeError("MODAL_ENDPOINT_TOKEN is not configured")
    return {"Authorization": f"Bearer {token}"}


async def _post(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                follow_redirects=False,
            ) as client:
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=_modal_headers(),
                )
            if response.status_code in {502, 503, 504} and attempt < MAX_RETRIES:
                continue
            return response
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
    raise RuntimeError(f"downstream request failed: {type(last_error).__name__}") from last_error


def _safe_error(response: httpx.Response) -> str:
    body = response.text[:500].replace("\n", " ")
    return f"downstream service returned HTTP {response.status_code}: {body}"


@mcp.tool
async def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
) -> Image | str:
    """Generate a PNG image using Flux on Modal GPU and return it as an MCP image artifact."""
    try:
        _validate_prompt(prompt)
        _validate_dimensions(width, height, max_side=2048)
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer")
        response = await _post(
            IMAGE_ENDPOINT,
            {"prompt": prompt, "width": width, "height": height, "seed": seed},
            timeout=300.0,
        )
    except (ValueError, RuntimeError) as exc:
        return f"Error: {exc}"

    if response.status_code != 200:
        return _safe_error(response)
    if not response.content or len(response.content) > 20 * 1024 * 1024:
        return "Error: generated image response was empty or too large"
    return Image(data=response.content, format="png")


@mcp.tool
async def check_gpu() -> dict[str, Any] | str:
    """Check whether the Modal T4 GPU is available without executing user-provided code."""
    try:
        response = await _post(GPU_ENDPOINT, {}, timeout=120.0)
    except RuntimeError as exc:
        return f"Error: {exc}"
    if response.status_code != 200:
        return _safe_error(response)
    try:
        return response.json()
    except ValueError:
        return "Error: GPU health endpoint returned invalid JSON"



@mcp.tool
async def run_python(
    code: str,
    gpu: bool = False,
    timeout_seconds: int = 60,
) -> dict[str, Any] | str:
    """Run preinstalled Python packages in an isolated, network-blocked sandbox."""
    if not isinstance(code, str) or not code.strip():
        return "Error: code must be a non-empty string"
    if len(code) > MAX_SANDBOX_CODE_LENGTH:
        return f"Error: code must be at most {MAX_SANDBOX_CODE_LENGTH} characters"
    if not isinstance(gpu, bool):
        return "Error: gpu must be a boolean"
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        return "Error: timeout_seconds must be an integer"
    if not 1 <= timeout_seconds <= MAX_SANDBOX_TIMEOUT:
        return f"Error: timeout_seconds must be between 1 and {MAX_SANDBOX_TIMEOUT}"
    try:
        response = await _post(
            SANDBOX_ENDPOINT,
            {"code": code, "gpu": gpu, "timeout_seconds": timeout_seconds},
            timeout=MAX_SANDBOX_TIMEOUT + 30.0,
        )
    except RuntimeError as exc:
        return f"Error: {exc}"
    if response.status_code != 200:
        return _safe_error(response)
    try:
        return response.json()
    except ValueError:
        return "Error: sandbox endpoint returned invalid JSON"


@mcp.tool
async def process_images_from_drive(
    target_w: int = 1080,
    target_h: int = 2340,
    file_id: str | None = None,
    force_reprocess: bool = False,
) -> dict[str, Any] | str:
    """Process Drive images idempotently, optionally selecting one file or forcing reprocessing."""
    try:
        _validate_dimensions(target_w, target_h)
        response = await _post(
            PROCESS_ENDPOINT,
            {
                "target_w": target_w,
                "target_h": target_h,
                "file_id": file_id,
                "force_reprocess": force_reprocess,
            },
            timeout=900.0,
        )
    except (ValueError, RuntimeError) as exc:
        return f"Error: {exc}"
    if response.status_code != 200:
        return _safe_error(response)
    try:
        result = response.json()
    except ValueError:
        return "Error: Drive endpoint returned invalid JSON"
    details = result.get("details", [])
    if isinstance(details, list):
        result["details"] = details[:100]
    return result


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Modal GPU Agent MCP Server on port {port}")
    mcp.run(transport="http", host="0.0.0.0", port=port)
