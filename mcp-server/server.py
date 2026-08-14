"""
MCP Server for Modal GPU Agent
Exposes image generation + GPU code execution to Claude / Grok / Cursor
WITH proper OAuth discovery endpoints for Claude integration
"""

import os
import httpx
from fastmcp import FastMCP
from contextlib import asynccontextmanager

# Modal Endpoints
IMAGE_ENDPOINT = os.getenv(
    "IMAGE_ENDPOINT",
    "https://vijender935--gpu-agent-generate-image-endpoint.modal.run"
)
CODE_ENDPOINT = os.getenv(
    "CODE_ENDPOINT",
    "https://vijender935--gpu-agent-run-code-endpoint.modal.run"
)

# Get base URL for OAuth discovery
BASE_URL = os.getenv("BASE_URL", "https://modal-mcp.onrender.com")

# Initialize FastMCP
mcp = FastMCP("Modal GPU Agent")


# ═══════════════════════════════════════════════════════════════════
# OAuth Discovery Endpoints (Claude UI ke liye)
# ═══════════════════════════════════════════════════════════════════

@mcp.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource():
    """OAuth protected resource discovery endpoint"""
    return {
        "urls": [f"{BASE_URL}/mcp"]
    }


@mcp.get("/.well-known/oauth-protected-resource/mcp")
async def oauth_protected_resource_mcp():
    """MCP-specific resource endpoint"""
    return {
        "mcp": f"{BASE_URL}/mcp"
    }


@mcp.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server():
    """OAuth authorization server discovery"""
    return {
        "authorization_endpoint": f"{BASE_URL}/authorize",
        "issuer": BASE_URL,
        "token_endpoint": f"{BASE_URL}/token"
    }


@mcp.post("/register")
async def register_oauth():
    """OAuth client registration endpoint"""
    return {
        "status": "ok",
        "message": "MCP Server ready for integration"
    }


# ═══════════════════════════════════════════════════════════════════
# Health Check (Render Free Tier ke liye - keep-alive)
# ═══════════════════════════════════════════════════════════════════

@mcp.get("/health")
async def health_check():
    """Health check endpoint to prevent service shutdown"""
    return {
        "status": "healthy",
        "server": "Modal GPU Agent",
        "mcp_ready": True
    }


# ═══════════════════════════════════════════════════════════════════
# MCP Tools
# ═══════════════════════════════════════════════════════════════════

@mcp.tool
async def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = None
) -> str:
    """
    Generate an image using Flux model on Modal GPU.
    
    Args:
        prompt: Text description of the image you want to generate
        width: Image width (default 1024)
        height: Image height (default 1024)
        seed: Optional random seed for reproducibility
    """
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
    }
    if seed is not None:
        payload["seed"] = seed

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(IMAGE_ENDPOINT, json=payload)

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    return f"✅ Image generated successfully for prompt: '{prompt}'\n\n(Image binary received. Next version will return a viewable URL.)"


@mcp.tool
async def run_gpu_code(
    code: str,
    requirements: list[str] = None
) -> str:
    """
    Run Python code on a Modal GPU (Tesla T4).
    
    Args:
        code: The Python code to execute
        requirements: Optional list of pip packages to install before running
    """
    payload = {
        "code": code,
        "requirements": requirements or []
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(CODE_ENDPOINT, json=payload)

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    result = response.json()
    
    output = []
    if result.get("stdout"):
        output.append(f"STDOUT:\n{result['stdout']}")
    if result.get("stderr"):
        output.append(f"STDERR:\n{result['stderr']}")
    output.append(f"Success: {result.get('success')}")
    
    return "\n\n".join(output)


@mcp.tool
async def check_gpu() -> str:
    """Check if the Modal GPU is available and working."""
    code = """
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
"""
    return await run_gpu_code(code)


# ═══════════════════════════════════════════════════════════════════
# Server Startup
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Modal GPU Agent MCP Server on port {port}")
    print(f"📍 Base URL: {BASE_URL}")
    print(f"✅ OAuth endpoints enabled for Claude integration")
    mcp.run(transport="http", host="0.0.0.0", port=port)
