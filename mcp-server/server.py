"""
MCP Server for Modal GPU Agent
Exposes image generation, GPU code execution, and Drive processing to Grok
"""

import os
import httpx
from fastmcp import FastMCP

IMAGE_ENDPOINT = os.getenv(
    "IMAGE_ENDPOINT",
    "https://vijender935--gpu-agent-generate-image-endpoint.modal.run",
)
CODE_ENDPOINT = os.getenv(
    "CODE_ENDPOINT",
    "https://vijender935--gpu-agent-run-code-endpoint.modal.run",
)
PROCESS_ENDPOINT = os.getenv(
    "PROCESS_ENDPOINT",
    "https://vijender935--gpu-agent-process-drive-endpoint.modal.run",
)

mcp = FastMCP("Modal GPU Agent")


@mcp.tool
async def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = None,
) -> str:
    """
    Generate an image using Flux model on Modal GPU.

    Args:
        prompt: Text description of the image you want to generate
        width: Image width (default 1024)
        height: Image height (default 1024)
        seed: Optional random seed for reproducibility
    """
    payload = {"prompt": prompt, "width": width, "height": height}
    if seed is not None:
        payload["seed"] = seed

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(IMAGE_ENDPOINT, json=payload)

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text[:500]}"

    return (
        f"Image generated successfully for prompt: '{prompt}'. "
        f"Size: {len(response.content)} bytes (PNG). "
        f"(Binary received; URL return can be added later.)"
    )


@mcp.tool
async def run_gpu_code(code: str, requirements: list[str] = None) -> str:
    """
    Run Python code on a Modal GPU (Tesla T4).

    Args:
        code: The Python code to execute
        requirements: Optional list of pip packages to install before running
    """
    payload = {"code": code, "requirements": requirements or []}

    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(CODE_ENDPOINT, json=payload)

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text[:500]}"

    result = response.json()
    parts = []
    if result.get("stdout"):
        parts.append(f"STDOUT:\n{result['stdout']}")
    if result.get("stderr"):
        parts.append(f"STDERR:\n{result['stderr']}")
    parts.append(f"Success: {result.get('success')}")
    return "\n\n".join(parts)


@mcp.tool
async def check_gpu() -> str:
    """Check if the Modal GPU is available and working."""
    code = """
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    print("No CUDA GPU")
"""
    return await run_gpu_code(code)


@mcp.tool
async def process_images_from_drive(
    target_w: int = 1080,
    target_h: int = 2340,
) -> str:
    """
    Process all images from Google Drive AI_Input folder using YOLO smart crop
    and save results to AI_Output folder.

    Args:
        target_w: Output width (default 1080)
        target_h: Output height (default 2340)
    """
    payload = {"target_w": target_w, "target_h": target_h}

    async with httpx.AsyncClient(timeout=900.0) as client:
        response = await client.post(PROCESS_ENDPOINT, json=payload)

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text[:800]}"

    result = response.json()
    lines = [
        result.get("message", "Done"),
        f"Processed: {result.get('processed', 0)}",
        f"Failed: {result.get('failed', 0)}",
    ]
    for d in result.get("details", [])[:30]:
        status = "OK" if d.get("ok") else "FAIL"
        lines.append(f"  [{status}] {d.get('file')}: {d.get('msg')}")
    return "\n".join(lines)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Modal GPU Agent MCP Server on port {port}")
    mcp.run(transport="http", host="0.0.0.0", port=port)
