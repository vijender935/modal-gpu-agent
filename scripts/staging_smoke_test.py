"""Manual staging smoke test for the deployed Modal endpoints.

Usage:
  MODAL_ENDPOINT_TOKEN=... \
  GPU_ENDPOINT=https://... \
  IMAGE_ENDPOINT=https://... \
  SANDBOX_ENDPOINT=https://... \
  HEALTH_ENDPOINT=https://... \
  python scripts/staging_smoke_test.py

Drive submission is optional. Set ASYNC_PROCESS_ENDPOINT to exercise it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx


def require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value



def main() -> int:
    token = require("MODAL_ENDPOINT_TOKEN")
    endpoints = {
        "health": os.getenv("HEALTH_ENDPOINT"),
        "gpu": require("GPU_ENDPOINT"),
        "image": require("IMAGE_ENDPOINT"),
        "sandbox": require("SANDBOX_ENDPOINT"),
        "async_process": os.getenv("ASYNC_PROCESS_ENDPOINT"),
    }
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(timeout=180, follow_redirects=False) as client:
        if endpoints["health"]:
            response = client.get(endpoints["health"])
            print(f"health: {response.status_code} {response.text[:300]}")
            response.raise_for_status()

        response = client.post(endpoints["gpu"], headers=headers, json={})
        print(f"gpu: {response.status_code} {response.text[:300]}")
        response.raise_for_status()

        response = client.post(
            endpoints["sandbox"],
            headers=headers,
            json={"code": "import torch; print(torch.cuda.is_available())", "gpu": True, "timeout_seconds": 30},
        )
        print(f"sandbox: {response.status_code} {response.text[:300]}")
        response.raise_for_status()

        response = client.post(
            endpoints["image"],
            headers=headers,
            json={"prompt": "a simple blue circle on a white background", "width": 512, "height": 512, "seed": 7},
        )
        print(f"image: {response.status_code} {len(response.content)} bytes")
        response.raise_for_status()
        Path("staging-smoke-output.png").write_bytes(response.content)

        if endpoints["async_process"]:
            response = client.post(endpoints["async_process"], headers=headers, json={})
            print(f"drive-async: {response.status_code} {response.text[:300]}")
            response.raise_for_status()

    print("staging smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
