"""
Modal GPU Agent - Image Generation + General Heavy Compute
Ready for MCP wrapping later
"""

import modal
import io
from pathlib import Path

# -----------------------------
# Modal App + Image
# -----------------------------
app = modal.App("gpu-agent")

# Base image with common ML libraries
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "Pillow",
        "numpy",
        "fastapi[standard]",
        "python-multipart",
    )
)

# Volume for caching models (optional but recommended)
model_volume = modal.Volume.from_name("gpu-agent-models", create_if_missing=True)


# -----------------------------
# 1. General GPU Python Runner
# -----------------------------
@app.function(
    image=image,
    gpu="T4",                    # Start cheap, upgrade later (A10, L40S, A100...)
    timeout=10 * 60,             # 10 minutes
    scaledown_window=60,
)
def run_python_code(code: str, requirements: list[str] = None):
    """
    Run arbitrary Python code on GPU.
    Useful for custom heavy compute tasks.
    """
    import subprocess
    import sys
    import tempfile
    import os

    # Install extra requirements if provided
    if requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + requirements)

    # Write code to temp file and execute
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=9 * 60,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }
    finally:
        os.unlink(temp_path)


# -----------------------------
# 2. Image Generation (Flux Schnell - Fast)
# -----------------------------
@app.cls(
    image=image,
    gpu="A10",                   # Better quality/speed balance
    timeout=10 * 60,
    scaledown_window=2 * 60,
    volumes={"/models": model_volume},
)
class ImageGenerator:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import FluxPipeline

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            cache_dir="/models",
        ).to("cuda")

    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        seed: int = None,
    ) -> bytes:
        import torch
        from PIL import Image

        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=0.0,  # Schnell doesn't need guidance
        ).images[0]

        # Convert to bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


# -----------------------------
# 3. Simple Web Endpoints (for testing)
# -----------------------------
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate_image_endpoint(item: dict):
    """
    Test endpoint:
    POST {"prompt": "a cyberpunk city", "width": 1024, "height": 1024}
    """
    prompt = item.get("prompt", "a beautiful landscape")
    width = item.get("width", 1024)
    height = item.get("height", 1024)
    seed = item.get("seed")

    generator = ImageGenerator()
    image_bytes = generator.generate.remote(
        prompt=prompt,
        width=width,
        height=height,
        seed=seed,
    )

    from fastapi.responses import Response
    return Response(content=image_bytes, media_type="image/png")


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def run_code_endpoint(item: dict):
    """
    Test endpoint for general code:
    POST {"code": "print('hello from GPU')", "requirements": []}
    """
    code = item.get("code", "print('No code provided')")
    requirements = item.get("requirements", [])
    result = run_python_code.remote(code=code, requirements=requirements)
    return result


# -----------------------------
# Local entrypoint for testing
# -----------------------------
@app.local_entrypoint()
def main():
    print("🚀 GPU Agent is ready.")
    print("Deploy with: modal deploy app.py")
    print("Then test the endpoints that Modal prints.")
