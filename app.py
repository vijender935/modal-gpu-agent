"""
Modal GPU Agent - Image Generation + Drive Processing + General GPU Compute
"""

import base64
import hmac
import io
import json
import logging
import mimetypes
import os
import resource
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import modal
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

app = modal.App("gpu-agent")
logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 2_000
MAX_DRIVE_FILES = 500
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 36_000_000
MAX_IMAGE_SIDE = 8_192
MAX_GENERATED_IMAGE_BYTES = 20 * 1024 * 1024
MAX_SANDBOX_CODE_LENGTH = 32_000
MAX_SANDBOX_TIMEOUT = 120
MAX_SANDBOX_OUTPUT_BYTES = 1_000_000
MAX_SANDBOX_FILE_BYTES = 2_000_000
MAX_SANDBOX_TOTAL_FILE_BYTES = 10_000_000
MAX_SANDBOX_FILES = 10
endpoint_bearer = HTTPBearer(auto_error=False)
endpoint_auth_dependency = Depends(endpoint_bearer)
_pose_model = None


def _require_endpoint_auth(
    credentials: HTTPAuthorizationCredentials | None = endpoint_auth_dependency,
) -> None:
    expected = os.getenv("MODAL_ENDPOINT_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Endpoint authentication is not configured",
        )
    if credentials is None or not hmac.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _validate_dimensions(
    width: int,
    height: int,
    *,
    max_side: int = 4096,
    require_multiple: bool = True,
) -> None:
    if isinstance(width, bool) or isinstance(height, bool):
        raise TypeError("width and height must be integers")
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError("width and height must be integers")
    if not (256 <= width <= max_side and 256 <= height <= max_side):
        raise ValueError(f"width and height must be between 256 and {max_side}")
    if require_multiple and (width % 8 or height % 8):
        raise ValueError("width and height must be multiples of 8")
    if width * height > 16_000_000:
        raise ValueError("requested image is too large")


def _new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _log_request(request_id: str, operation: str, status_code: int, started_at: float) -> None:
    logger.info(
        "modal_request request_id=%s operation=%s status=%s duration_ms=%d",
        request_id,
        operation,
        status_code,
        round((time.monotonic() - started_at) * 1000),
    )


def _validate_image_array(img) -> None:
    height, width = img.shape[:2]
    if height <= 0 or width <= 0 or height > MAX_IMAGE_SIDE or width > MAX_IMAGE_SIDE:
        raise ValueError("image dimensions exceed the allowed limit")
    if height * width > MAX_IMAGE_PIXELS:
        raise ValueError("image pixel count exceeds the allowed limit")


def _validate_prompt(prompt: str) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(f"prompt must be at most {MAX_PROMPT_LENGTH} characters")

# Base image with ML + Drive + OpenCV + YOLO deps
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "Pillow",
        "numpy",
        "opencv-python-headless",
        "ultralytics",
        "fastapi[standard]",
        "python-multipart",
        "google-api-python-client",
        "google-auth",
        "google-auth-httplib2",
    )
)

model_volume = modal.Volume.from_name("gpu-agent-models", create_if_missing=True)


def _get_drive_service():
    """Build Google Drive service using OAuth user credentials or service account fallback."""
    from googleapiclient.discovery import build

    scopes = ["https://www.googleapis.com/auth/drive"]
    oauth_json = os.environ.get("GOOGLE_OAUTH_TOKEN_JSON")

    if oauth_json:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        token_info = json.loads(oauth_json)
        creds = Credentials.from_authorized_user_info(token_info, scopes=scopes)
        if not creds.valid and creds.refresh_token:
            creds.refresh(Request())
        if not creds.valid:
            raise RuntimeError("Google OAuth credentials are invalid or expired")
    else:
        from google.oauth2 import service_account

        sa_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
        creds = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=scopes,
        )

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _shared_drive_params():
    """Return Google Drive API parameters needed for My Drive and Shared Drives."""
    params = {
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": True,
    }
    drive_id = os.environ.get("DRIVE_ID")
    if drive_id:
        params.update({"corpora": "drive", "driveId": drive_id})
    return params


def _list_images(service, folder_id: str, file_id: str | None = None):
    """List image files in a Drive folder, optionally selecting one file."""
    if file_id:
        q = f"id = '{file_id}' and trashed = false and mimeType contains 'image/'"
    else:
        q = (
            f"'{folder_id}' in parents and trashed = false and "
            "(mimeType contains 'image/' or "
            "name contains '.jpg' or name contains '.jpeg' or "
            "name contains '.png' or name contains '.webp')"
        )
    files = []
    page_token = None
    while True:
        params = {
            "q": q,
            "fields": "nextPageToken, files(id, name, mimeType, size)",
            "pageSize": 100,
            **_shared_drive_params(),
        }
        if page_token:
            params["pageToken"] = page_token
        results = service.files().list(**params).execute()
        files.extend(results.get("files", []))
        if len(files) > MAX_DRIVE_FILES:
            raise RuntimeError(f"input folder exceeds the {MAX_DRIVE_FILES}-file limit")
        page_token = results.get("nextPageToken")
        if not page_token:
            return files


def _list_output_names(service, folder_id: str) -> set[str]:
    """Return existing output filenames so repeated runs can be skipped safely."""
    names: set[str] = set()
    page_token = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(name)",
            "pageSize": 100,
            **_shared_drive_params(),
        }
        if page_token:
            params["pageToken"] = page_token
        result = service.files().list(**params).execute()
        names.update(item["name"] for item in result.get("files", []) if item.get("name"))
        page_token = result.get("nextPageToken")
        if not page_token:
            return names


def _download_file(service, file_id: str, dest_path: str):
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True,
    )
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _upload_file(service, local_path: str, folder_id: str, filename: str):
    from googleapiclient.http import MediaFileUpload

    # Escape apostrophes because Drive query strings use single quotes.
    safe_filename = filename.replace("'", "\\'")
    q = f"name = '{safe_filename}' and '{folder_id}' in parents and trashed = false"
    existing = (
        service.files()
        .list(
            q=q,
            fields="files(id)",
            pageSize=10,
            **_shared_drive_params(),
        )
        .execute()
        .get("files", [])
    )

    media = MediaFileUpload(local_path, resumable=True)
    if existing:
        service.files().update(
            fileId=existing[0]["id"],
            media_body=media,
            supportsAllDrives=True,
        ).execute()
        return existing[0]["id"]
    else:
        meta = {"name": filename, "parents": [folder_id]}
        created = (
            service.files()
            .create(
                body=meta,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            .execute()
        )
        return created["id"]


def _sandbox_preexec():
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_SANDBOX_OUTPUT_BYTES, MAX_SANDBOX_OUTPUT_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
    except (ValueError, OSError):
        pass


def _read_bounded_text(path: Path, limit: int = MAX_SANDBOX_OUTPUT_BYTES) -> str:
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    suffix = "\n[output truncated]" if len(data) > limit else ""
    return data[:limit].decode("utf-8", errors="replace") + suffix


def _collect_sandbox_files(workdir: Path) -> list[dict]:
    results = []
    total_bytes = 0
    ignored = {"main.py", "stdout.txt", "stderr.txt"}
    for path in sorted(workdir.rglob("*")):
        if len(results) >= MAX_SANDBOX_FILES or total_bytes >= MAX_SANDBOX_TOTAL_FILE_BYTES:
            break
        if not path.is_file() or path.is_symlink() or path.name in ignored:
            continue
        relative = path.relative_to(workdir)
        if any(part.startswith(".") for part in relative.parts):
            continue
        size = path.stat().st_size
        if size > MAX_SANDBOX_FILE_BYTES:
            continue
        remaining = MAX_SANDBOX_TOTAL_FILE_BYTES - total_bytes
        data = path.read_bytes()[: min(size, remaining)]
        total_bytes += len(data)
        results.append(
            {
                "path": str(relative),
                "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                "size": len(data),
                "data_base64": base64.b64encode(data).decode("ascii"),
                "truncated": len(data) != size,
            }
        )
    return results


def _execute_python_sandbox(code: str, timeout_seconds: int = 60) -> dict:
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")
    if len(code) > MAX_SANDBOX_CODE_LENGTH:
        raise ValueError(f"code must be at most {MAX_SANDBOX_CODE_LENGTH} characters")
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool):
        raise TypeError("timeout_seconds must be an integer")
    if not 1 <= timeout_seconds <= MAX_SANDBOX_TIMEOUT:
        raise ValueError(f"timeout_seconds must be between 1 and {MAX_SANDBOX_TIMEOUT}")

    secret_names = {
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_ENDPOINT_TOKEN",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_OAUTH_TOKEN_JSON",
        "INPUT_FOLDER_ID",
        "OUTPUT_FOLDER_ID",
    }
    safe_env = {
        key: value
        for key, value in os.environ.items()
        if key not in secret_names and not key.endswith("_SECRET") and not key.endswith("_TOKEN")
    }
    safe_env.update({"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})

    with tempfile.TemporaryDirectory(prefix="modal-sandbox-") as tempdir:
        workdir = Path(tempdir)
        (workdir / "main.py").write_text(code, encoding="utf-8")
        stdout_path = workdir / "stdout.txt"
        stderr_path = workdir / "stderr.txt"
        process = None
        timed_out = False
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-I", "main.py"],
                cwd=workdir,
                env=safe_env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                preexec_fn=_sandbox_preexec,  # noqa: PLW1509
            )
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait(timeout=5)

        return {
            "success": not timed_out and process.returncode == 0,
            "returncode": -9 if timed_out else process.returncode,
            "timed_out": timed_out,
            "stdout": _read_bounded_text(stdout_path),
            "stderr": _read_bounded_text(stderr_path),
            "files": _collect_sandbox_files(workdir),
        }


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=MAX_SANDBOX_TIMEOUT + 30,
    block_network=True,
    restrict_modal_access=True,
    single_use_containers=True,
    max_containers=2,
)
def run_python_sandbox_cpu(code: str, timeout_seconds: int = 60):
    return _execute_python_sandbox(code, timeout_seconds)


@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=8192,
    timeout=MAX_SANDBOX_TIMEOUT + 30,
    block_network=True,
    restrict_modal_access=True,
    single_use_containers=True,
    max_containers=2,
)
def run_python_sandbox_gpu(code: str, timeout_seconds: int = 60):
    return _execute_python_sandbox(code, timeout_seconds)


def _get_pose_model():
    global _pose_model
    if _pose_model is not None:
        return _pose_model

    import shutil

    from ultralytics import YOLO

    model_path = Path(os.getenv("YOLO_MODEL_PATH", "/models/yolov8n-pose.pt"))
    if model_path.exists():
        _pose_model = YOLO(str(model_path))
        return _pose_model

    _pose_model = YOLO("yolov8n-pose.pt")
    downloaded_path = getattr(_pose_model, "ckpt_path", None)
    if downloaded_path and Path(downloaded_path).exists():
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded_path, model_path)
        except OSError:
            logger.warning("Could not persist YOLO model at %s", model_path)
    return _pose_model


def _smart_crop(img, target_w=1080, target_h=2340):
    """
    YOLO Pose based smart crop (from original body cropper script).
    Returns (success, mode, cropped_bgr_image, message)
    """
    import cv2
    import numpy as np
    _validate_image_array(img)
    h, w = img.shape[:2]

    model = _get_pose_model()
    results = model(img, verbose=False)

    if (
        len(results) == 0
        or results[0].keypoints is None
        or len(results[0].keypoints.xy) == 0
    ):
        return False, None, None, "No pose detected"

    person_index = 0
    if results[0].boxes is not None and results[0].boxes.conf is not None:
        person_index = int(results[0].boxes.conf.argmax().item())
    kpts = results[0].keypoints.xy[person_index].cpu().numpy()
    confs = (
        results[0].keypoints.conf[person_index].cpu().numpy()
        if results[0].keypoints.conf is not None
        else np.zeros(17)
    )

    if len(kpts) < 13:
        return False, None, None, "Insufficient keypoints"

    mode = "closeup"
    knees_visible = confs[13] > 0.5 or confs[14] > 0.5
    ankles_visible = confs[15] > 0.5 or confs[16] > 0.5

    if knees_visible or ankles_visible:
        mode = "portrait"
    else:
        hips_visible = confs[11] > 0.5 or confs[12] > 0.5
        if hips_visible and confs[0] > 0.5:
            face_y_real = kpts[0][1]
            l_hip_y = kpts[11][1]
            r_hip_y = kpts[12][1]
            midriff_y_real = (
                (l_hip_y + r_hip_y) / 2
                if confs[11] > 0.5 and confs[12] > 0.5
                else max(l_hip_y, r_hip_y)
            )
            body_ratio = h / (midriff_y_real - face_y_real + 1)
            if body_ratio >= 3.0:
                mode = "portrait"

    face_y = int(kpts[0][1]) if kpts[0][1] > 0 else h // 6
    l_shoulder_x = kpts[5][0]
    r_shoulder_x = kpts[6][0]
    l_hip_y = kpts[11][1]
    r_hip_y = kpts[12][1]

    midriff_y = (
        int((l_hip_y + r_hip_y) / 2)
        if l_hip_y > 0 and r_hip_y > 0
        else h // 2
    )
    center_x = (
        int((l_shoulder_x + r_shoulder_x) / 2)
        if l_shoulder_x > 0 and r_shoulder_x > 0
        else w // 2
    )

    desired_ratio = target_w / target_h
    body_segment_height = midriff_y - face_y
    if body_segment_height <= 0:
        body_segment_height = h // 4

    if mode == "closeup":
        crop_h = int(body_segment_height * 1.45)
        headroom_ratio = 0.18
        noise_level = 1.5
        sharp_blend = 0.2
    else:
        crop_h = int(body_segment_height * 2.6)
        headroom_ratio = 0.15
        noise_level = 1.2
        sharp_blend = 0.3

    crop_w = int(crop_h * desired_ratio)

    if crop_h > h or crop_w > w:
        scale = min(h / crop_h, w / crop_w)
        crop_h = int(crop_h * scale)
        crop_w = int(crop_w * scale)

    ymin = face_y - int(crop_h * headroom_ratio)
    xmin = center_x - int(crop_w / 2)

    if ymin < 0:
        ymin = 0
    elif ymin + crop_h > h:
        ymin = h - crop_h

    if xmin < 0:
        xmin = 0
    elif xmin + crop_w > w:
        xmin = w - crop_w

    ymax = ymin + crop_h
    xmax = xmin + crop_w

    cropped = img[ymin:ymax, xmin:xmax]
    if cropped.size == 0:
        return False, None, None, "Crop failed"

    final = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(final, -1, sharpen_kernel)
    final = cv2.addWeighted(final, 1.0 - sharp_blend, sharpened, sharp_blend, 0)

    noise = np.random.normal(0, noise_level, final.shape)
    final = np.clip(final.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return True, mode, final, f"Processed with {mode} mode"


# -----------------------------
# 2. GPU health check
# -----------------------------
@app.function(
    image=image,
    gpu="T4",
    timeout=3 * 60,
    scaledown_window=60,
    max_containers=2,
)
def check_gpu_status():
    import torch

    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )
        if torch.cuda.is_available()
        else 0,
    }


# -----------------------------
# 3. Image Generation (Flux)
# -----------------------------
# Changed from A10 → T4 for better availability, lower cost, and consistency
# with the rest of the stack. Added robust load error handling to prevent
# crash-loops when the model fails to load or volume is empty.
@app.cls(
    image=image,
    gpu="T4",
    timeout=12 * 60,
    scaledown_window=5 * 60,
    max_containers=2,
    volumes={"/models": model_volume},
    memory=16384,
)
class ImageGenerator:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import FluxPipeline

        logger.info("ImageGenerator: starting FLUX.1-schnell model load...")
        try:
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.bfloat16,
                cache_dir="/models",
            ).to("cuda")
            # Quick sanity check that the pipeline is usable
            if self.pipe is None:
                raise RuntimeError("FluxPipeline loaded as None")
            logger.info("ImageGenerator: FLUX model loaded successfully")
        except Exception as exc:
            logger.exception("ImageGenerator: failed to load FLUX model")
            # Re-raise so Modal marks the container as failed instead of
            # leaving a half-initialized instance that will crash on generate()
            raise RuntimeError(f"Failed to load image generation model: {exc}") from exc

    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        seed: int | None = None,
    ) -> bytes:
        import torch

        if not hasattr(self, "pipe") or self.pipe is None:
            raise RuntimeError("Image generation model is not loaded")

        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=0.0,
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


# -----------------------------
# 4. Google Drive smart crop processor
# -----------------------------
@app.function(
    image=image,
    gpu="T4",
    timeout=15 * 60,
    scaledown_window=60,
    max_containers=1,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("google-drive")],
)
def process_drive_images(
    target_w: int = 1080,
    target_h: int = 2340,
    file_id: str | None = None,
    force_reprocess: bool = False,
):
    """
    Download images from Drive INPUT folder, smart-crop with YOLO Pose,
    upload results to OUTPUT folder.
    """
    import cv2

    input_folder_id = os.environ["INPUT_FOLDER_ID"]
    output_folder_id = os.environ["OUTPUT_FOLDER_ID"]

    _validate_dimensions(target_w, target_h, require_multiple=False)
    if file_id is not None and (not isinstance(file_id, str) or not file_id.strip()):
        raise ValueError("file_id must be a non-empty string when provided")

    try:
        service = _get_drive_service()
        files = _list_images(service, input_folder_id, file_id=file_id)
        existing_outputs = set() if force_reprocess else _list_output_names(service, output_folder_id)
    except Exception:
        logger.exception("Drive setup or listing failed")
        return {
            "success": False,
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "message": "Drive setup or listing failed",
            "details": [],
        }

    if not files:
        return {
            "success": True,
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "message": "No images found in input folder",
            "details": [],
        }

    details = []
    success_count = 0
    failed_count = 0
    skipped_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for fmeta in files:
            name = fmeta["name"]
            fid = fmeta["id"]
            try:
                file_size = int(fmeta.get("size") or 0)
            except (TypeError, ValueError):
                file_size = 0
            if file_size > MAX_IMAGE_BYTES:
                details.append({"file": name, "ok": False, "msg": "Image exceeds size limit"})
                failed_count += 1
                continue
            local_in = os.path.join(tmpdir, f"in_{fid}{Path(name).suffix.lower()}")

            if name in existing_outputs:
                details.append({"file": name, "ok": True, "skipped": True, "msg": "Output already exists"})
                skipped_count += 1
                continue

            try:
                _download_file(service, fid, local_in)
                img = cv2.imread(local_in)
                if img is None:
                    details.append({"file": name, "ok": False, "msg": "Failed to read image"})
                    failed_count += 1
                    continue
                _validate_image_array(img)

                ok, mode, cropped, msg = _smart_crop(img, target_w, target_h)
                if not ok:
                    details.append({"file": name, "ok": False, "msg": msg})
                    failed_count += 1
                    continue

                out_name = name
                if not out_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    out_name = out_name + ".jpg"
                local_out = os.path.join(tmpdir, f"out_{fid}{Path(out_name).suffix.lower()}")

                ext = Path(out_name).suffix.lower()
                if ext in (".jpg", ".jpeg"):
                    cv2.imwrite(local_out, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    cv2.imwrite(local_out, cropped)

                _upload_file(service, local_out, output_folder_id, out_name)
                details.append({"file": name, "ok": True, "msg": msg, "mode": mode})
                success_count += 1

            except Exception:
                logger.exception("Drive processing failed for file %s", name)
                details.append({"file": name, "ok": False, "msg": "Processing failed"})
                failed_count += 1

    return {
        "success": failed_count == 0,
        "processed": success_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "message": f"Done. Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}",
        "details": details,
    }


# -----------------------------
# Web endpoints
# -----------------------------
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("modal-endpoint-auth")],
)
@modal.fastapi_endpoint(method="POST")
def generate_image_endpoint(
    item: dict,
    _credentials: HTTPAuthorizationCredentials | None = endpoint_auth_dependency,
):
    _require_endpoint_auth(_credentials)
    request_id = _new_request_id()
    started_at = time.monotonic()
    prompt = item.get("prompt", "a beautiful landscape")
    width = item.get("width", 1024)
    height = item.get("height", 1024)
    seed = item.get("seed")
    try:
        _validate_prompt(prompt)
        _validate_dimensions(width, height, max_side=2048)
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        generator = ImageGenerator()
        image_bytes = generator.generate.remote(
            prompt=prompt, width=width, height=height, seed=seed
        )
    except Exception as exc:
        _log_request(request_id, "generate_image", 502, started_at)
        raise HTTPException(status_code=502, detail=f"image generation failed; request_id={request_id}") from exc
    if not image_bytes or len(image_bytes) > MAX_GENERATED_IMAGE_BYTES:
        _log_request(request_id, "generate_image", 502, started_at)
        raise HTTPException(status_code=502, detail=f"image output invalid; request_id={request_id}")

    from fastapi.responses import Response

    _log_request(request_id, "generate_image", 200, started_at)
    return Response(content=image_bytes, media_type="image/png", headers={"X-Request-ID": request_id})



@app.function(
    image=image,
    secrets=[modal.Secret.from_name("modal-endpoint-auth")],
)
@modal.fastapi_endpoint(method="POST")
def check_gpu_endpoint(
    _item: dict | None = None,
    _credentials: HTTPAuthorizationCredentials | None = endpoint_auth_dependency,
):
    _require_endpoint_auth(_credentials)
    request_id = _new_request_id()
    started_at = time.monotonic()
    try:
        result = check_gpu_status.remote()
    except Exception as exc:
        _log_request(request_id, "check_gpu", 502, started_at)
        raise HTTPException(status_code=502, detail=f"GPU health check failed; request_id={request_id}") from exc
    if isinstance(result, dict):
        result = {**result, "request_id": request_id}
    _log_request(request_id, "check_gpu", 200, started_at)
    return result


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("modal-endpoint-auth")],
)
@modal.fastapi_endpoint(method="POST")
def run_python_sandbox_endpoint(
    item: dict,
    _credentials: HTTPAuthorizationCredentials | None = endpoint_auth_dependency,
):
    _require_endpoint_auth(_credentials)
    request_id = _new_request_id()
    started_at = time.monotonic()
    code = item.get("code", "")
    timeout_seconds = item.get("timeout_seconds", 60)
    gpu = item.get("gpu", False)
    requirements = item.get("requirements", [])
    if requirements:
        raise HTTPException(
            status_code=422,
            detail="Dynamic package installation is disabled; use the prebuilt sandbox image",
        )
    if not isinstance(gpu, bool):
        raise HTTPException(status_code=422, detail="gpu must be a boolean")
    try:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        if len(code) > MAX_SANDBOX_CODE_LENGTH:
            raise ValueError(f"code must be at most {MAX_SANDBOX_CODE_LENGTH} characters")
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool):
            raise TypeError("timeout_seconds must be an integer")
        if not 1 <= timeout_seconds <= MAX_SANDBOX_TIMEOUT:
            raise ValueError(f"timeout_seconds must be between 1 and {MAX_SANDBOX_TIMEOUT}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    runner = run_python_sandbox_gpu if gpu else run_python_sandbox_cpu
    try:
        result = runner.remote(code=code, timeout_seconds=timeout_seconds)
    except Exception as exc:
        _log_request(request_id, "run_python", 502, started_at)
        raise HTTPException(status_code=502, detail=f"sandbox execution failed; request_id={request_id}") from exc
    if isinstance(result, dict):
        result = {**result, "request_id": request_id}
    _log_request(request_id, "run_python", 200, started_at)
    return result


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("google-drive"),
        modal.Secret.from_name("modal-endpoint-auth"),
    ],
)
@modal.fastapi_endpoint(method="GET")
def service_health_endpoint(
    _credentials: HTTPAuthorizationCredentials | None = endpoint_auth_dependency,
):
    _require_endpoint_auth(_credentials)
    checks = {
        "modal_endpoint_auth": bool(os.getenv("MODAL_ENDPOINT_TOKEN")),
        "drive_folder_config": bool(os.getenv("INPUT_FOLDER_ID") and os.getenv("OUTPUT_FOLDER_ID")),
        "drive_auth": "not_checked",
    }
    try:
        service = _get_drive_service()
        service.about().get(fields="user(emailAddress)").execute()
        checks["drive_auth"] = "ok"
    except Exception:
        logger.exception("Drive health check failed")
        checks["drive_auth"] = "error"
    healthy = all(value in {True, "ok"} for value in checks.values())
    from fastapi.responses import JSONResponse

    return JSONResponse(
        {
            "status": "healthy" if healthy else "degraded",
            "service": "modal-gpu-agent",
            "checks": checks,
        },
        status_code=200 if healthy else 503,
    )


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("google-drive"),
        modal.Secret.from_name("modal-endpoint-auth"),
    ],
)
@modal.fastapi_endpoint(method="POST")
def process_drive_async_endpoint(
    item: dict | None = None,
    _credentials: HTTPAuthorizationCredentials | None = endpoint_auth_dependency,
):
    _require_endpoint_auth(_credentials)
    item = item or {}
    target_w = item.get("target_w", 1080)
    target_h = item.get("target_h", 2340)
    file_id = item.get("file_id")
    force_reprocess = item.get("force_reprocess", False)
    try:
        _validate_dimensions(target_w, target_h, require_multiple=False)
        if file_id is not None and (not isinstance(file_id, str) or not file_id.strip()):
            raise ValueError("file_id must be a non-empty string when provided")
        if not isinstance(force_reprocess, bool):
            raise TypeError("force_reprocess must be a boolean")
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    call = process_drive_images.spawn(
        target_w=target_w,
        target_h=target_h,
        file_id=file_id,
        force_reprocess=force_reprocess,
    )
    return {"status": "queued", "job_id": call.object_id}


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("modal-endpoint-auth")],
)
@modal.fastapi_endpoint(method="GET")
def process_drive_status_endpoint(
    job_id: str,
    _credentials: HTTPAuthorizationCredentials | None = endpoint_auth_dependency,
):
    _require_endpoint_auth(_credentials)
    if not isinstance(job_id, str) or not job_id.strip() or len(job_id) > 200:
        raise HTTPException(status_code=422, detail="job_id is invalid")
    try:
        call = modal.FunctionCall.from_id(job_id)
        result = call.get(timeout=0)
    except TimeoutError:
        return {"status": "running", "job_id": job_id}
    except Exception as exc:
        logger.exception("Drive job lookup failed for %s", job_id)
        raise HTTPException(status_code=404, detail="job not found") from exc
    return {"status": "succeeded", "job_id": job_id, "result": result}


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("google-drive"),
        modal.Secret.from_name("modal-endpoint-auth"),
    ],
)
@modal.fastapi_endpoint(method="POST")
def process_drive_endpoint(
    item: dict | None = None,
    _credentials: HTTPAuthorizationCredentials | None = endpoint_auth_dependency,
):
    _require_endpoint_auth(_credentials)
    request_id = _new_request_id()
    started_at = time.monotonic()
    item = item or {}
    target_w = item.get("target_w", 1080)
    target_h = item.get("target_h", 2340)
    file_id = item.get("file_id")
    force_reprocess = item.get("force_reprocess", False)
    try:
        _validate_dimensions(target_w, target_h, require_multiple=False)
        if file_id is not None and (not isinstance(file_id, str) or not file_id.strip()):
            raise ValueError("file_id must be a non-empty string when provided")
        if not isinstance(force_reprocess, bool):
            raise TypeError("force_reprocess must be a boolean")
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        result = process_drive_images.remote(
            target_w=target_w,
            target_h=target_h,
            file_id=file_id,
            force_reprocess=force_reprocess,
        )
    except Exception as exc:
        _log_request(request_id, "process_drive_images", 502, started_at)
        raise HTTPException(status_code=502, detail=f"Drive processing failed; request_id={request_id}") from exc
    if isinstance(result, dict):
        result = {**result, "request_id": request_id}
    _log_request(request_id, "process_drive_images", 200, started_at)
    return result


@app.local_entrypoint()
def main():
    print("GPU Agent ready.")
    print("Deploy: modal deploy app.py")
    print("Secrets required: google-drive and modal-endpoint-auth")
    print("  Drive keys: GOOGLE_OAUTH_TOKEN_JSON, INPUT_FOLDER_ID, OUTPUT_FOLDER_ID")
    print("  Endpoint key: MODAL_ENDPOINT_TOKEN")
