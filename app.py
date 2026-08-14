"""
Modal GPU Agent - Image Generation + Drive Processing + General GPU Compute
"""

import modal
import io
import os
import json
import tempfile
from pathlib import Path

app = modal.App("gpu-agent")

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
    """Build Google Drive service from Modal secret."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    sa_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/drive"],
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


def _list_images(service, folder_id: str):
    """List all image files in a Drive folder, including Shared Drive items."""
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
            "fields": "nextPageToken, files(id, name, mimeType)",
            "pageSize": 100,
            **_shared_drive_params(),
        }
        if page_token:
            params["pageToken"] = page_token
        results = service.files().list(**params).execute()
        files.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            return files


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


def _smart_crop(img, target_w=1080, target_h=2340):
    """
    YOLO Pose based smart crop (from original body cropper script).
    Returns (success, mode, cropped_bgr_image, message)
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

    h, w = img.shape[:2]

    model = YOLO("yolov8n-pose.pt")
    results = model(img, verbose=False)

    if (
        len(results) == 0
        or results[0].keypoints is None
        or len(results[0].keypoints.xy) == 0
    ):
        return False, None, None, "No pose detected"

    kpts = results[0].keypoints.xy[0].cpu().numpy()
    confs = (
        results[0].keypoints.conf[0].cpu().numpy()
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
# 1. General GPU Python Runner
# -----------------------------
@app.function(
    image=image,
    gpu="T4",
    timeout=10 * 60,
    scaledown_window=60,
)
def run_python_code(code: str, requirements: list[str] = None):
    import subprocess
    import sys

    if requirements:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + requirements
        )

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
# 2. Image Generation (Flux)
# -----------------------------
@app.cls(
    image=image,
    gpu="A10",
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
# 3. Google Drive smart crop processor
# -----------------------------
@app.function(
    image=image,
    gpu="T4",
    timeout=15 * 60,
    scaledown_window=60,
    secrets=[modal.Secret.from_name("google-drive")],
)
def process_drive_images(target_w: int = 1080, target_h: int = 2340):
    """
    Download images from Drive INPUT folder, smart-crop with YOLO Pose,
    upload results to OUTPUT folder.
    """
    import cv2

    input_folder_id = os.environ["INPUT_FOLDER_ID"]
    output_folder_id = os.environ["OUTPUT_FOLDER_ID"]

    service = _get_drive_service()
    files = _list_images(service, input_folder_id)

    if not files:
        return {
            "success": True,
            "processed": 0,
            "failed": 0,
            "message": "No images found in input folder",
            "details": [],
        }

    details = []
    success_count = 0
    failed_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for fmeta in files:
            name = fmeta["name"]
            fid = fmeta["id"]
            local_in = os.path.join(tmpdir, f"in_{name}")
            local_out = os.path.join(tmpdir, f"out_{name}")

            try:
                _download_file(service, fid, local_in)
                img = cv2.imread(local_in)
                if img is None:
                    details.append({"file": name, "ok": False, "msg": "Failed to read image"})
                    failed_count += 1
                    continue

                ok, mode, cropped, msg = _smart_crop(img, target_w, target_h)
                if not ok:
                    details.append({"file": name, "ok": False, "msg": msg})
                    failed_count += 1
                    continue

                out_name = name
                if not out_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    out_name = out_name + ".jpg"

                ext = Path(out_name).suffix.lower()
                if ext in (".jpg", ".jpeg"):
                    cv2.imwrite(local_out, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    cv2.imwrite(local_out, cropped)

                _upload_file(service, local_out, output_folder_id, out_name)
                details.append({"file": name, "ok": True, "msg": msg, "mode": mode})
                success_count += 1

            except Exception as e:
                details.append({"file": name, "ok": False, "msg": str(e)})
                failed_count += 1

    return {
        "success": failed_count == 0,
        "processed": success_count,
        "failed": failed_count,
        "message": f"Done. Success: {success_count}, Failed: {failed_count}",
        "details": details,
    }


# -----------------------------
# Web endpoints
# -----------------------------
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate_image_endpoint(item: dict):
    prompt = item.get("prompt", "a beautiful landscape")
    width = item.get("width", 1024)
    height = item.get("height", 1024)
    seed = item.get("seed")

    generator = ImageGenerator()
    image_bytes = generator.generate.remote(
        prompt=prompt, width=width, height=height, seed=seed
    )

    from fastapi.responses import Response

    return Response(content=image_bytes, media_type="image/png")


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def run_code_endpoint(item: dict):
    code = item.get("code", "print('No code provided')")
    requirements = item.get("requirements", [])
    return run_python_code.remote(code=code, requirements=requirements)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("google-drive")],
)
@modal.fastapi_endpoint(method="POST")
def process_drive_endpoint(item: dict = None):
    item = item or {}
    target_w = item.get("target_w", 1080)
    target_h = item.get("target_h", 2340)
    return process_drive_images.remote(target_w=target_w, target_h=target_h)


@app.local_entrypoint()
def main():
    print("GPU Agent ready.")
    print("Deploy: modal deploy app.py")
    print("Secret required: google-drive")
    print("  Keys: GOOGLE_SERVICE_ACCOUNT_JSON, INPUT_FOLDER_ID, OUTPUT_FOLDER_ID")
