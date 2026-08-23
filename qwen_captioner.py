"""
Qwen2.5-VL Captioner for Multimodal RAG Dataset Generation
Runs on Modal H100. Produces structured JSONL entries.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

logger = logging.getLogger(__name__)

# Separate image with Qwen VL dependencies
qwen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.45.0",
        "accelerate",
        "qwen-vl-utils",
        "Pillow",
        "huggingface_hub",
        "google-api-python-client",
        "google-auth",
        "google-auth-httplib2",
    )
)

model_volume = modal.Volume.from_name("gpu-agent-models", create_if_missing=True)

SYSTEM_PROMPT = """You are an expert visual analyst for creating high-quality multimodal RAG datasets of attractive/hot images.
Analyze the image carefully and return ONLY a valid JSON object with these exact fields:

{
  "caption": "Detailed visual description including outfit, pose, expression, body features, lighting and overall mood",
  "short_caption": "Short 8-15 word summary",
  "tags": ["tag1", "tag2", "tag3", "tag4"],
  "outfit": {
    "primary": "main clothing item",
    "details": ["detail1", "detail2"],
    "color": ["color1", "color2"],
    "style": "style description"
  },
  "pose": "description of pose",
  "expression": "facial expression",
  "body_focus": ["face", "cleavage", "midriff"],
  "setting": "location/background",
  "lighting": "lighting type",
  "intensity": 7,
  "nsfw_level": 7,
  "mood": ["seductive", "confident"]
}

Rules:
- intensity and nsfw_level must be integers from 1 to 10
- Be accurate and detailed
- If the image is revealing or explicit, describe it honestly
- Return ONLY the JSON, no extra text
"""


def _get_drive_service():
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
            sa_info, scopes=scopes
        )

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _download_file(service, file_id: str, dest_path: str):
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _upload_file(service, local_path: str, folder_id: str, filename: str):
    from googleapiclient.http import MediaFileUpload

    safe_filename = filename.replace("'", "\\'")
    q = f"name = '{safe_filename}' and '{folder_id}' in parents and trashed = false"
    existing = (
        service.files()
        .list(
            q=q,
            fields="files(id)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
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


def _list_images(service, folder_id: str, max_files: int = 500):
    q = (
        f"'{folder_id}' in parents and trashed = false and "
        "(mimeType contains 'image/' or name contains '.jpg' or "
        "name contains '.jpeg' or name contains '.png' or name contains '.webp')"
    )
    files = []
    page_token = None
    while True:
        params = {
            "q": q,
            "fields": "nextPageToken, files(id, name, mimeType, size)",
            "pageSize": 100,
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
        }
        if page_token:
            params["pageToken"] = page_token
        results = service.files().list(**params).execute()
        files.extend(results.get("files", []))
        if len(files) >= max_files:
            return files[:max_files]
        page_token = results.get("nextPageToken")
        if not page_token:
            return files


# This function is meant to be registered on the main Modal app
def build_qwen_captioner(app: modal.App):
    """Register Qwen2.5-VL captioning function on the given Modal app."""

    @app.function(
        image=qwen_image,
        gpu="H100",
        timeout=60 * 60 * 3,
        memory=65536,
        volumes={"/models": model_volume},
        secrets=[
            modal.Secret.from_name("huggingface"),
            modal.Secret.from_name("google-drive"),
        ],
        scaledown_window=5 * 60,
        max_containers=1,
    )
    def generate_rag_captions(
        max_items: int = 50,
        start_from: int = 0,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        output_filename: str = "metadata_qwen_v1.jsonl",
    ) -> dict[str, Any]:
        """
        Download images from Drive INPUT folder, caption with Qwen2.5-VL on H100,
        and upload structured JSONL to OUTPUT folder.
        """
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        from PIL import Image

        input_folder_id = os.environ.get("INPUT_FOLDER_ID")
        output_folder_id = os.environ.get("OUTPUT_FOLDER_ID")
        if not input_folder_id or not output_folder_id:
            return {
                "success": False,
                "message": "INPUT_FOLDER_ID or OUTPUT_FOLDER_ID not configured",
                "processed": 0,
                "errors": 0,
            }

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            return {
                "success": False,
                "message": "HF_TOKEN missing. Add huggingface secret with HF_TOKEN.",
                "processed": 0,
                "errors": 0,
            }

        logger.info("Loading model %s on H100...", model_name)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="/models",
            token=hf_token,
        )
        processor = AutoProcessor.from_pretrained(model_name, token=hf_token)

        service = _get_drive_service()
        files = _list_images(service, input_folder_id, max_files=start_from + max_items)
        files = files[start_from : start_from + max_items]

        if not files:
            return {
                "success": True,
                "message": "No images found",
                "processed": 0,
                "errors": 0,
            }

        results = []
        errors = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, fmeta in enumerate(files):
                name = fmeta["name"]
                fid = fmeta["id"]
                local_path = os.path.join(tmpdir, f"{fid}{Path(name).suffix.lower()}")

                try:
                    _download_file(service, fid, local_path)
                    img = Image.open(local_path).convert("RGB")

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": local_path},
                                {"type": "text", "text": SYSTEM_PROMPT},
                            ],
                        }
                    ]

                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(model.device)

                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.3,
                            do_sample=False,
                        )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()

                    if output_text.startswith("```"):
                        output_text = output_text.split("```")[1]
                        if output_text.startswith("json"):
                            output_text = output_text[4:]
                    output_text = output_text.strip()

                    data = json.loads(output_text)

                    entry = {
                        "id": f"img_{fid[:12]}",
                        "type": "image",
                        "file_name": name,
                        "drive_file_id": fid,
                        "file_path": f"Input images/{name}",
                        **data,
                        "model_used": model_name,
                        "source": "Input images",
                    }
                    results.append(entry)
                    logger.info("Captioned %s (%d/%d)", name, idx + 1, len(files))

                except Exception as e:
                    logger.exception("Failed on %s", name)
                    errors.append({"file": name, "error": str(e)})

            # Write JSONL and upload
            local_jsonl = os.path.join(tmpdir, output_filename)
            with open(local_jsonl, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            _upload_file(service, local_jsonl, output_folder_id, output_filename)

            if errors:
                err_name = output_filename.replace(".jsonl", "_errors.jsonl")
                local_err = os.path.join(tmpdir, err_name)
                with open(local_err, "w", encoding="utf-8") as f:
                    for e in errors:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                _upload_file(service, local_err, output_folder_id, err_name)

        return {
            "success": True,
            "processed": len(results),
            "errors": len(errors),
            "output_file": output_filename,
            "message": f"Done. Processed {len(results)}, errors {len(errors)}",
        }

    return generate_rag_captions
