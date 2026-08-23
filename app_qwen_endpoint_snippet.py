# ==============================
# 5. Qwen2.5-VL RAG Captioning (H100)
# Paste this entire block into app.py RIGHT BEFORE @app.local_entrypoint()
# ==============================
from qwen_captioner import build_qwen_captioner

generate_rag_captions = build_qwen_captioner(app)


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("modal-endpoint-auth"),
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("google-drive"),
    ],
)
@modal.fastapi_endpoint(method="POST")
def generate_rag_captions_endpoint(
    item: dict | None = None,
    _credentials: HTTPAuthorizationCredentials | None = endpoint_auth_dependency,
):
    """Queue Qwen2.5-VL H100 captioning job for multimodal RAG dataset generation."""
    _require_endpoint_auth(_credentials)
    item = item or {}
    max_items = item.get("max_items", 20)
    start_from = item.get("start_from", 0)
    model_name = item.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
    output_filename = item.get("output_filename", "metadata_qwen_v1.jsonl")

    if not isinstance(max_items, int) or isinstance(max_items, bool) or max_items < 1 or max_items > 500:
        raise HTTPException(status_code=422, detail="max_items must be an integer between 1 and 500")
    if not isinstance(start_from, int) or isinstance(start_from, bool) or start_from < 0:
        raise HTTPException(status_code=422, detail="start_from must be a non-negative integer")
    if not isinstance(model_name, str) or not model_name.strip():
        raise HTTPException(status_code=422, detail="model_name must be a non-empty string")
    if not isinstance(output_filename, str) or not output_filename.endswith(".jsonl"):
        raise HTTPException(status_code=422, detail="output_filename must end with .jsonl")

    call = generate_rag_captions.spawn(
        max_items=max_items,
        start_from=start_from,
        model_name=model_name,
        output_filename=output_filename,
    )
    return {
        "status": "queued",
        "job_id": call.object_id,
        "message": "Qwen2.5-VL H100 captioning started",
    }
