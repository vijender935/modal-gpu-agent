# Qwen2.5-VL + H100 RAG Captioning Tool

## What was added

1. `qwen_captioner.py` — H100 + Qwen2.5-VL captioning module
2. MCP tool `generate_rag_captions` in `mcp-server/server.py`
3. This setup guide

## Required: Wire into app.py

Add these lines to `app.py` **before** `@app.local_entrypoint()`:

```python
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
    _require_endpoint_auth(_credentials)
    item = item or {}
    max_items = item.get("max_items", 20)
    start_from = item.get("start_from", 0)
    model_name = item.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
    output_filename = item.get("output_filename", "metadata_qwen_v1.jsonl")

    if not isinstance(max_items, int) or max_items < 1 or max_items > 500:
        raise HTTPException(status_code=422, detail="max_items must be 1-500")
    if not isinstance(start_from, int) or start_from < 0:
        raise HTTPException(status_code=422, detail="start_from must be >= 0")

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
```

## Deploy

```bash
modal deploy app.py
```

After deploy, the endpoint will be:
`https://vijender935--gpu-agent-generate-rag-captions-endpoint.modal.run`

## MCP tool usage (from Grok)

```
generate_rag_captions(
  max_items=20,
  start_from=0,
  model_name="Qwen/Qwen2.5-VL-7B-Instruct",
  output_filename="metadata_qwen_v1.jsonl"
)
```

## Notes

- Needs H100 availability on your Modal account
- Needs `huggingface` secret with `HF_TOKEN`
- Needs `google-drive` secret with INPUT/OUTPUT folder IDs
- Start with max_items=10-20 for testing
- Output JSONL is uploaded to Drive OUTPUT folder
