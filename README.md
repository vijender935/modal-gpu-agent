# Modal GPU Agent

GPU compute on Modal + MCP for Grok:

- Image generation (Flux Schnell)
- Arbitrary Python on GPU
- **Google Drive smart crop** (YOLO Pose) – AI_Input → AI_Output

## One-time setup

### 1. Modal token (GitHub Actions deploy)

Repo → Settings → Secrets → Actions:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

### 2. Google Drive secret (on Modal dashboard)

Modal → Secrets → create or update **`google-drive`** with:

| Key | Value |
|-----|--------|
| `GOOGLE_OAUTH_TOKEN_JSON` | Full contents of the user's OAuth `token.json` (preferred for My Drive) |
| `INPUT_FOLDER_ID` | Drive folder ID for input |
| `OUTPUT_FOLDER_ID` | Drive folder ID for output |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Optional service-account JSON fallback |
| `DRIVE_ID` | Optional Shared Drive ID |

For the current `AI_Input` and `AI_Output` folders in My Drive, use `GOOGLE_OAUTH_TOKEN_JSON`. The application prefers OAuth user credentials when this key is present, so output files use the user's Drive quota. Keep the JSON only in the Modal secret; never commit it to GitHub. Shared Drive mode remains supported through the service account plus `DRIVE_ID`.

### 3. Deploy

Push to `main` → GitHub Actions runs `modal deploy app.py`.

Or manually: `modal deploy app.py`

## Endpoints (after deploy)

- `...-generate-image-endpoint.modal.run`
- `...-check-gpu-endpoint.modal.run`
- `...-run-python-sandbox-endpoint.modal.run`
- `...-service-health-endpoint.modal.run`
- `...-process-drive-async-endpoint.modal.run`
- `...-process-drive-status-endpoint.modal.run`
- `...-process-drive-endpoint.modal.run`

The old unrestricted code endpoint has been replaced by an isolated sandbox endpoint. It runs only preinstalled packages, blocks network access, receives no Drive/OAuth secrets, and enforces a 120-second timeout, bounded output, bounded files, and single-use containers. All Modal endpoints require `Authorization: Bearer <MODAL_ENDPOINT_TOKEN>`. Create a Modal secret named **`modal-endpoint-auth`** containing `MODAL_ENDPOINT_TOKEN`. The MCP gateway must use the same token in its `MODAL_ENDPOINT_TOKEN` environment variable. The authenticated service-health endpoint checks endpoint configuration and Drive OAuth health without exposing secret values.

## MCP (Render)

See `mcp-server/README.md`. The Render service requires `MCP_GATEWAY_TOKEN`; configure the Manus connector with the same bearer token. The unauthenticated health check is available at `/health`.

## Test from Grok

- "Check if GPU is working"
- "Generate an image of a cyberpunk city"
- "Run Python: check torch CUDA availability" (sandboxed, preinstalled packages only)
- "Start Drive processing and return a job ID"
- "Check Drive job status: <job ID>"
- "Process images from Drive" / "Run smart crop on AI_Input"

For large Drive batches, use the asynchronous start/status tools rather than holding one HTTP request open. Each endpoint response includes a request ID for log correlation. Before merging a production change, run a staging smoke test in this order: service health, wrong-token rejection, GPU health, small image generation, sandbox CPU/GPU execution, then one-image Drive processing.
