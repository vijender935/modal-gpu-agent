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

Modal → Secrets → create **`google-drive-secret`** with:

| Key | Value |
|-----|--------|
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Full service account JSON |
| `INPUT_FOLDER_ID` | Drive folder ID for input |
| `OUTPUT_FOLDER_ID` | Drive folder ID for output |

Share both Drive folders with the service account email (Editor).

### 3. Deploy

Push to `main` → GitHub Actions runs `modal deploy app.py`.

Or manually: `modal deploy app.py`

## Endpoints (after deploy)

- `...-generate-image-endpoint.modal.run`
- `...-run-code-endpoint.modal.run`
- `...-process-drive-endpoint.modal.run`

## MCP (Render)

See `mcp-server/README.md`.

## Test from Grok

- "Check if GPU is working"
- "Generate an image of a cyberpunk city"
- "Process images from Drive" / "Run smart crop on AI_Input"
