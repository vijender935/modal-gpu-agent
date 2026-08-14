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

Modal → Secrets → create **`google-drive`** with:

| Key | Value |
|-----|--------|
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Full service account JSON |
| `INPUT_FOLDER_ID` | Shared Drive folder ID for input |
| `OUTPUT_FOLDER_ID` | Shared Drive folder ID for output |
| `DRIVE_ID` | Shared Drive ID (recommended) |

Create `AI_Input` and `AI_Output` inside the same Google Shared Drive. Share both folders, or the Shared Drive itself, with the service-account email as **Content manager**. Set `DRIVE_ID` to the Shared Drive ID so the API searches the correct drive. A normal My Drive folder may still fail on upload because service accounts do not have personal storage quota.

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
