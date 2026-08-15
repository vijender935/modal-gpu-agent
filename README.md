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
- `...-process-drive-endpoint.modal.run`

The arbitrary-code endpoint has been removed from the production application. All remaining Modal endpoints require `Authorization: Bearer <MODAL_ENDPOINT_TOKEN>`. Create a Modal secret named **`modal-endpoint-auth`** containing `MODAL_ENDPOINT_TOKEN`. The MCP gateway must use the same token in its `MODAL_ENDPOINT_TOKEN` environment variable.

## MCP (Render)

See `mcp-server/README.md`. The Render service requires `MCP_GATEWAY_TOKEN`; configure the Manus connector with the same bearer token. The unauthenticated health check is available at `/health`.

## Test from Grok

- "Check if GPU is working"
- "Generate an image of a cyberpunk city"
- "Process images from Drive" / "Run smart crop on AI_Input"
