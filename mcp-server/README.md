# MCP Server for Modal GPU Agent

Control Modal GPU functions from Grok.

## Tools

- `generate_image` – Flux image generation with an MCP image artifact response
- `check_gpu` – CUDA / GPU health check without user-provided code
- `process_images_from_drive` – YOLO smart-crop all images from Drive AI_Input → AI_Output

Arbitrary Python execution is intentionally not exposed in the production MCP server. If that capability is ever needed, it must be implemented as a separate sandboxed service with its own identity, network policy, package policy, and cost quota.

## Deploy on Render

1. Web Service, connect repo `vijender935/modal-gpu-agent`.
2. Root Directory: `mcp-server`.
3. Build: `pip install -r requirements.txt`.
4. Start: `python server.py`.
5. Required environment variables:
   - `MCP_GATEWAY_TOKEN` – long random bearer token required by MCP clients.
   - `MODAL_ENDPOINT_TOKEN` – same token stored in the Modal `modal-endpoint-auth` secret.
6. Optional endpoint overrides:
   - `IMAGE_ENDPOINT`
   - `GPU_ENDPOINT`
   - `PROCESS_ENDPOINT`

Health check: `GET https://YOUR-RENDER-URL/health`.

## Grok connector

URL: `https://YOUR-RENDER-URL/mcp`

Configure the connector with the header `Authorization: Bearer <MCP_GATEWAY_TOKEN>`. Never put the token in the URL or commit it to GitHub.

## Modal secret required

Secret name: `google-drive`

For normal My Drive folders, add the full contents of the user's OAuth `token.json` as `GOOGLE_OAUTH_TOKEN_JSON`. This is the preferred mode because output files use the user's Drive quota. Keep the token only in the Modal secret and never commit it to GitHub.

Keys:
- `GOOGLE_OAUTH_TOKEN_JSON` – full OAuth authorized-user token JSON (preferred)
- `INPUT_FOLDER_ID` – Drive folder ID for input images
- `OUTPUT_FOLDER_ID` – Drive folder ID for output images
- `GOOGLE_SERVICE_ACCOUNT_JSON` – optional fallback for Shared Drives
- `DRIVE_ID` – optional Shared Drive ID

If `GOOGLE_OAUTH_TOKEN_JSON` is present, the application uses it instead of the service account. Shared Drive mode remains supported with `supportsAllDrives=true`, Shared Drive-aware listing, and pagination.
