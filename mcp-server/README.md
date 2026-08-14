# MCP Server for Modal GPU Agent

Control Modal GPU functions from Grok.

## Tools

- `generate_image` – Flux image generation
- `run_gpu_code` – run arbitrary Python on T4 GPU
- `check_gpu` – CUDA / GPU health check
- `process_images_from_drive` – YOLO smart-crop all images from Drive AI_Input → AI_Output

## Deploy on Render

1. Web Service, connect repo `vijender935/modal-gpu-agent`
2. Root Directory: `mcp-server`
3. Build: `pip install -r requirements.txt`
4. Start: `python server.py`
5. Optional env vars:
   - `IMAGE_ENDPOINT`
   - `CODE_ENDPOINT`
   - `PROCESS_ENDPOINT`

## Grok connector

URL: `https://YOUR-RENDER-URL/mcp`

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
