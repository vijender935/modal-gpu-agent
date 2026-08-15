# MCP Server for Modal GPU Agent

Control Modal GPU functions from Grok.

## Tools

- `generate_image` – Flux image generation with an MCP image artifact response
- `check_gpu` – CUDA / GPU health check without user-provided code
- `run_python` – general-purpose Python in an isolated, network-blocked CPU/GPU sandbox
- `start_drive_processing` – queue a long-running Drive batch and return a job ID
- `get_drive_processing_status` – poll an asynchronous Drive job by ID
- `process_images_from_drive` – synchronous YOLO smart-crop for smaller batches

`run_python` uses packages already installed in the Modal image; dynamic package installation is intentionally disabled. It accepts a maximum 32,000-character program and a timeout from 1 to 120 seconds. The sandbox receives no Drive/OAuth credentials, blocks network access, uses a single-use container, and returns bounded stdout/stderr plus at most ten generated files. GPU and CPU sandbox concurrency is capped at two containers each.

For larger Drive batches, call `start_drive_processing` and then poll `get_drive_processing_status` with the returned job ID. The synchronous tool remains useful for small batches. Modal responses include a request ID, which should be retained when investigating logs or failures.

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
   - `SANDBOX_ENDPOINT`
   - `PROCESS_ENDPOINT`
   - `ASYNC_PROCESS_ENDPOINT`
   - `STATUS_ENDPOINT`

Health check: `GET https://YOUR-RENDER-URL/health`. The response reports whether required gateway tokens and endpoint URLs are configured without revealing secret values. For a pre-merge live test, use temporary staging tokens and run `python scripts/staging_smoke_test.py` with the Modal endpoint URLs; the script checks health, GPU, sandbox, image generation, and optional async Drive submission. For production, rotate `MCP_GATEWAY_TOKEN` and `MODAL_ENDPOINT_TOKEN` independently, update both matching deployment locations, and verify wrong-token rejection before enabling client traffic.

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
