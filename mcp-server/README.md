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

Keys:
- `GOOGLE_SERVICE_ACCOUNT_JSON` – full service account JSON
- `INPUT_FOLDER_ID` – Drive folder ID for input images
- `OUTPUT_FOLDER_ID` – Drive folder ID for output images
