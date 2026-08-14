# MCP Server for Modal GPU Agent

This lets you control the Modal GPU functions from Grok using natural language.

## Deploy on Render

1. Create a new **Web Service** on Render
2. Connect the repo `vijender935/modal-gpu-agent`
3. Root Directory: `mcp-server`
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `python server.py`
6. Create Web Service

After deploy, copy the Render URL (e.g. https://modal-mcp.onrender.com)

## Add to Grok

1. Go to https://grok.com/connectors
2. Click **New Connector** → **Custom**
3. Paste: `https://YOUR-RENDER-URL/mcp`
4. Save

Now you can say in Grok:
- "Generate an image of a cyberpunk city"
- "Run this code on GPU: print(torch.cuda.is_available())"
