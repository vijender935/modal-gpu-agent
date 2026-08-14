# Modal GPU Agent

Full automatic GPU compute for:
- Image Generation (Flux)
- General heavy Python code on GPU
- Ready to be wrapped as MCP server for Grok

## How it deploys (Phone friendly)

This repo uses **GitHub Actions** to deploy to Modal automatically.

### Setup (one time)

1. Go to [modal.com](https://modal.com) → create account
2. Create a token: Settings → API Tokens → New Token
3. Copy **Token ID** and **Token Secret**
4. In this GitHub repo → Settings → Secrets and variables → Actions
5. Add two secrets:
   - `MODAL_TOKEN_ID`
   - `MODAL_TOKEN_SECRET`

### Deploy

Just push to `main` branch.  
GitHub Actions will automatically run `modal deploy`.

You can also manually trigger from the **Actions** tab.

## After deploy

Modal will create public endpoints.  
Check the Actions logs or Modal dashboard for the URLs.

## Test

**Image Generation:**
```bash
curl -X POST https://YOUR-ENDPOINT/generate_image_endpoint \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cyberpunk city at night, highly detailed"}' \
  --output result.png
```

## Next

After this works, we will:
1. Add more image/video tools
2. Create MCP server so you can control it from Grok with natural language
