# Security Policy

## Supported deployment posture

The Modal endpoints require a bearer token stored in the Modal secret `modal-endpoint-auth`. The Render MCP gateway requires its own `MCP_GATEWAY_TOKEN` and forwards the endpoint token only to the configured Modal web functions. Do not place either token in source control, URLs, logs, or issue comments.

Google Drive credentials must remain in the Modal secret `google-drive`. OAuth user credentials are preferred for My Drive folders; service-account credentials are supported for Shared Drives. Never commit `token.json`, service-account JSON, or client secrets.

## Arbitrary code execution

General-purpose Python is available only through the separate `run_python` sandbox endpoint. The sandbox uses a single-use Modal container with blocked network access, restricted Modal access, no Drive/OAuth secrets, bounded CPU/GPU time, bounded output and files, and a prebuilt package set. Dynamic package installation is disabled. These controls reduce risk but do not make arbitrary code harmless; keep the endpoint authenticated, rate-limited, and monitored.

## Reporting

Please report suspected vulnerabilities privately to the repository owner rather than opening a public issue with credentials, tokens, or exploit details. Revoke any exposed token immediately and rotate the corresponding Modal or Render secret.
