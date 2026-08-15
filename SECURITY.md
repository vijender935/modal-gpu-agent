# Security Policy

## Supported deployment posture

The Modal endpoints require a bearer token stored in the Modal secret `modal-endpoint-auth`. The Render MCP gateway requires its own `MCP_GATEWAY_TOKEN` and forwards the endpoint token only to the configured Modal web functions. Do not place either token in source control, URLs, logs, or issue comments.

Google Drive credentials must remain in the Modal secret `google-drive`. OAuth user credentials are preferred for My Drive folders; service-account credentials are supported for Shared Drives. Never commit `token.json`, service-account JSON, or client secrets.

## Arbitrary code execution

The production application does not expose arbitrary Python execution. A future code runner must be deployed as a separate sandbox with an isolated identity, no application secrets, restricted network access, an explicit package allowlist, resource quotas, and audit logging.

## Reporting

Please report suspected vulnerabilities privately to the repository owner rather than opening a public issue with credentials, tokens, or exploit details. Revoke any exposed token immediately and rotate the corresponding Modal or Render secret.
