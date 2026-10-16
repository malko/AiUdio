# Security & deployment

## Authentication

Disabled by default (`auth.enabled: false`). To enable it:

```bash
AUTH_ENABLED=true
AUTH_USERNAME=admin
AUTH_PASSWORD='strong-password'
AUTH_API_KEYS='machine-key-1,machine-key-2'
```

| Usage | Method |
|---|---|
| Web UI | **HTTP Basic** (browser prompt) |
| Machine clients / scripts | `Authorization: Bearer <key>` or `X-API-Key: <key>` |
| WebSocket (browser) | API key as a parameter: `?api_key=<key>` |

- `/health` (and `/ready`) stay public (`auth.exempt_paths`).
- Comparisons use `secrets.compare_digest` (constant time).
- **Important**: a browser cannot send an `Authorization` header over a WebSocket; for a
  protected UI over WS, configure **at least one API key** (the "Client API key" field in the
  Settings tab, stored in `localStorage`).

## LAN deployment recommendations

- Only publish the gateway port (`8004`). The STT port (`8080`) does not need to be publicly
  exposed: the UI goes through the proxies `/v1/audio/transcriptions` and `/ws/stt`.
- Enable authentication as soon as the network is not fully trusted.
- Restrict CORS origins (`server.cors_origins`) in production.
- Only upload cloning audio with the speaker's consent.

## Internet exposure

Do not expose it directly. Put a **TLS reverse proxy** (Caddy, Nginx, Traefik) in front of the
gateway. Minimal Caddy example:

```
speech.example.org {
    reverse_proxy localhost:8004
}
```

Caddy handles certificates automatically and can add HTTP Basic if desired. WebSocket is
proxied automatically. Keep the application-level authentication as well.

## Secrets

- Never commit `.env` (git-ignored).
- `AUTH_PASSWORD`, `AUTH_API_KEYS`, `HF_TOKEN`, `LLM_API_KEY`: provide them through
  environment variables / a Docker secrets manager.
- API keys should be long random strings (`openssl rand -hex 32`).
