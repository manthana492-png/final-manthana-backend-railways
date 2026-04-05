# Cloud inference configuration

## `cloud_inference.yaml`

Single declarative map of **logical roles** to **OpenRouter** models (and optional fallbacks). The runtime secret is `OPENROUTER_API_KEY` in your environment or secrets manager — never commit it into this file.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | Bearer token for `https://openrouter.ai/api/v1` |
| `CLOUD_INFERENCE_CONFIG_PATH` | Absolute path to this YAML (default: `config/cloud_inference.yaml` relative to repo root in dev; `/app/config/cloud_inference.yaml` in Oracle-2 Docker images) |
| `OPENROUTER_HTTP_REFERER` | Optional; forwarded as `HTTP-Referer` if set (OpenRouter attribution) |
| `OPENROUTER_APP_TITLE` | Optional; forwarded as `X-Title` if set |

### Python loader

The [`packages/manthana-inference`](../packages/manthana-inference) package exposes `load_cloud_inference_config`, `resolve_role`, and OpenAI-SDK-compatible client builders for sync/async chat completions against OpenRouter.

### Phases

- **Phase 0:** This file and the package exist as the SSOT.
- **Phase 1:** Services read this file and call OpenRouter by role.
- **Phase 2:** Legacy per-vendor LLM env vars are removed from deployment.

When you finalize orchestration, update only `roles.*.model` and `fallback_models` here (and redeploy).
