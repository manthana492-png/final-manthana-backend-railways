# Manthana Radio Frontend (Production UI)

This Next.js app is the **canonical production UI** for Manthana Radiology public SaaS.

- Use this codebase for customer-facing deployments, release gates, and regression testing.
- Gateway integration and modality routing are maintained here first (`lib/api.ts`, `lib/constants.ts`).

## Local development

```bash
npm install
npm run dev
```

Configure environment variables:

- `NEXT_PUBLIC_GATEWAY_URL` — browser → gateway (JWT from `LoginGate`).
- `MANTHANA_GATEWAY_URL` / `MANTHANA_DEV_TOKEN` — optional server proxy at `/api/*` (same-origin BFF).

See `.env.local` for examples.
