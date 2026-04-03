# Manthana Radio Frontend (Production UI)

This Next.js app is the **canonical production UI** for Manthana Radiology public SaaS.

- Use this codebase for customer-facing deployments, release gates, and regression testing.
- Gateway integration and modality routing are maintained here first (`lib/api.ts`, `lib/constants.ts`).

## Non-production duplicate

`new manthana radiology/manthana-scan` is **not** the production UI. It exists for internal or legacy development only; do not ship it to end users without an explicit product decision.

## Local development

```bash
npm install
npm run dev
```

Configure `NEXT_PUBLIC_*` gateway URLs per your environment (see `.env.example` if present).
