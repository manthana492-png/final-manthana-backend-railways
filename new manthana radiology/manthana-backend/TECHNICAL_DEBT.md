# Manthana Backend — Known Technical Debt

## TD-001: In-Memory Session Store

**Risk:** Data loss on restart; incorrect counts on multi-replica.

**Fix when:** Railway scales to more than one gateway replica.

**Implementation:** Redis-backed SessionStore (Redis client already in gateway requirements).

## TD-002: In-Memory Rate Limiter

**Risk:** Per-replica limits; a user with two replicas gets twice the limit.

**Fix when:** Same trigger as TD-001.

**Implementation:** Redis-backed rate limiter using the same Redis client.

## TD-003: Audit Log Durability

**Risk:** `audit.log` is lost on Railway redeploy (ephemeral filesystem).

**Fix when:** Before DPDP Act enforcement (May 2027) or at roughly 1000 active users.

**Implementation:** Ship logs to Logtail, Datadog, or S3 via a log forwarder.

## TD-004: Scanned PDF OCR

**Risk:** PDFs without a text layer (scanned documents) return a fallback message only.

**Fix when:** User feedback shows this is frequent.

**Implementation:** Add Tesseract or cloud OCR fallback in `gateway/dicom_converter.py` `extract_pdf_text()`.

## TD-005: NIM X-Ray Interrogator

**Risk:** Hosted NIM catalog slugs change; `interrogator_xray_nim` may fail until verified.

**Fix when:** Slug confirmed via `GET https://integrate.api.nvidia.com/v1/models` with `NVIDIA_NIM_API_KEY`.

**Implementation:** Set `NVIDIA_NIM_API_KEY` and confirm the slug in `orch_chains.interrogator_xray` / `interrogator_xray_nim` against NIM `GET /v1/models`.

## TD-006: Multi-Slice DICOM Series

**Risk:** DICOM converter picks a single frame (middle slice heuristic).

**Fix when:** Phase C launch or strong user demand.

**Implementation:** Accept optional slice index on `/ai/interrogate` and expose in the UI.

## TD-007: Chunked Upload Body Size

**Risk:** `MAX_AI_REQUEST_BYTES` middleware only checks `Content-Length`; chunked requests without length are not capped in the gateway.

**Fix when:** Before accepting very large uploads from untrusted clients.

**Implementation:** Enforce limits at reverse proxy (Railway/nginx) or stream-read with a hard cap in a custom Starlette handler.
