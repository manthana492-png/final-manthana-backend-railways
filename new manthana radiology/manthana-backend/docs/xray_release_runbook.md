# X-Ray Release Runbook

## Scope

This runbook governs production rollout of `xray` in gateway + `body_xray` service.

## Required Environment Pins

- `KIMI_MODEL=kimi-k2.5`
- `XRAY_REQUIRE_KIMI_NARRATIVE=1`
- `XRAY_TRIAGE_POLICY=always_deep` (or `threshold` with approved threshold)
- `TRIAGE_ABNORMALITY_THRESHOLD=0.30` (only for threshold policy)

## Pre-Release Gates

- Unit tests pass:
  - `tests/test_msk_unavailable.py`
  - `tests/test_triage.py`
  - `tests/test_xray_kimi_policy.py`
- Smoke tests pass for chest + non-chest:
  - `scripts/smoke_test_cxr.py <chest_image>`
  - Upload abdomen/spine/extremity/skull samples through gateway `/analyze`.
- Confirm zero `status=unavailable` for supported xray regions.
- Confirm `models_used` includes only Kimi narrative provider for narrative step.

## Canary Rollout (Staging -> Prod)

1. Deploy with pinned env values above.
2. Run synthetic canary set:
   - chest normal
   - chest high-risk
   - abdomen
   - spine
   - extremity
   - skull
3. Verify:
   - response schema (`findings`, `pathology_scores`, `structures`, `models_used`, `detected_region`)
   - no 5xx spikes
   - acceptable latency on cold and warm requests.
4. Observe 24-72h before full traffic.

## Monitoring Checklist

- HTTP success/error rates on `/analyze`.
- P50/P95 latency split by `analysis_depth`.
- Cold-start and warm-start timing from xray runtime fields.
- Kimi failure rate and 503 count.

## Rollback

- Fast rollback: set `XRAY_TRIAGE_POLICY=always_deep` and route only chest uploads operationally.
- Emergency rollback: scale down `body_xray` deployment and revert to last known stable image tag.
- Post-rollback: capture failing payload samples and container logs for root-cause analysis.
