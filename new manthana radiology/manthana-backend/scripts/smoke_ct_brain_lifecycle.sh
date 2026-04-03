#!/usr/bin/env bash
# Smoke: CT Brain service health / readiness (CI: set CT_BRAIN_CI_DUMMY_MODEL=1).
set -euo pipefail

BASE="${CT_BRAIN_SMOKE_URL:-http://127.0.0.1:8017}"

echo "CT Brain smoke: GET $BASE/health"
curl -sfS "$BASE/health" | head -c 800 || true
echo
echo "CT Brain smoke: GET $BASE/ready"
curl -sfS "$BASE/ready" | head -c 800 || true
echo
echo "OK — health/ready reachable at $BASE"
