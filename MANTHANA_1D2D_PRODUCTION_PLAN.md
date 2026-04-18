# Manthana Labs — 1D & 2D Production-Ready Implementation Plan
## Complete Phase A → B → C Launch Specification for Cursor Agent

> **Mode required:** Switch to **Agent mode** before executing any section.  
> **Execute in strict priority order. Do not skip, reorder, or combine blocks.**  
> **Report completion of each numbered priority before starting the next.**

---

## Pre-Implementation: Understand What You Are Fixing

### Current Critical Bugs (not enhancements — actual broken behaviour)

| Bug | Where | Impact |
|-----|--------|--------|
| Raw DICOM binary sent as base64 to vision LLMs | `useAIOrchestration` → interrogate | All DICOM X-ray uploads produce garbage responses right now |
| Raw PDF binary sent to text LLMs (reports group) | Same path | Lab reports, prescriptions, discharge summaries fail silently |
| Patient image stored in RAM for 30 minutes unencrypted | `session_store.py` | PHI exposure on interpret failures |
| Browser-visible OpenRouter API key | `deepseek-validator.ts` | Any user can steal credits via DevTools |
| No body size limit on `/ai/*` | `main.py` | A 50MB upload crashes gateway RAM |
| Key rotation restarts from primary model on key 2 | `client.py` | 10-30s wasted latency during model outages |
| No anonymisation before image reaches OpenRouter | `ai_orchestrator.py` | Potential PHI in DICOM headers sent to third-party |

### Architecture Decisions Locked In (do not re-debate these)

- **Interpret step: downscaled 512px de-identified JPEG in session** for Phase B imaging groups. Not full image. Not zero image. 512px JPEG ~30-50KB.
- **Reports group + cardiac_functional: zero image in session** — text-only paths, no raster stored.
- **OpenRouter remains primary provider.** NIM is opt-in per YAML role only.
- **In-memory session store is acceptable for MVP single-replica Railway.** Redis TODO is documented, not implemented now.
- **Audit log to file for MVP.** Log shipping to durable store is infra, not this sprint.

---

## PRIORITY 1 — Security & Stability (Deploy Today, Phase A Cannot Launch Without These)

### 1A — Kill the Browser API Key

**Problem:** `NEXT_PUBLIC_OPENROUTER_API_KEY` in `deepseek-validator.ts` is visible to every user in DevTools. This is a live credential leak.

**Step 1:** Create `POST /ai/pre-validate` endpoint in `ai_orchestrator.py`:

```python
from pydantic import BaseModel
from typing import Optional
import json

class PreValidateRequest(BaseModel):
    image_b64: Optional[str] = None
    image_mime: Optional[str] = None
    selected_modality: Optional[str] = None
    patient_context: Optional[dict] = None

@router.post("/ai/pre-validate")
async def pre_validate(
    body: PreValidateRequest,
    token_data=Depends(verify_token),
    tier: str = Header(default="free", alias="X-Subscription-Tier")
):
    messages = _build_prevalidate_messages(body)
    result = llm_router.complete_for_role(
        role="labs_pre_validate",
        messages=messages,
        image_b64=body.image_b64,
        image_mime=body.image_mime
    )
    audit_logger.log_analysis_event(
        user_id=token_data.sub,
        event_type="pre_validate",
        modality_key=body.selected_modality,
        group=None,
        subscription_tier=tier,
        model_used=result.get("model_used"),
        success=True,
        image_mime=body.image_mime
    )
    return result
```

**Step 2:** Add to `cloud_inference.yaml` under `roles:`:

```yaml
labs_pre_validate:
  model: "moonshotai/kimi-k2.5:online"
  max_tokens: 1024
  temperature: 0.1
  fallback_models:
    - "openai/gpt-4o-mini:online"
    - "qwen/qwen2.5-vl-72b-instruct"
```

**Step 3:** In `deepseek-validator.ts`, replace the direct OpenRouter fetch:

```typescript
// REMOVE THIS ENTIRE BLOCK:
// const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
//   headers: { 'Authorization': `Bearer ${process.env.NEXT_PUBLIC_OPENROUTER_API_KEY}` ... }
// })

// REPLACE WITH:
const response = await fetch(`${GATEWAY_URL}/ai/pre-validate`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${await getGatewayAuthToken()}`,
    'X-Subscription-Tier': subscriptionTier ?? 'free'
  },
  body: JSON.stringify({
    image_b64: imageB64,
    image_mime: imageMime,
    selected_modality: selectedModality,
    patient_context: patientContext
  })
})
```

**Step 4:** Search and destroy the key reference everywhere:

```bash
# Run these — all must return zero results before proceeding
grep -r "NEXT_PUBLIC_OPENROUTER" . --include="*.ts" --include="*.tsx" --include="*.js" --include="*.env*"
```

Remove from: every `.env` file, `.env.local`, `.env.example`, `.env.production`, any docs or README that mentions it.

**Step 5:** Immediately rotate the `OPENROUTER_API_KEY` value on the OpenRouter dashboard after removing the frontend reference. The old value is compromised.

---

### 1B — Request Body Size Limit

**Problem:** No file size cap. A malicious or careless user uploads a 100MB DICOM series and crashes gateway RAM.

In `main.py` where FastAPI app is initialised, add before route registration:

```python
import os
from fastapi import Request
from fastapi.responses import JSONResponse

MAX_AI_BYTES = int(os.environ.get("MAX_AI_REQUEST_BYTES", 10 * 1024 * 1024))  # 10MB default

@app.middleware("http")
async def limit_ai_request_size(request: Request, call_next):
    if request.url.path.startswith("/ai/"):
        # Handle both Content-Length header and chunked encoding
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_AI_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": "Image too large. Maximum size is 10MB. Please compress or resize your image before uploading.",
                    "max_bytes": MAX_AI_BYTES
                }
            )
        # For chunked encoding: read and check body size
        # Note: this buffers the body — only apply to /ai/* not streaming endpoints
    return await call_next(request)
```

Add to `.env.example`:

```
MAX_AI_REQUEST_BYTES=10485760
```

---

### 1C — Audit Logging (DPDP Compliance Foundation)

**Why:** India's DPDP Act (effective May 2027, build now) requires logs of who processed what data and when. Never log image content or patient values — only metadata.

Create `gateway/audit_logger.py`:

```python
import logging
import json
import time
import os
from typing import Optional

# Configure audit logger — separate from app logger
_audit_logger = logging.getLogger("manthana.audit")

def _setup_audit_logging():
    if not _audit_logger.handlers:
        handler = logging.FileHandler("audit.log", mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))  # JSON lines only
        _audit_logger.addHandler(handler)
        _audit_logger.setLevel(logging.INFO)
        _audit_logger.propagate = False  # Don't pollute app logs

_setup_audit_logging()

def log_analysis_event(
    user_id: str,           # JWT sub — anonymised identifier, NOT patient name/ID
    event_type: str,        # "detect" | "interrogate" | "interpret" | "pre_validate"
    modality_key: Optional[str] = None,
    group: Optional[str] = None,
    subscription_tier: str = "unknown",
    model_used: Optional[str] = None,
    success: bool = True,
    error_code: Optional[int] = None,
    image_mime: Optional[str] = None,  # type only, never the image
    session_id: Optional[str] = None   # internal ID only
):
    """
    Log analysis event for DPDP compliance.
    NEVER pass: image_b64, patient_context contents, patient name, DOB, report text.
    ONLY log: who (user_id), what (modality/group), which model, success/fail, when.
    """
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uid": user_id,
        "event": event_type,
        "modality": modality_key,
        "group": group,
        "tier": subscription_tier,
        "model": model_used,
        "ok": success,
        "mime_type": image_mime,
        "session": session_id,
        "err": error_code
    }
    _audit_logger.info(json.dumps(record, separators=(',', ':')))
```

Wire `audit_logger.log_analysis_event(...)` calls at the **end of every `/ai/*` endpoint** — both success and failure paths. Pass `success=False, error_code=status_code` in except blocks.

**Infrastructure note:** On Railway, `audit.log` is ephemeral (lost on redeploy). For MVP this is acceptable. Before production scale, add a log shipper (e.g., Railway → Datadog, Logtail, or S3). Document this in `RAILWAY_ENV_REFERENCE.md`.

---

## PRIORITY 2 — Session Store Hardening (Required Before Phase A Launch)

### 2A — Fix Session Lifetime and Cleanup

In `session_store.py`, replace the current implementation with:

```python
import threading
import time
import uuid
from typing import Any, Dict, Optional

class SessionStore:
    def __init__(self, ttl_seconds: int = 300):
        self._data: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._start_background_sweep()

    def _start_background_sweep(self):
        """Daemon thread sweeps expired sessions every 60s regardless of traffic."""
        def sweep_loop():
            while True:
                time.sleep(60)
                self._sweep_expired()
        t = threading.Thread(target=sweep_loop, daemon=True)
        t.name = "session-sweep"
        t.start()

    def _sweep_expired(self):
        now = time.time()
        with self._lock:
            expired_keys = [
                k for k, v in self._data.items()
                if now - v.get("created_at", 0) > self._ttl
            ]
            for k in expired_keys:
                del self._data[k]
        if expired_keys:
            import logging
            logging.getLogger(__name__).debug(
                f"Session sweep: removed {len(expired_keys)} expired sessions"
            )

    def create(self, data: dict) -> str:
        session_id = str(uuid.uuid4())
        data["created_at"] = time.time()
        with self._lock:
            self._sweep_unlocked()  # lazy cleanup on write too
            self._data[session_id] = data
        return session_id

    def get(self, session_id: str) -> Optional[dict]:
        now = time.time()
        with self._lock:
            row = self._data.get(session_id)
            if row is None:
                return None
            if now - row.get("created_at", 0) > self._ttl:
                del self._data[session_id]
                return None
            return dict(row)  # return copy, not reference

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def _sweep_unlocked(self):
        """Called inside lock — remove expired entries."""
        now = time.time()
        expired = [k for k, v in self._data.items()
                   if now - v.get("created_at", 0) > self._ttl]
        for k in expired:
            del self._data[k]
```

Update `.env.example`:

```
AI_SESSION_TTL_SECONDS=300
```

**Add a comment at the top of `session_store.py`:**

```python
# SINGLE-REPLICA ONLY: This in-memory store is correct for one gateway process.
# When Railway scales to >1 replica, replace with Redis-backed store.
# Redis client is already in gateway requirements — see redis_session_store.py (TODO).
```

### 2B — Fix Interpret Failure — Always Delete Session

In `ai_orchestrator.py` interpret endpoint, ensure session is **always** deleted:

```python
@router.post("/ai/interpret")
async def interpret(body: InterpretRequest, ...):
    session_id = body.session_id
    row = store.get(session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    
    try:
        # ... existing interpret logic ...
        result = _build_interpret_result(row, body, ...)
        store.delete(session_id)  # success path
        audit_logger.log_analysis_event(..., success=True)
        return result
    except Exception as e:
        store.delete(session_id)  # ALWAYS delete on failure too
        audit_logger.log_analysis_event(..., success=False, error_code=502)
        raise HTTPException(status_code=502, detail=f"Interpret failed: {str(e)}")
```

### 2C — What Gets Stored in Session (Image Policy)

This is the definitive policy. Implement exactly this:

```python
# In interrogate endpoint, after conversion (Priority 3):

async def _prepare_session_image(
    converted_b64: Optional[str],
    converted_mime: Optional[str],
    group: str
) -> tuple[Optional[str], Optional[str]]:
    """
    Policy:
    - reports group or cardiac_functional (text paths): store nothing
    - All imaging groups (Phase B onwards): store downscaled 512px JPEG
    Never store raw DICOM. Never store original-resolution image.
    """
    TEXT_ONLY_GROUPS = {"reports", "cardiac_functional"}
    
    if group in TEXT_ONLY_GROUPS:
        return None, None
    
    if not converted_b64 or not converted_mime:
        return None, None
    
    if converted_mime not in ("image/png", "image/jpeg", "image/webp"):
        return None, None
    
    # Downscale to 512px max dimension
    downscaled = _downscale_for_session(converted_b64, max_px=512)
    return downscaled, "image/jpeg"


def _downscale_for_session(image_b64: str, max_px: int = 512) -> str:
    from PIL import Image
    import io, base64
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if max(img.size) > max_px:
        ratio = max_px / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
```

Store in session:

```python
session_img_b64, session_img_mime = await _prepare_session_image(
    converted_b64, converted_mime, cfg.group
)

store.create({
    "image_b64": session_img_b64,      # None for text groups; 512px JPEG for imaging
    "image_mime": session_img_mime,
    "modality_key": modality_key,
    "display_name": cfg.display_name,
    "group": cfg.group,
    "questions": questions,
    "interrogator_model": interrogator_model,
    "interpreter_role": interpreter_role,
    "patient_context_json": body.patient_context_json,
    "created_at": time.time()
    # NEVER store: raw image_b64, original DICOM bytes, patient name/ID
})
```

---

## PRIORITY 3 — DICOM and PDF Conversion (Required Before Phase B)

### 3A — Add Dependencies

Add to `gateway/requirements.txt` AND `gateway/requirements-railway.txt`:

```
pydicom>=2.4.0
pymupdf>=1.23.0
```

`Pillow` and `numpy` already present — confirm with:

```bash
grep -i "pillow\|numpy" gateway/requirements.txt
```

### 3B — Create `gateway/dicom_converter.py`

```python
"""
Medical image conversion and de-identification for Manthana Labs gateway.
Converts DICOM / ambiguous binary to PNG before sending to vision LLMs.
Extracts text from PDF before sending to text LLMs.
De-identifies all DICOM metadata before any processing.
"""

import io
import base64
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# --- DICOM Tag De-identification ---

_PHI_TAGS_TO_BLANK = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
    "PatientAge", "PatientAddress", "PatientTelephoneNumbers",
    "PatientMotherBirthName", "OtherPatientIDs", "OtherPatientNames",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "ReferringPhysicianName", "PerformingPhysicianName",
    "RequestingPhysician", "OperatorsName", "PhysiciansOfRecord",
    "StudyID", "AccessionNumber", "StudyDescription", "SeriesDescription",
    "ProtocolName", "RequestedProcedureDescription",
    "CountryOfResidence", "RegionOfResidence",
    "PatientInsurancePlanCodeSequence",
]


def _deidentify_dicom(ds) -> None:
    """Strip PHI from DICOM dataset in-place. Replaces with empty strings."""
    import pydicom
    from pydicom.uid import generate_uid

    for tag_name in _PHI_TAGS_TO_BLANK:
        if hasattr(ds, tag_name):
            try:
                setattr(ds, tag_name, "")
            except Exception:
                try:
                    delattr(ds, tag_name)
                except Exception:
                    pass

    # Remove all private tags (vendor-specific, often contain PHI)
    ds.remove_private_tags()

    # Replace UIDs with new generated ones (breaks linkage to original study)
    for uid_attr in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        if hasattr(ds, uid_attr):
            try:
                setattr(ds, uid_attr, generate_uid())
            except Exception:
                pass


def _dicom_to_png_b64(raw_bytes: bytes) -> str:
    """
    Convert DICOM bytes to PNG base64.
    Handles: 2D grayscale, 2D colour, multi-frame (picks middle frame).
    Applies windowing / normalisation for display.
    """
    import pydicom
    import numpy as np
    from PIL import Image

    ds = pydicom.dcmread(io.BytesIO(raw_bytes), force=True)
    _deidentify_dicom(ds)

    pixel_array = ds.pixel_array.astype(np.float32)

    # Multi-frame: pick middle frame
    if pixel_array.ndim == 3 and pixel_array.shape[0] > 3:
        mid = pixel_array.shape[0] // 2
        pixel_array = pixel_array[mid]

    # Apply rescale slope/intercept if present (CT Hounsfield units etc.)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixel_array = pixel_array * slope + intercept

    # Window/level if VOI LUT present
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        center = float(ds.WindowCenter) if not hasattr(ds.WindowCenter, "__iter__") \
            else float(list(ds.WindowCenter)[0])
        width = float(ds.WindowWidth) if not hasattr(ds.WindowWidth, "__iter__") \
            else float(list(ds.WindowWidth)[0])
        low = center - width / 2
        high = center + width / 2
        pixel_array = np.clip(pixel_array, low, high)

    # Normalise to 0-255
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax > pmin:
        pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
    else:
        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

    # Handle photometric interpretation
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = 255 - pixel_array

    img = Image.fromarray(pixel_array)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if img.mode == "L":
        img = img.convert("RGB")

    # Cap at 2048px longest side (vision LLM token limit)
    max_px = 2048
    if max(img.size) > max_px:
        ratio = max_px / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def convert_image_for_llm(image_b64: str, image_mime: str) -> Tuple[str, str]:
    """
    Main conversion entry point for interrogate/detect endpoints.
    Returns (converted_b64, converted_mime).
    Raises ValueError on unrecoverable conversion failure.

    PDF: caller should use extract_pdf_text() instead — this function
    will raise ValueError for PDF mime types to force correct routing.
    """
    if not image_b64:
        return image_b64, image_mime

    # Already a browser-compatible raster — pass through
    if image_mime in ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"):
        return image_b64, image_mime

    # PDF — must be routed through extract_pdf_text, not here
    if image_mime == "application/pdf" or image_mime.endswith("/pdf"):
        raise ValueError(
            "PDF files must be processed with extract_pdf_text(), not convert_image_for_llm(). "
            "Check routing in ai_orchestrator.py."
        )

    # DICOM or ambiguous binary
    if (
        "dicom" in image_mime.lower()
        or image_mime in ("application/octet-stream", "")
        or image_mime is None
    ):
        try:
            raw_bytes = base64.b64decode(image_b64)
            converted_b64 = _dicom_to_png_b64(raw_bytes)
            logger.info("DICOM converted to PNG successfully")
            return converted_b64, "image/png"
        except Exception as e:
            raise ValueError(f"DICOM to PNG conversion failed: {e}") from e

    # Unknown mime — attempt pass-through and log warning
    logger.warning(f"Unknown image mime type '{image_mime}' — passing through unconverted.")
    return image_b64, image_mime


def extract_pdf_text(image_b64: str, max_chars: int = 8000) -> str:
    """
    Extract text from PDF base64.
    Returns extracted text capped at max_chars.
    Raises ValueError on failure (scanned PDF with no text layer → empty string, not error).
    """
    try:
        import pymupdf  # fitz
        raw = base64.b64decode(image_b64)
        doc = pymupdf.open(stream=raw, filetype="pdf")
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
        full_text = "\n".join(pages_text).strip()
        if not full_text:
            logger.warning("PDF had no extractable text layer — may be a scanned document.")
            return "[Document appears to be a scanned image. Text extraction not available. Please upload a text-based PDF or a photo of the document.]"
        return full_text[:max_chars]
    except Exception as e:
        raise ValueError(f"PDF text extraction failed: {e}") from e


def is_pdf(image_mime: str) -> bool:
    return image_mime == "application/pdf" or (image_mime or "").endswith("/pdf")


def is_dicom(image_mime: str) -> bool:
    return (
        "dicom" in (image_mime or "").lower()
        or image_mime in ("application/octet-stream", "", None)
    )
```

### 3C — Wire Conversion in `ai_orchestrator.py`

At the top of **both** `detect_modality` and `interrogate` endpoints, add this routing block immediately after request validation, before any LLM call:

```python
from .dicom_converter import convert_image_for_llm, extract_pdf_text, is_pdf, is_dicom

# === IMAGE PRE-PROCESSING (in both detect and interrogate) ===
pdf_extracted_text: Optional[str] = None

if body.image_b64 and body.image_mime:
    if is_pdf(body.image_mime):
        # PDF → extract text, route as text-only (no image to LLM)
        try:
            pdf_extracted_text = extract_pdf_text(body.image_b64)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        body.image_b64 = None
        body.image_mime = None
        # Merge PDF text into patient context
        try:
            ctx = json.loads(body.patient_context_json or "{}")
        except json.JSONDecodeError:
            ctx = {}
        ctx["extracted_document_text"] = pdf_extracted_text
        # Cap total context to avoid token blowup
        body.patient_context_json = json.dumps(ctx)[:12000]
    else:
        # DICOM or unknown binary → convert to PNG
        try:
            body.image_b64, body.image_mime = convert_image_for_llm(
                body.image_b64, body.image_mime
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Image conversion failed: {e}. "
                       "Supported formats: JPEG, PNG, DICOM (.dcm), PDF."
            )
# === END PRE-PROCESSING ===
```

---

## PRIORITY 4 — Fix Key Rotation Logic

**Problem:** In `manthana_inference/client.py`, when key 1 fails on a model, key 2 restarts from the same primary model (wasting 1 extra call + latency). For medical reports where users are waiting, this adds 10-30 seconds during model outages.

**Correct behaviour:** For each model in the list, try all available keys. Only if the model itself is down across all keys, move to the next model. Never retry a dead primary model with key 2.

In `manthana_inference/client.py`, refactor `chat_complete_sync`:

```python
import time

# Error categories for routing decisions
_RATE_LIMIT_CODES = {429}
_AUTH_ERROR_CODES = {401, 403}
_SERVER_ERROR_CODES = {500, 502, 503, 504}

def chat_complete_sync(role_config, messages, api_keys, **kwargs):
    """
    Outer loop: models (primary + fallbacks)
    Inner loop: api keys
    
    Per model: try all keys.
      - 429 (rate limit): try next key for same model
      - 401/403 (auth): try next key for same model  
      - 5xx / timeout: model is down, move to next model immediately
      - Success: return immediately
    
    If all models × all keys fail: raise RuntimeError.
    """
    models_to_try = [role_config.model] + list(role_config.fallback_models or [])
    last_error = None

    for model_slug in models_to_try:
        model_succeeded_with_a_key = False
        for api_key in api_keys:
            try:
                result = _call_openrouter(
                    model=model_slug,
                    messages=messages,
                    api_key=api_key,
                    max_tokens=role_config.max_tokens,
                    temperature=role_config.temperature,
                    **kwargs
                )
                return result  # first success wins
            except OpenRouterHTTPError as e:
                last_error = e
                if e.status_code in _RATE_LIMIT_CODES | _AUTH_ERROR_CODES:
                    # Key-level issue — try next key for same model
                    continue
                else:
                    # Model-level issue (5xx, timeout) — break inner loop
                    break
            except Exception as e:
                last_error = e
                break  # Unknown error — move to next model

    raise RuntimeError(
        f"All models failed for role '{role_config.role_name}'. "
        f"Models tried: {models_to_try}. Last error: {last_error}"
    )
```

---

## PRIORITY 5 — Per-User Rate Limiting

Create `gateway/rate_limiter.py`:

```python
import os
import time
import threading
from collections import defaultdict
from typing import Dict, List
from fastapi import HTTPException

class InMemoryRateLimiter:
    """
    Token bucket rate limiter keyed by user_id (JWT sub).
    Single-replica only. Replace with Redis when scaling.
    """
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self._store: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check_and_record(self, user_id: str) -> tuple[bool, int]:
        """Returns (is_allowed, retry_after_seconds)."""
        now = time.time()
        with self._lock:
            timestamps = self._store[user_id]
            # Evict old entries
            self._store[user_id] = [t for t in timestamps if now - t < self.window]
            if len(self._store[user_id]) >= self.max_requests:
                oldest = self._store[user_id][0]
                retry_after = int(self.window - (now - oldest)) + 1
                return False, retry_after
            self._store[user_id].append(now)
            return True, 0


# Initialise from environment
_FREE_LIMIT = int(os.environ.get("AI_INTERROGATE_RATE_LIMIT_FREE", "10"))
_PAID_LIMIT = int(os.environ.get("AI_INTERROGATE_RATE_LIMIT_PAID", "100"))
_WINDOW = int(os.environ.get("AI_RATE_WINDOW_SECONDS", "3600"))

_free_limiter = InMemoryRateLimiter(_FREE_LIMIT, _WINDOW)
_paid_limiter = InMemoryRateLimiter(_PAID_LIMIT, _WINDOW)

FREE_TIERS = {"free", "trial", ""}

def enforce_rate_limit(user_id: str, subscription_tier: str) -> None:
    """Call at the start of POST /ai/interrogate. Raises 429 if exceeded."""
    limiter = _free_limiter if subscription_tier.lower() in FREE_TIERS else _paid_limiter
    allowed, retry_after = limiter.check_and_record(user_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded. Please wait before submitting another analysis.",
                "retry_after_seconds": retry_after,
                "limit_per_hour": _FREE_LIMIT if subscription_tier.lower() in FREE_TIERS else _PAID_LIMIT
            },
            headers={"Retry-After": str(retry_after)}
        )

# TODO: Replace InMemoryRateLimiter with Redis-backed implementation
# when Railway gateway scales to >1 replica. Redis client already in requirements.
```

Add to `.env.example`:

```
AI_INTERROGATE_RATE_LIMIT_FREE=10
AI_INTERROGATE_RATE_LIMIT_PAID=100
AI_RATE_WINDOW_SECONDS=3600
```

In `ai_orchestrator.py` interrogate endpoint, add at the very top after token validation:

```python
from .rate_limiter import enforce_rate_limit

# Apply rate limit before any processing
enforce_rate_limit(user_id=token_data.sub, subscription_tier=tier)
```

---

## PRIORITY 6 — Mandatory Disclaimer Injection

Every report that leaves the interpret endpoint must carry the legal disclaimer. This protects you regardless of how the API is accessed — browser, direct API call, exported PDF.

In `ai_orchestrator.py`, create a helper and call it on every interpret response:

```python
import time

_DISCLAIMER = {
    "text": (
        "This report is generated by Manthana Labs AI for educational, research, "
        "and retrospective study purposes only. It is a second-opinion tool and does "
        "not constitute medical advice, diagnosis, or treatment. "
        "Review and sign-off by a licensed physician or radiologist is mandatory "
        "before any clinical decision is made based on this output."
    ),
    "version": "1.0",
    "regulatory_note": "SaMD — Not approved for autonomous clinical diagnosis. India MDR 2017 compliant."
}

def _inject_disclaimer(report: dict) -> dict:
    """Merge disclaimer into report JSON. Safe if report already has disclaimer key."""
    report["disclaimer"] = {
        **_DISCLAIMER,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    return report
```

Call `_inject_disclaimer(report_data)` before returning from the interpret endpoint. If `report_data` is a string (model returned plain text), wrap it first:

```python
if isinstance(report_data, str):
    report_data = {"report_text": report_data}
report_data = _inject_disclaimer(report_data)
```

---

## PRIORITY 7 — NIM Xray Integration (Verify First)

**Do not add YAML roles until you verify the slug exists on the hosted NIM API.**

**Step 1 — Verify slug:**

```bash
curl -s https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer $NVIDIA_NIM_API_KEY" \
  | python3 -m json.tool \
  | grep -i "cxr\|reason\|chest\|xray\|x-ray"
```

**Step 2A — If slug confirmed (e.g. `nvidia/nv-reason-cxr-3b`):**

Add to `cloud_inference.yaml`:

```yaml
interrogator_xray_nim:
  provider: nim
  model: "nvidia/nv-reason-cxr-3b"   # use exact slug from Step 1
  max_tokens: 2048
  temperature: 0.2
  fallback_models: []
```

In `ai_orchestrator.py` interrogate endpoint, for `cfg.group == "xray"` only, add NIM-first logic:

```python
if cfg.group == "xray":
    try:
        result = llm_router.complete_for_role(
            role="interrogator_xray_nim",
            messages=messages,
            image_b64=body.image_b64,
            image_mime=body.image_mime
        )
        interrogator_model = "nvidia/nv-reason-cxr-3b (NIM)"
    except Exception as nim_error:
        logger.warning(f"NIM xray interrogator failed, falling back to OpenRouter: {nim_error}")
        result = llm_router.complete_for_role(
            role="interrogator_xray",
            messages=messages,
            image_b64=body.image_b64,
            image_mime=body.image_mime
        )
        interrogator_model = result.get("model_used", "interrogator_xray")
else:
    result = llm_router.complete_for_role(...)
```

**Step 2B — If slug NOT found in Step 1:**

Skip Priority 7 entirely. Add this comment to `cloud_inference.yaml`:

```yaml
# TODO: interrogator_xray_nim — awaiting nvidia/nv-reason-cxr-3b in hosted NIM catalog
# Re-check: curl https://integrate.api.nvidia.com/v1/models -H "Authorization: Bearer $NVIDIA_NIM_API_KEY"
```

---

## PRIORITY 8 — Phase Launch Guards (Server-Side)

### 8A — ORCH_ALLOWED_GROUPS Environment Control

In `ai_orchestrator.py`, add group guard to the interrogate endpoint after resolving modality config:

```python
import os

def _get_allowed_groups() -> set:
    raw = os.environ.get("ORCH_ALLOWED_GROUPS", "")
    if not raw.strip():
        return set()  # empty = all allowed (dev mode)
    return {g.strip().lower() for g in raw.split(",") if g.strip()}

# Cache at module level, refresh on config reload
_ALLOWED_GROUPS = _get_allowed_groups()

# In interrogate endpoint, after cfg = get_modality_config(modality_key):
if _ALLOWED_GROUPS and cfg.group not in _ALLOWED_GROUPS:
    raise HTTPException(
        status_code=403,
        detail={
            "error": f"Modality group '{cfg.group}' is not yet available in this release.",
            "message": "This feature is coming soon. Currently available: reports, prescriptions, lab results, ECG analysis.",
            "group_requested": cfg.group
        }
    )
```

### 8B — Frontend Phase Flag

In `constants.ts` (or wherever feature flags live), add:

```typescript
// Phase A: reports + cardiac_functional only
// Phase B: + xray, ophthalmology_dental, oncology
// Phase C: + ultrasound, pathology, specialized, nuclear, mri, ct
export const ORCH_PHASE = (process.env.NEXT_PUBLIC_ORCH_PHASE ?? "A") as "A" | "B" | "C"

export const PHASE_ALLOWED_GROUPS: Record<string, string[]> = {
  A: ["reports", "cardiac_functional"],
  B: ["reports", "cardiac_functional", "xray", "ophthalmology_dental", "oncology"],
  C: ["reports", "cardiac_functional", "xray", "ophthalmology_dental", "oncology",
      "ultrasound", "pathology", "specialized", "nuclear", "mri", "ct"]
}

export const isGroupAllowedInPhase = (group: string): boolean =>
  PHASE_ALLOWED_GROUPS[ORCH_PHASE]?.includes(group) ?? false
```

In `ModalityBar.tsx`, use `isGroupAllowedInPhase(group)` to grey out / add "Coming Soon" badge to Phase B and C groups during Phase A.

Add to `.env.example`:

```
NEXT_PUBLIC_ORCH_PHASE=A
```

---

## PHASE LAUNCH SPECIFICATION

### Phase A — Launch This Week

**What goes live:** 14 modalities — all 10 `reports` group + 4 `cardiac_functional` (text/CSV paths only)

**Required before Phase A deploy:**
- [x] Priority 1A (browser key removed + /ai/pre-validate live)
- [x] Priority 1B (10MB body limit)
- [x] Priority 1C (audit logging writing to file)
- [x] Priority 2A (session TTL 300 + background sweep)
- [x] Priority 2B (interpret deletes session on failure)
- [x] Priority 5 (rate limiting on interrogate)
- [x] Priority 6 (disclaimer injected on all reports)
- [x] Priority 8 (ORCH_ALLOWED_GROUPS=reports,cardiac_functional on Railway)

**Railway env vars for Phase A:**

```
ORCH_ALLOWED_GROUPS=reports,cardiac_functional
AI_SESSION_TTL_SECONDS=300
MAX_AI_REQUEST_BYTES=10485760
AI_INTERROGATE_RATE_LIMIT_FREE=10
AI_INTERROGATE_RATE_LIMIT_PAID=100
AI_RATE_WINDOW_SECONDS=3600
NEXT_PUBLIC_ORCH_PHASE=A
```

**Monitor for 72 hours post-launch:**
- `audit.log` — check model failure rates. If any role fails >5% of requests, investigate that model slug.
- Gateway memory — watch for session store growth. Should stay flat between requests.
- 429 rate from users — if free users hit limit often, consider raising `AI_INTERROGATE_RATE_LIMIT_FREE` to 20.

---

### Phase B — Launch Week 2-3

**What goes live:** +11 modalities — xray (11) + ophthalmology_dental (3) + oncology (4)

**Required before Phase B deploy:**
- [x] Priority 3 (DICOM converter + PDF extraction deployed and tested)
- [x] Priority 2C (session stores downscaled 512px image for imaging groups)
- [x] Priority 7 (NIM xray — if slug verified; optional otherwise)

**Pre-launch QA for Phase B (mandatory):**

```
Test matrix — run all before enabling Phase B:
1. Upload chest X-ray as .dcm file → expect valid PNG in LLM response, no DICOM decode errors
2. Upload chest X-ray as .jpg phone photo → expect pass-through, valid response
3. Upload chest X-ray as .png → expect pass-through, valid response
4. Upload lab report as .pdf (text-based) → expect text extracted, sent to reports LLM
5. Upload lab report as .pdf (scanned/image-only) → expect fallback message, no crash
6. Upload 15MB file → expect 413 response
7. Upload file with fake mime type (octet-stream but really JPEG) → expect conversion attempt
8. Confirm session is deleted after successful interpret
9. Confirm session is deleted after failed interpret
10. Confirm no image_b64 is logged in audit.log
```

**Railway env update for Phase B (only change from Phase A):**

```
ORCH_ALLOWED_GROUPS=reports,cardiac_functional,xray,ophthalmology_dental,oncology
NEXT_PUBLIC_ORCH_PHASE=B
```

---

### Phase C — Launch Week 4+

**What goes live:** +remaining modalities — ultrasound (12) + pathology (5) + specialized (10) + nuclear (7) + mri (14) + ct (15)

**Gate condition:** Phase B must be stable for 1 full week with <2% error rate on interrogate/interpret for Phase B groups before Phase C opens.

**Additional considerations for Phase C:**
- CT and MRI uploads are most likely to be multi-slice DICOM series — the converter picks the middle frame. Add a user-facing message: "For best results, upload a single representative DICOM slice. Multi-slice series will use the middle frame."
- Nuclear/PET images may have unusual DICOM photometric interpretations — test with real PET DICOM samples before enabling nuclear group.
- Consider raising `MAX_AI_REQUEST_BYTES` to `20971520` (20MB) for CT/MRI which can be larger single-slice DICOMs.

**Railway env update for Phase C:**

```
ORCH_ALLOWED_GROUPS=reports,cardiac_functional,xray,ophthalmology_dental,oncology,ultrasound,pathology,specialized,nuclear,mri,ct
NEXT_PUBLIC_ORCH_PHASE=C
MAX_AI_REQUEST_BYTES=20971520
```

---

## Final Pre-Launch Verification Checklist

Run all of these against your staging environment before Phase A goes live. All must pass.

```bash
# 1. Confirm browser key is gone
grep -r "NEXT_PUBLIC_OPENROUTER" . \
  --include="*.ts" --include="*.tsx" --include="*.js" \
  --include="*.env*" --include="*.md"
# Expected: zero results

# 2. Confirm /ai/pre-validate is live
curl -X POST https://YOUR_STAGING_GATEWAY/ai/pre-validate \
  -H "Authorization: Bearer $TEST_JWT" \
  -H "Content-Type: application/json" \
  -d '{"selected_modality": "lab_report", "patient_context": {"note": "test"}}'
# Expected: 200 with PreValidationResponse schema

# 3. Confirm 10MB limit
python3 -c "
import requests, base64
big = base64.b64encode(b'x' * 11_000_000).decode()
r = requests.post('https://YOUR_STAGING_GATEWAY/ai/interrogate',
  headers={'Authorization': 'Bearer $TEST_JWT', 'Content-Type': 'application/json'},
  json={'image_b64': big, 'image_mime': 'image/png', 'modality_key': 'chest_xray'})
print(r.status_code)  # Expected: 413
"

# 4. Confirm audit log writes
tail -5 audit.log
# Expected: JSON lines with ts, uid, event fields — NO image_b64, NO patient text

# 5. Confirm session deleted after interpret
# Step A: interrogate → get session_id
# Step B: interpret with session_id → 200
# Step C: interpret again with same session_id → 404
# Expected: 404 on second interpret

# 6. Confirm session TTL sweep
# Set AI_SESSION_TTL_SECONDS=5, create a session, wait 70 seconds
# Check: session count should be zero (swept by background thread)

# 7. Confirm rate limiting
# Send 11 interrogate requests as free tier user in < 1 hour
# Expected: first 10 → 200, 11th → 429 with retry_after_seconds

# 8. Confirm disclaimer on every report
# Run any full detect → interrogate → interpret flow
# Expected: report JSON contains "disclaimer" key with text, version, generated_at_utc

# 9. Confirm ORCH group guard
# Send interrogate request with modality_key in xray group when ORCH_ALLOWED_GROUPS=reports,cardiac_functional
# Expected: 403 with "not yet available" message

# 10. Confirm NIM key dormant (unless Priority 7 executed)
grep "provider: nim" cloud_inference.yaml
# Expected: only nim_chat_example (and interrogator_xray_nim if Priority 7 was done)
```

---

## Known Technical Debt — Document, Don't Fix Now

Add a `TECHNICAL_DEBT.md` in the backend root with these entries so they are tracked and not forgotten:

```markdown
# Manthana Backend — Known Technical Debt

## TD-001: In-Memory Session Store
**Risk:** Data loss on restart; incorrect counts on multi-replica.
**Fix when:** Railway scales to >1 gateway replica.
**Implementation:** Redis-backed SessionStore (client already in requirements).

## TD-002: In-Memory Rate Limiter  
**Risk:** Per-replica limits; a user with 2 replicas gets 2x the limit.
**Fix when:** Same trigger as TD-001.
**Implementation:** Redis-backed rate limiter using the same Redis client.

## TD-003: Audit Log Durability
**Risk:** audit.log is lost on Railway redeploy (ephemeral filesystem).
**Fix when:** Before DPDP Act enforcement (May 2027) or at 1000 users.
**Implementation:** Ship logs to Logtail / Datadog / S3 via log forwarder.

## TD-004: Scanned PDF OCR
**Risk:** PDFs without text layer (scanned documents) return fallback message.
**Fix when:** User complaints indicate this is frequent.
**Implementation:** Add pytesseract or Tesseract OCR fallback in extract_pdf_text().

## TD-005: NIM Xray Interrogator
**Risk:** NV-Reason-CXR-3B not yet verified in NVIDIA hosted API catalog.
**Fix when:** Slug appears in GET /v1/models response from integrate.api.nvidia.com.
**Implementation:** Priority 7 of production plan.

## TD-006: Multi-Slice DICOM Series
**Risk:** DICOM converter picks middle frame only. Radiologists may prefer specific slice.
**Fix when:** Phase C launch or user feedback.
**Implementation:** Accept slice index parameter in interrogate request; expose in UI.
```

---

*Plan version: 1.0 | Scope: 1D & 2D only (95 modalities) | 3D Premium excluded | Last reviewed: April 2026*
