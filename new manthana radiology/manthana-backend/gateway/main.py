import sys, os
import sys as _sys
_GATEWAY_DIR  = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_GATEWAY_DIR)
sys.path.insert(0, _BACKEND_ROOT)
sys.path.insert(0, os.path.join(_BACKEND_ROOT, "shared"))
sys.path.insert(0, _GATEWAY_DIR)

"""
Manthana — API Gateway
Central entry point. Routes requests by modality → individual services.
Handles JWT authentication and file upload.
"""

import asyncio
import uuid
import time
import shutil
import httpx
import logging
import zipfile
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Body
from fastapi.responses import Response, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from admin import admin_router
from consent import router as consent_router
from auth_routes import router as auth_router
from ai_orchestrator import router as ai_orchestrator_router
from auth import verify_token, JWT_SECRET
from router import ALIASES, route_to_service
from ct_routing import enrich_ct_gateway_response
from mri_routing import enrich_mri_gateway_response
from schemas import (
    GatewayResponse,
    CopilotRequest,
    CopilotResponse,
)


def invoke_triage(saved_path: str, modality: str):
    """Load triage only when needed so slim Railway images start without torch/torchxrayvision."""
    from triage import run_triage as _run_triage

    return _run_triage(saved_path, modality)


def _gateway_cors_allow_origins() -> list[str]:
    """Production: set GATEWAY_CORS_ORIGINS=comma-separated URLs. Empty = legacy allow-all (dev only)."""
    raw = os.getenv("GATEWAY_CORS_ORIGINS", "").strip()
    if not raw:
        return ["*"]
    return [x.strip() for x in raw.split(",") if x.strip()]


# ════════════════════════════════════════════════════════════
# MODEL NAME OBFUSCATION — V5
# No real model names leave the backend. Frontend sees only
# Manthana-branded engine names. Reverse engineering reveals
# nothing about underlying models.
# ════════════════════════════════════════════════════════════
MODEL_DISPLAY_NAMES = {
    # CXR
    "MedRAX-2": "Manthana CXR Engine",
    "EVA-X": "Manthana CXR Engine v2",
    "CheXagent-8b": "Manthana Report AI",
    "TorchXRayVision-DenseNet121-triage": "Manthana Quick Screen",
    "TorchXRayVision-DenseNet121-all": "Manthana CXR Engine",
    "TorchXRayVision-DenseNet121-chex": "Manthana CXR Engine v2",
    "TorchXRayVision-DenseNet121-mimic_nb": "Manthana CXR Engine v2",
    # ECG
    "ecg-fm": "Manthana ECG Engine",
    "HeartLang": "Manthana ECG Language AI",
    "Manthana-ECG-Engine": "Manthana ECG Engine",
    "ecg-digitiser": "Manthana ECG Digitizer",
    "ecg-signal-extractor": "Manthana ECG Engine",
    # Brain/Neuro
    "Prima": "Manthana Neuro Engine",
    # Segmentation
    "TotalSegmentator-v2": "Manthana Segment Engine",
    "TotalSegmentator-MRI": "Manthana Segment Engine",
    "TotalSegmentator": "Manthana Segment Engine",
    "TotalSegmentator-heartchambers": "Manthana Segment Engine",
    "TotalSegmentator-vertebrae": "Manthana Segment Engine",
    "Film-photo-stack": "Manthana Film Reconstruction",
    "SynthSeg": "Manthana Neuro Engine",
    # Pathology / cytology
    "Virchow": "Manthana Pathology Engine",
    "Virchow2": "Manthana Pathology Engine",
    "Virchow (Apache 2.0)": "Manthana Pathology Engine",
    "DSMIL-MIL": "Manthana Slide Intelligence",
    # CT / cardiac
    "RadGPT": "Manthana CT Intelligence",
    "TotalSegmentator-AAQ-proxy": "Manthana Vascular Analysis Engine",
    "Comp2Comp AAQ (FDA K243779)": "Manthana Vascular Analysis (FDA-ref K243779)",
    "Comp2Comp BMD (FDA K242295)": "Manthana Bone Density (FDA-ref K242295)",
    "Comp2Comp-spine": "Manthana Spine Density Engine",
    "Comp2Comp-liver_spleen_pancreas": "Manthana Abdominal Organ Engine",
    "Comp2Comp-spine_muscle_adipose_tissue": "Manthana Body Composition Engine",
    "nnUNet": "Manthana Cardiac Engine",
    "MedSAM2": "Manthana Assist Engine",
    # Oral
    "EfficientNet-B3": "Manthana Oral Screening Engine",
    # Dermatology
    "EfficientNet-B4-derm": "Manthana Derm Engine",
    "claude-vision-derm": "Manthana Derm Engine",
    "openrouter-vision-derm-scores": "Manthana Derm Engine",
    "openrouter_vision_v1": "Manthana Derm Engine",
    "kimi_k2.5_vision_v1": "Manthana Derm Engine",
    "claude-sonnet-4-20250514": "Manthana Intelligence Core",
    # Ultrasound / mammo
    "XZheng0427/OpenUS": "Manthana Ultrasound Engine",
    "OpenUS": "Manthana Ultrasound Engine",
    "openus": "Manthana Ultrasound Engine",
    "EchoCare": "Manthana Ultrasound Engine v2",
    "Mirai": "Manthana Mammography Engine",
    "Mirai (MIT)": "Manthana Mammography Engine",
    "DigitalEye": "Manthana Imaging Engine",
    # Report LLMs (unified / assembly)
    "DeepSeek-V3": "Manthana Report AI",
    "DeepSeek": "Manthana Report AI",
    "deepseek-v3": "Manthana Report AI",
    "gemini-1.5-flash": "Manthana Report AI",
    "gemini-2.0-flash-lite": "Manthana Report AI",
    "gemini-2.0-flash": "Manthana Report AI",
    "groq-llama-3.3-70b": "Manthana Report AI",
    "groq:llama-3.3-70b-versatile": "Manthana Report AI",
    "groq-llama-3.3-70b-versatile": "Manthana Report AI",
    "groq-llama-3.1-8b-instant": "Manthana Report AI",
    "qwen-2.5-max": "Manthana Report AI",
    "none": "Manthana Report AI",
    "fallback-en": "Manthana Report AI",
    # Triage
    "triage-heuristic": "Manthana Quick Screen",
    # CT Brain (NCCT)
    "CT-Brain-TorchScript": "Manthana Neuro CT Engine",
    "CT-Brain-CI-Dummy": "Manthana Neuro CT Engine",
    "CT-Brain-NoWeights": "Manthana Neuro CT Engine",
    "CT-Brain-TorchScript-MissingOrFailed": "Manthana Neuro CT Engine",
    "Kimi-narrative-CT-Brain": "Manthana Report AI",
    "Anthropic-narrative-CT-Brain": "Manthana Report AI",
    "OpenRouter-narrative-CT-Brain": "Manthana Report AI",
    "Kimi-narrative-MRI": "Manthana Report AI",
    "Anthropic-narrative-MRI": "Manthana Report AI",
    "OpenRouter-narrative-MRI": "Manthana Report AI",
    "openrouter-vision-oral": "Manthana Oral Intelligence",
    # Generic fallback
    "Demo-Model": "Manthana AI Engine",
}


def _obfuscate_model_names(models: list | None) -> list:
    """Replace real model names with Manthana-branded names."""
    if not models:
        return []
    out: list[str] = []
    for m in models:
        if not isinstance(m, str):
            out.append("Manthana AI Engine")
            continue
        if m.startswith("triage-pass-through-"):
            out.append("Manthana Quick Screen")
            continue
        out.append(MODEL_DISPLAY_NAMES.get(m, "Manthana AI Engine"))
    return list(dict.fromkeys(out))


def _canonical_modality(modality: str) -> str:
    m = modality.lower().strip()
    return ALIASES.get(m, m)


_PREMIUM_MODALITIES = frozenset({"ct_brain_vista", "premium_ct_unified"})

APP_VERSION = "1.0.1-jwks-auth"

app = FastAPI(
    title="Manthana Radiology Suite — Gateway",
    description="India's Complete AI Radiology Second-Opinion Suite",
    version=APP_VERSION,
)

app.include_router(admin_router)
app.include_router(consent_router)
app.include_router(auth_router)
app.include_router(ai_orchestrator_router)

# CORS — set GATEWAY_CORS_ORIGINS in production (comma-separated)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_gateway_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    incoming = (request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID") or "").strip()
    rid = incoming or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


MAX_AI_REQUEST_BYTES = int(os.environ.get("MAX_AI_REQUEST_BYTES", 10 * 1024 * 1024))


@app.middleware("http")
async def limit_ai_request_size(request: Request, call_next):
    """Reject oversized JSON bodies on /ai/* when Content-Length is set."""
    if request.url.path.startswith("/ai/"):
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > MAX_AI_REQUEST_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": (
                                "Request too large. Please compress or resize before uploading."
                            ),
                            "max_bytes": MAX_AI_REQUEST_BYTES,
                        },
                    )
            except ValueError:
                pass
    return await call_next(request)


_JWT_SECRET_DEFAULT = "change-this-to-a-random-secret-minimum-32-chars"


@app.on_event("startup")
async def _validate_secrets():
    import os as _os

    secret = _os.getenv("JWT_SECRET", _JWT_SECRET_DEFAULT)
    if secret == _JWT_SECRET_DEFAULT or len(secret) < 32:
        print(
            "FATAL: JWT_SECRET is not set or is still the default placeholder. "
            "Set a random secret of at least 32 characters in your .env before "
            "running in production.",
            file=_sys.stderr,
        )
        _sys.exit(1)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/manthana_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# MedGemma CXR middle layer (Modal workspace 2 or dedicated container). No trailing slash.
CXR_MEDGEMMA_SERVICE_URL = os.getenv(
    "CXR_MEDGEMMA_SERVICE_URL", "http://cxr_medgemma:8019"
).rstrip("/")


class CxrMedgemmaCompleteBody(BaseModel):
    session_id: str = Field(..., min_length=8)
    answers: Optional[Dict[str, Any]] = None
    skip_all: bool = False


# CT/MRI modalities that accept ZIP or multi-file film-photo batches (gateway bundles extras into one ZIP).
_FILM_PHOTO_GATEWAY_MODALITIES = frozenset(
    {"ct_brain", "brain_mri", "cardiac_ct", "spine_neuro", "abdominal_ct"}
)


async def _bundle_film_photos_as_zip(
    job_id: str,
    main_saved_path: str,
    main_filename: str | None,
    extras: List[UploadFile],
) -> tuple[str, str]:
    """
    Pack the primary upload plus additional images into a single ZIP for downstream extract.
    Caller must have at least 3 extras so total images >= 4 with the main file.
    """
    bundle_dir = os.path.join(UPLOAD_DIR, f"{job_id}_film_bundle")
    os.makedirs(bundle_dir, exist_ok=True)
    try:
        ext = Path(main_filename or "image").suffix or ".bin"
        shutil.copy2(main_saved_path, os.path.join(bundle_dir, f"00_main{ext}"))
        for i, uf in enumerate(extras, start=1):
            raw = await uf.read()
            safe = Path(uf.filename or f"extra_{i}.jpg").name
            with open(os.path.join(bundle_dir, f"{i:02d}_{safe}"), "wb") as out:
                out.write(raw)
        zip_path = os.path.join(UPLOAD_DIR, f"{job_id}_film_photos.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in sorted(os.listdir(bundle_dir)):
                fp = os.path.join(bundle_dir, name)
                if os.path.isfile(fp):
                    zf.write(fp, arcname=name)
        return zip_path, "film_photos.zip"
    finally:
        shutil.rmtree(bundle_dir, ignore_errors=True)

# ── Heatmap static file serving ──
HEATMAP_DIR = os.path.join(UPLOAD_DIR, "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)
app.mount("/heatmaps", StaticFiles(directory=HEATMAP_DIR), name="heatmaps")


@app.get("/health")
async def health():
    return {
        "service": "gateway",
        "status": "ok",
        "version": APP_VERSION,
    }


@app.post("/analyze", response_model=GatewayResponse)
async def analyze(
    request: Request,
    modality: str = Form(..., description="Service type: xray, ecg, oral_cancer, etc."),
    file: UploadFile = File(..., description="Medical image/file to analyze"),
    patient_id: str = Form(None, description="Optional patient identifier"),
    series_dir: Optional[str] = Form(
        None,
        description="Optional path to DICOM series directory on shared volume (PACS)",
    ),
    clinical_notes: Optional[str] = Form(
        None,
        description="Optional clinical context (e.g. tobacco use) forwarded to analysis services",
    ),
    source_modality: Optional[str] = Form(
        None,
        description="Optional DICOM Modality hint (e.g. MR) for CT services when PACS routes MR to CT pipeline",
    ),
    patient_context_json: Optional[str] = Form(
        None,
        description="Optional JSON object with patient context (e.g. dermatology age/sex/location)",
    ),
    skip_llm_narrative: Optional[str] = Form(
        None,
        description="Chest X-ray only: if true, body_xray returns TXRV output without Kimi narrative (MedGemma flow).",
    ),
    film_files: Annotated[
        Optional[List[UploadFile]],
        File(
            description=(
                "Optional extra images for CT/MRI film-photo mode: send with the main `file` so that "
                "total uploads are at least 4 (1 main + 3+ here). Gateway bundles into a ZIP."
            ),
        ),
    ] = None,
    token_data: dict = Depends(verify_token),
):
    """
    Main analysis endpoint.
    
    1. Validates JWT token
    2. Saves uploaded file
    3. Routes to correct service by modality
    4. Returns job ID for async polling OR direct result
    """
    job_id = str(uuid.uuid4())
    start_time = time.time()

    # Save uploaded file
    file_ext = Path(file.filename).suffix if file.filename else ""
    saved_path = os.path.join(UPLOAD_DIR, f"{job_id}{file_ext}")
    
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    canon = _canonical_modality(modality)

    # Premium modalities are enforced in gateway (do not trust UI alone).
    if canon in _PREMIUM_MODALITIES:
        sub_tier = (request.headers.get("X-Subscription-Tier") or "free").strip().lower()
        if canon == "premium_ct_unified":
            if sub_tier not in ("premium", "enterprise"):
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "Premium 3D CT requires Premium (₹3999) or Enterprise subscription. "
                        f"Current tier: {sub_tier}."
                    ),
                )
        elif sub_tier not in ("pro", "proplus", "premium", "enterprise"):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"'{modality}' requires Pro or higher subscription. "
                    f"Current tier: {sub_tier}."
                ),
            )

    # MSK MRI — no downstream Docker service (v1)
    if canon == "unsupported_mr_msk":
        import sys as _sys

        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
        from disclaimer import DISCLAIMER as _DISC

        return {
            "job_id": job_id,
            "status": "complete",
            "modality": modality,
            "analysis_depth": "deep",
            "findings": [
                {
                    "label": "MSK MRI — Not Yet Supported",
                    "description": (
                        "Musculoskeletal MRI analysis is not available in this version. "
                        "Please refer to a musculoskeletal radiologist."
                    ),
                    "severity": "info",
                    "confidence": 100.0,
                }
            ],
            "impression": "MSK MRI analysis unavailable in current version.",
            "pathology_scores": {},
            "structures": [],
            "confidence": "medium",
            "heatmap_url": None,
            "processing_time_sec": round(time.time() - start_time, 2),
            "models_used": [],
            "disclaimer": _DISC,
        }

    # Deterministic xray policy: always deep or threshold triage.
    xray_triage_policy = os.getenv("XRAY_TRIAGE_POLICY", "always_deep").strip().lower()
    # Backward-compat env: SKIP_XRAY_TRIAGE=1 forces always_deep.
    if os.getenv("SKIP_XRAY_TRIAGE", "").lower() in ("1", "true", "yes"):
        xray_triage_policy = "always_deep"
    if (
        canon == "xray"
        and xray_triage_policy == "always_deep"
    ):
        triage_result = {
            "needs_deep": True,
            "findings": [],
            "triage_scores": {},
            "triage_time_ms": 0,
            "models_used": ["triage-policy-always-deep"],
        }
    else:
        triage_result = invoke_triage(saved_path, canon)
    if not triage_result["needs_deep"]:
        return {
            "job_id": job_id,
            "status": "complete",
            "modality": modality,
            "analysis_depth": "triage",
            "findings": triage_result["findings"],
            "impression": "No significant abnormality detected on initial screening.",
            "pathology_scores": triage_result.get("triage_scores") or {},
            "structures": [],
            "confidence": "medium",
            "heatmap_url": None,
            "processing_time_sec": round(triage_result["triage_time_ms"] / 1000.0, 2),
            "models_used": _obfuscate_model_names(triage_result["models_used"]),
            "disclaimer": "AI screening triage only — clinical correlation required.",
        }

    # Route to correct service
    try:
        service_url = route_to_service(modality)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    forward_path = saved_path
    forward_name = file.filename or "upload"
    forward_mime = file.content_type or "application/octet-stream"
    bundled_zip: str | None = None
    extras = list(film_files or [])
    if canon in _FILM_PHOTO_GATEWAY_MODALITIES and len(extras) >= 3:
        bundled_zip, forward_name = await _bundle_film_photos_as_zip(
            job_id, saved_path, file.filename, extras
        )
        forward_path = bundled_zip
        forward_mime = "application/zip"

    # Forward to service (retries help Modal/serverless cold starts)
    rid = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    if canon in _PREMIUM_MODALITIES:
        sub_tier = (request.headers.get("X-Subscription-Tier") or "free").strip().lower()
        logging.getLogger("manthana.gateway").info(
            "premium_3d_forward modality=%s service_url=%s subscription_tier=%s "
            "request_id=%s model_provider=nim",
            canon,
            service_url,
            sub_tier,
            rid,
        )
    try:
        response: Optional[httpx.Response] = None
        async with httpx.AsyncClient(timeout=600.0) as client:
            for attempt in range(1):  # No retries
                with open(forward_path, "rb") as f:
                    fwd: dict = {
                        "job_id": job_id,
                        "patient_id": patient_id or "",
                        "series_dir": series_dir or "",
                        "clinical_notes": clinical_notes or "",
                        "source_modality": source_modality or "",
                        "patient_context_json": patient_context_json or "",
                    }
                    if canon == "xray" and str(skip_llm_narrative or "").strip().lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    ):
                        fwd["skip_llm_narrative"] = "true"
                    response = await client.post(
                        service_url,
                        files={"file": (forward_name, f, forward_mime)},
                        data=fwd,
                        headers={"X-Request-ID": rid},
                    )
                break  # No retries

        if response is not None and response.status_code == 200:
            result = response.json()
            result["job_id"] = job_id
            result["processing_time_sec"] = round(time.time() - start_time, 2)
            result["analysis_depth"] = "deep"
            result["models_used"] = _obfuscate_model_names(result.get("models_used"))
            enrich_ct_gateway_response(
                result,
                request_modality=modality,
                patient_context_json=patient_context_json,
            )
            enrich_mri_gateway_response(result, request_modality=modality)

            # Auto-generate case embedding (async fire-and-forget)
            _trigger_case_embedding(job_id, patient_id, modality, result)
            
            return result
        elif response is not None:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Service error: {response.text}",
            )
        raise HTTPException(status_code=502, detail="No response from analysis service.")

    except httpx.TimeoutException:
        return GatewayResponse(
            job_id=job_id,
            status="queued",
            message=f"Analysis queued. Poll GET /job/{job_id}/status",
        ).model_dump()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Service '{modality}' is not available. Check if it's running.",
        )
    finally:
        if bundled_zip and os.path.isfile(bundled_zip):
            try:
                os.unlink(bundled_zip)
            except OSError:
                pass


@app.get("/job/{job_id}/status")
async def job_status(job_id: str, token_data: dict = Depends(verify_token)):
    """Poll job status for async analysis."""
    # In production, this checks Redis queue
    import sys
    sys.path.insert(0, "/app/shared")
    from queue_client import get_job_status
    
    status = get_job_status(job_id)
    return status


@app.get("/services")
async def list_services():
    """List all available analysis services and their status."""
    from router import SERVICE_MAP
    
    services = []
    for modality, url in SERVICE_MAP.items():
        # Serverless Modalities are always "online" if configured.
        # Making live HTTP requests defeats scale-to-zero and costs money.
        status = "online" if url and url.startswith("http") else "offline"
        
        services.append({
            "modality": modality,
            "status": status,
            "endpoint": f"POST /analyze (modality={modality})",
        })
    
    return {"services": services}


@app.get("/health/services")
async def health_services():
    """Alias for frontend clients that call /health/services."""
    return await list_services()


# ═════════════════════════════════════════════════════════════════════════════
# Case Embeddings API (Parrotlet-e Integration)
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/cases/{case_id}/embed")
async def embed_case(
    case_id: str,
    patient_id: Optional[str] = None,
    modality: Optional[str] = None,
    findings: Optional[list] = None,
    impression: Optional[str] = None,
    pathology_scores: Optional[dict] = None,
    structures: Optional[list] = None,
    lab_values: Optional[dict] = None,
    token_data: dict = Depends(verify_token),
):
    """
    Generate and store embedding for a completed case.
    Called after analysis completes to enable similarity search.
    """
    import sys
    sys.path.insert(0, "/app/shared")
    
    try:
        from case_embeddings import build_case_summary, store_case_embedding
        
        # Build canonical summary
        case_summary = build_case_summary(
            modality=modality or "unknown",
            findings=findings or [],
            impression=impression or "",
            pathology_scores=pathology_scores or {},
            structures=structures or [],
            lab_values=lab_values,
        )
        
        # Store embedding
        record = store_case_embedding(
            case_id=case_id,
            case_summary=case_summary,
            patient_id=patient_id,
            modality=modality,
            metadata={
                "pathology_scores": pathology_scores,
                "structures": structures,
                "findings_count": len(findings) if findings else 0,
            },
        )
        
        return {
            "case_id": case_id,
            "status": "embedded",
            "embedding_dim": record["embedding_dim"],
            "model": record["model"],
            "created_at": record["created_at"],
        }
        
    except Exception as e:
        logger = logging.getLogger("manthana.gateway")
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {str(e)}",
        )


@app.get("/cases/similar")
async def find_similar_cases(
    case_id: Optional[str] = None,
    query_text: Optional[str] = None,
    patient_id: Optional[str] = None,
    top_k: int = 5,
    modality_filter: Optional[str] = None,
    token_data: dict = Depends(verify_token),
):
    """
    Find similar cases by case_id, text query, or patient history.
    
    - case_id: Find cases similar to this completed case
    - query_text: Semantic search by description (e.g., "pneumonia with pleural effusion")
    - patient_id: Get similar cases for this patient's history
    """
    import sys
    sys.path.insert(0, "/app/shared")
    
    try:
        from case_embeddings import find_similar_cases as _find_similar
        
        # Build embedding from text if provided
        query_embedding = None
        if query_text and not case_id:
            from case_embeddings import generate_case_embedding
            temp = generate_case_embedding(
                case_summary=query_text,
                case_id="temp_query",
            )
            query_embedding = temp["embedding"]
        
        results = _find_similar(
            case_id=case_id,
            query_embedding=query_embedding,
            top_k=top_k,
            modality_filter=modality_filter,
        )
        
        # Filter by patient if requested (don't expose cross-patient data)
        if patient_id:
            import hashlib
            patient_hash = hashlib.sha256(patient_id.encode()).hexdigest()[:16]
            results = [r for r in results if r.get("patient_hash") == patient_hash]
        
        return {
            "query_case_id": case_id,
            "query_text": query_text[:100] if query_text else None,
            "results_count": len(results),
            "results": results,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Similarity search failed: {str(e)}",
        )


@app.post("/copilot", response_model=CopilotResponse)
async def copilot(
    request: CopilotRequest,
    token_data: dict = Depends(verify_token),
):
    import json

    from llm_router import llm_router

    modality = request.context.get("modality", "unknown")
    findings = request.context.get("findings", [])
    impression = request.context.get("impression", "")
    pathology_scores = request.context.get("pathology_scores", {})

    system_prompt = f"""You are a senior radiologist AI assistant for Indian 
clinical practice. A radiologist is asking a follow-up question about an 
AI-generated analysis report.

Modality: {modality}
Current impression: {impression}
Key findings: {json.dumps(findings, indent=2)[:3000]}
Pathology scores: {json.dumps(pathology_scores, indent=2)[:2000]}

Answer the clinician's question concisely and accurately. Use Indian 
epidemiological context where relevant (TB, NCC, NAFLD, etc.). 
If the question is outside radiology scope, say so clearly.
End with: "Clinical correlation and radiologist verification required."
"""

    def _sync() -> tuple[str, str]:
        r = llm_router.complete(
            prompt=request.question,
            system_prompt=system_prompt,
            task_type="copilot",
            max_tokens=1024,
            temperature=0.3,
        )
        return (r.get("content") or "").strip(), str(r.get("model_used", "openrouter"))

    try:
        content, model_used = await asyncio.to_thread(_sync)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"CoPilot LLM unavailable: {exc!s}",
        ) from exc

    if not content:
        raise HTTPException(
            status_code=503,
            detail="CoPilot service temporarily unavailable. Set OPENROUTER_API_KEY.",
        )

    return CopilotResponse(
        response=content,
        model_used=_obfuscate_model_names([model_used])[0],
    )


@app.post("/cxr-medgemma/session/start")
async def cxr_medgemma_session_start(
    file: UploadFile = File(...),
    pathology_scores_json: str = Form(...),
    patient_context_json: Optional[str] = Form(None),
    token_data: dict = Depends(verify_token),
):
    """Proxy to CXR MedGemma service (TXRV scores + image + context → questions)."""
    if not CXR_MEDGEMMA_SERVICE_URL:
        raise HTTPException(
            status_code=503,
            detail="CXR_MEDGEMMA_SERVICE_URL is not configured.",
        )
    url = f"{CXR_MEDGEMMA_SERVICE_URL}/medgemma-cxr/session/start"
    raw = await file.read()
    files = {
        "file": (
            file.filename or "upload.bin",
            raw,
            file.content_type or "application/octet-stream",
        )
    }
    data: dict = {"pathology_scores_json": pathology_scores_json}
    if patient_context_json is not None:
        data["patient_context_json"] = patient_context_json
    timeout = httpx.Timeout(600.0, connect=60.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, files=files, data=data)
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"CXR MedGemma service unreachable: {e}",
        ) from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail="CXR MedGemma request timed out.") from e
    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=resp.text[:4000] or "upstream error",
        )
    return resp.json()


@app.post("/cxr-medgemma/session/complete")
async def cxr_medgemma_session_complete(
    body: Annotated[CxrMedgemmaCompleteBody, Body()],
    token_data: dict = Depends(verify_token),
):
    """Proxy finalize: answers → Kimi final narrative."""
    if not CXR_MEDGEMMA_SERVICE_URL:
        raise HTTPException(
            status_code=503,
            detail="CXR_MEDGEMMA_SERVICE_URL is not configured.",
        )
    url = f"{CXR_MEDGEMMA_SERVICE_URL}/medgemma-cxr/session/complete"
    timeout = httpx.Timeout(300.0, connect=30.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body.model_dump())
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"CXR MedGemma service unreachable: {e}",
        ) from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail="CXR MedGemma finalize timed out.") from e
    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=resp.text[:4000] or "upstream error",
        )
    return resp.json()


# ════════════════════════════════════════════════════════
# PACS REVERSE PROXY
# Forwards all /pacs/* requests to the pacs_bridge service.
# Frontend calls: gateway:8000/pacs/studies → pacs_bridge:8030/pacs/studies
# ════════════════════════════════════════════════════════

PACS_BRIDGE_URL = os.getenv("PACS_BRIDGE_URL", "http://pacs_bridge:8030")


@app.api_route("/pacs/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def pacs_proxy(
    path: str,
    request: Request,
    token_data: dict = Depends(verify_token),
):
    """
    Transparent reverse proxy for all PACS operations.

    Forwards the full request (method, headers, query params, body)
    to the pacs_bridge service, which hosts all /pacs/* endpoints:
      /pacs/studies, /pacs/worklist, /pacs/send-to-ai,
      /pacs/modalities/{name}/echo, /pacs/config, etc.
    """
    target_url = f"{PACS_BRIDGE_URL}/pacs/{path}"

    # Preserve query string
    if request.url.query:
        target_url += f"?{request.url.query}"

    # Read raw body (for POST/PUT)
    body = await request.body()

    # Forward headers (skip hop-by-hop headers)
    forward_headers = {}
    skip_headers = {"host", "content-length", "transfer-encoding", "connection"}
    for key, value in request.headers.items():
        if key.lower() not in skip_headers:
            forward_headers[key] = value

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=forward_headers,
                content=body if body else None,
            )

        # Return the pacs_bridge response as-is
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type"),
        )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="PACS Bridge service is not available. Check if pacs_bridge is running.",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="PACS request timed out. The Orthanc server may be slow or unreachable.",
        )


# ════════════════════════════════════════════════════════
# ORACLE SERVICE REVERSE PROXY (/v1/*)
# Forwards to oracle-service (chat, M5, health). Use ORACLE_SERVICE_URL with
# *.railway.internal on Railway for free private egress.
# ════════════════════════════════════════════════════════

ORACLE_SERVICE_URL = os.getenv("ORACLE_SERVICE_URL", "http://oracle_service:8000").rstrip(
    "/"
)

_ORACLE_SKIP_REQ_HEADERS = frozenset(
    {"host", "content-length", "transfer-encoding", "connection", "te"}
)
_ORACLE_SKIP_RESP_HEADERS = frozenset(
    {"transfer-encoding", "connection", "content-length"}
)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])
async def oracle_proxy(
    path: str,
    request: Request,
    token_data: dict = Depends(verify_token),
):
    """
    Proxy Oracle API (e.g. POST /v1/chat SSE, /v1/chat/m5, GET /v1/health).
    Requires the same Bearer JWT as other protected routes (Supabase or legacy).
    """
    target_url = f"{ORACLE_SERVICE_URL}/v1/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    body = await request.body()
    forward_headers = {}
    for key, value in request.headers.items():
        if key.lower() not in _ORACLE_SKIP_REQ_HEADERS:
            forward_headers[key] = value

    timeout = httpx.Timeout(300.0, connect=30.0)
    client = httpx.AsyncClient(timeout=timeout)
    try:
        req = client.build_request(
            request.method,
            target_url,
            headers=forward_headers,
            content=body if body else None,
        )
        upstream = await client.send(req, stream=True)
    except httpx.ConnectError:
        await client.aclose()
        raise HTTPException(
            status_code=503,
            detail="Oracle service is not available. Check ORACLE_SERVICE_URL.",
        ) from None
    except httpx.TimeoutException:
        await client.aclose()
        raise HTTPException(
            status_code=504,
            detail="Oracle request timed out.",
        ) from None
    except Exception:
        await client.aclose()
        raise

    out_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in _ORACLE_SKIP_RESP_HEADERS
    }

    async def _iterate():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        _iterate(),
        status_code=upstream.status_code,
        headers=out_headers,
    )


def _sanitize_for_embedding(obj):
    """Convert unhashable types (numpy arrays, slices) to serializable Python types."""
    if hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    if isinstance(obj, dict):
        return {k: _sanitize_for_embedding(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_embedding(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_embedding(item) for item in obj)
    return obj


def _trigger_case_embedding(job_id: str, patient_id: str, modality: str, result: dict):
    """
    Fire-and-forget case embedding generation.
    Runs in background thread to not block response.
    """
    import threading
    
    def _embed():
        try:
            import sys
            sys.path.insert(0, "/app/shared")
            from case_embeddings import build_case_summary, store_case_embedding
            
            # Sanitize result to remove unhashable numpy arrays/slices
            safe_result = _sanitize_for_embedding(result)
            
            case_summary = build_case_summary(
                modality=modality,
                findings=safe_result.get("findings", []),
                impression=safe_result.get("impression", ""),
                pathology_scores=safe_result.get("pathology_scores", {}),
                structures=safe_result.get("structures", []),
                lab_values=safe_result.get("labs"),
            )
            
            store_case_embedding(
                case_id=job_id,
                case_summary=case_summary,
                patient_id=patient_id,
                modality=modality,
                metadata={
                    "pathology_scores": safe_result.get("pathology_scores"),
                    "structures": safe_result.get("structures"),
                    "models_used": safe_result.get("models_used", []),
                },
            )
            
            logging.getLogger("manthana.gateway").info(f"Embedding created for case: {job_id}")
            
        except Exception as e:
            logging.getLogger("manthana.gateway").error(f"Background embedding failed: {e}", exc_info=True)
    
    # Run in background thread
    thread = threading.Thread(target=_embed, daemon=True)
    thread.start()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("GATEWAY_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
