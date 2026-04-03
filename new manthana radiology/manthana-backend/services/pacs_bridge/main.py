"""
Manthana — PACS Bridge Service
DICOMweb proxy + Worklist management + Hospital PACS connectivity.
Bridges Orthanc DICOM server with the Manthana Gateway.
"""
import os
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from orthanc_client import OrthancClient
from worklist_manager import WorklistManager
from dicom_router import detect_modality_from_tags

app = FastAPI(
    title="Manthana — PACS Bridge",
    description="DICOMweb proxy, worklist management, and hospital PACS connectivity",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ORTHANC_URL = os.getenv("ORTHANC_URL", "http://orthanc:8042")
ORTHANC_USER = os.getenv("ORTHANC_USERNAME", "manthana")
ORTHANC_PASS = os.getenv("ORTHANC_PASSWORD", "manthana-pacs-secret")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/manthana_uploads")

orthanc = OrthancClient(ORTHANC_URL, ORTHANC_USER, ORTHANC_PASS)
worklist = WorklistManager(orthanc)


# ══════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    orthanc_ok = await orthanc.check_health()
    return {
        "service": "pacs_bridge",
        "status": "ok",
        "orthanc": "connected" if orthanc_ok else "disconnected",
        "orthanc_url": ORTHANC_URL,
    }


# ══════════════════════════════════════════════════════════
# STUDIES — QIDO-RS proxy
# ══════════════════════════════════════════════════════════

class StudyResponse(BaseModel):
    orthanc_id: str
    study_instance_uid: str
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None
    study_date: Optional[str] = None
    study_description: Optional[str] = None
    modality: Optional[str] = None
    body_part: Optional[str] = None
    institution: Optional[str] = None
    series_count: int = 0
    instance_count: int = 0
    ai_status: Optional[str] = None
    ai_job_id: Optional[str] = None


@app.get("/pacs/studies", response_model=List[StudyResponse])
async def list_studies(
    patient_name: Optional[str] = Query(None, description="Filter by patient name"),
    patient_id: Optional[str] = Query(None, description="Filter by patient ID"),
    modality: Optional[str] = Query(None, description="Filter by modality (CR, CT, MR, etc.)"),
    date_from: Optional[str] = Query(None, description="Study date from (YYYYMMDD)"),
    date_to: Optional[str] = Query(None, description="Study date to (YYYYMMDD)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List studies from Orthanc with optional filters."""
    studies = await orthanc.list_studies(limit=limit, offset=offset)
    results = []

    for study in studies:
        tags = study.get("MainDicomTags", {})
        patient_tags = study.get("PatientMainDicomTags", {})

        # Apply filters
        if patient_name and patient_name.upper() not in (patient_tags.get("PatientName", "")).upper():
            continue
        if patient_id and patient_id != patient_tags.get("PatientID", ""):
            continue
        if modality:
            study_modalities = tags.get("ModalitiesInStudy", "")
            if modality.upper() not in study_modalities.upper():
                continue
        if date_from and tags.get("StudyDate", "") < date_from:
            continue
        if date_to and tags.get("StudyDate", "") > date_to:
            continue

        # Get AI metadata
        ai_status = await orthanc.get_metadata(f"/studies/{study['ID']}/metadata/ai-status")
        ai_job_id = await orthanc.get_metadata(f"/studies/{study['ID']}/metadata/ai-job-id")

        results.append(StudyResponse(
            orthanc_id=study["ID"],
            study_instance_uid=tags.get("StudyInstanceUID", ""),
            patient_name=patient_tags.get("PatientName", ""),
            patient_id=patient_tags.get("PatientID", ""),
            patient_age=patient_tags.get("PatientBirthDate", ""),
            patient_sex=patient_tags.get("PatientSex", ""),
            study_date=tags.get("StudyDate", ""),
            study_description=tags.get("StudyDescription", ""),
            modality=tags.get("ModalitiesInStudy", ""),
            body_part=tags.get("BodyPartExamined", ""),
            institution=tags.get("InstitutionName", ""),
            series_count=len(study.get("Series", [])),
            instance_count=sum(
                len(await orthanc.get_json(f"/series/{sid}/instances"))
                for sid in study.get("Series", [])
            ) if study.get("Series") else 0,
            ai_status=ai_status,
            ai_job_id=ai_job_id,
        ))

    return results


@app.get("/pacs/studies/{study_id}")
async def get_study(study_id: str):
    """Get detailed study info including series and instance counts."""
    return await orthanc.get_json(f"/studies/{study_id}")


@app.get("/pacs/studies/{study_id}/series")
async def get_study_series(study_id: str):
    """List all series in a study."""
    return await orthanc.get_json(f"/studies/{study_id}/series")


# ══════════════════════════════════════════════════════════
# WADO-RS — DICOMweb retrieval proxy
# ══════════════════════════════════════════════════════════

@app.get("/pacs/wado-rs/studies/{study_uid}")
async def wado_retrieve_study(study_uid: str):
    """WADO-RS study retrieval (proxied to Orthanc DICOMweb)."""
    return await orthanc.wado_rs(f"/studies/{study_uid}")


@app.get("/pacs/wado-rs/studies/{study_uid}/series/{series_uid}")
async def wado_retrieve_series(study_uid: str, series_uid: str):
    """WADO-RS series retrieval."""
    return await orthanc.wado_rs(f"/studies/{study_uid}/series/{series_uid}")


@app.get("/pacs/wado-rs/studies/{study_uid}/series/{series_uid}/instances/{instance_uid}")
async def wado_retrieve_instance(study_uid: str, series_uid: str, instance_uid: str):
    """WADO-RS instance retrieval."""
    return await orthanc.wado_rs(
        f"/studies/{study_uid}/series/{series_uid}/instances/{instance_uid}"
    )


# ══════════════════════════════════════════════════════════
# SEND TO AI — Trigger analysis on a study
# ══════════════════════════════════════════════════════════

class SendToAIRequest(BaseModel):
    study_id: str = Field(..., description="Orthanc study ID")
    modality_override: Optional[str] = Field(None, description="Override auto-detected modality")
    patient_id: Optional[str] = None


@app.post("/pacs/send-to-ai")
async def send_to_ai(req: SendToAIRequest):
    """
    Download all instances from the first series to a shared directory, send the first slice
    plus series_dir to the Gateway for AI analysis (full volume + Comp2Comp when applicable).
    Tags the study with AI status metadata.
    """
    # 1. Get study details
    study = await orthanc.get_json(f"/studies/{req.study_id}")
    if not study:
        raise HTTPException(404, f"Study {req.study_id} not found in Orthanc")

    tags = study.get("MainDicomTags", {})
    patient_tags = study.get("PatientMainDicomTags", {})

    # 2. Detect modality
    modality = req.modality_override or detect_modality_from_tags(tags, patient_tags)

    # 3. First series — materialize full series on disk (shared volume with gateway)
    series_list = study.get("Series", [])
    if not series_list:
        raise HTTPException(400, "Study has no series")

    series_id = series_list[0]
    instances = await orthanc.get_json(f"/series/{series_id}/instances")
    if not instances:
        raise HTTPException(400, "Series has no instances")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    series_path = os.path.join(UPLOAD_DIR, f"series_{series_id}")
    os.makedirs(series_path, exist_ok=True)

    first_dicom_bytes = None
    for inst in instances:
        iid = inst["ID"] if isinstance(inst, dict) else inst
        dicom_bytes = await orthanc.get_bytes(f"/instances/{iid}/file")
        out_fp = os.path.join(series_path, f"{iid}.dcm")
        with open(out_fp, "wb") as f:
            f.write(dicom_bytes)
        if first_dicom_bytes is None:
            first_dicom_bytes = dicom_bytes

    if first_dicom_bytes is None:
        raise HTTPException(400, "Could not download series instances")

    # 5. Send to Gateway
    patient_id = req.patient_id or patient_tags.get("PatientID", "PACS-IMPORT")

    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{GATEWAY_URL}/analyze",
            files={"file": ("study.dcm", first_dicom_bytes, "application/dicom")},
            data={
                "modality": modality,
                "patient_id": patient_id,
                "series_dir": series_path,
            },
        )

    if response.status_code == 200:
        result = response.json()
        job_id = result.get("job_id", "")

        # Tag study with AI job
        await orthanc.put_metadata(f"/studies/{req.study_id}/metadata/ai-job-id", job_id)
        await orthanc.put_metadata(f"/studies/{req.study_id}/metadata/ai-status", "analyzing")

        return {
            "status": "sent",
            "job_id": job_id,
            "modality": modality,
            "patient_name": patient_tags.get("PatientName", ""),
            "study_id": req.study_id,
        }
    else:
        await orthanc.put_metadata(f"/studies/{req.study_id}/metadata/ai-status", "failed")
        raise HTTPException(response.status_code, f"Gateway error: {response.text}")


# ══════════════════════════════════════════════════════════
# AUTO-ROUTE — Called by Orthanc Lua script
# ══════════════════════════════════════════════════════════

class AutoRouteRequest(BaseModel):
    study_id: str
    patient_name: str = ""
    patient_id: str = ""
    modality: str = "xray"
    series_count: int = 0
    instance_count: int = 0


@app.post("/pacs/auto-route")
async def auto_route(req: AutoRouteRequest):
    """Called by Orthanc Lua auto-route script on stable study."""
    try:
        result = await send_to_ai(SendToAIRequest(
            study_id=req.study_id,
            modality_override=req.modality,
            patient_id=req.patient_id or None,
        ))
        return {"status": "routed", **result}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ══════════════════════════════════════════════════════════
# WORKLIST
# ══════════════════════════════════════════════════════════

class WorklistItem(BaseModel):
    id: Optional[str] = None
    patient_name: str
    patient_id: str
    accession_number: Optional[str] = None
    modality: str = "CR"
    scheduled_date: str  # YYYYMMDD
    scheduled_time: Optional[str] = None  # HHMMSS
    scheduled_aet: str = "MANTHANA"
    procedure_description: Optional[str] = None
    referring_physician: Optional[str] = None
    status: str = "SCHEDULED"  # SCHEDULED | IN_PROGRESS | COMPLETED


@app.get("/pacs/worklist")
async def list_worklist():
    """List all worklist items."""
    return await worklist.list_items()


@app.post("/pacs/worklist")
async def create_worklist_item(item: WorklistItem):
    """Create a new worklist item."""
    return await worklist.create_item(item.model_dump())


@app.put("/pacs/worklist/{item_id}")
async def update_worklist_item(item_id: str, item: WorklistItem):
    """Update a worklist item."""
    return await worklist.update_item(item_id, item.model_dump())


@app.delete("/pacs/worklist/{item_id}")
async def delete_worklist_item(item_id: str):
    """Delete a worklist item."""
    return await worklist.delete_item(item_id)


# ══════════════════════════════════════════════════════════
# REMOTE MODALITIES — Hospital PACS connectivity
# ══════════════════════════════════════════════════════════

@app.get("/pacs/modalities")
async def list_modalities():
    """List configured remote DICOM modalities (hospital PACS)."""
    return await orthanc.get_json("/modalities")


class ModalityConfig(BaseModel):
    name: str
    aet: str
    host: str
    port: int = 11112
    manufacturer: Optional[str] = None


@app.post("/pacs/modalities")
async def add_modality(config: ModalityConfig):
    """Add or update a remote DICOM modality."""
    payload = [config.aet, config.host, config.port]
    if config.manufacturer:
        payload.append(config.manufacturer)
    await orthanc.put_json(f"/modalities/{config.name}", payload)
    return {"status": "added", "name": config.name}


@app.delete("/pacs/modalities/{name}")
async def delete_modality(name: str):
    """Remove a remote modality."""
    await orthanc.delete(f"/modalities/{name}")
    return {"status": "deleted", "name": name}


class QueryRequest(BaseModel):
    level: str = "Study"  # Study | Series | Instance
    query: dict = {}  # DICOM tag filters e.g. {"PatientName": "DOE*"}


@app.post("/pacs/modalities/{name}/query")
async def query_modality(name: str, req: QueryRequest):
    """C-FIND query on a remote PACS."""
    payload = {"Level": req.level, "Query": req.query}
    return await orthanc.post_json(f"/modalities/{name}/query", payload)


@app.post("/pacs/modalities/{name}/retrieve")
async def retrieve_from_modality(name: str, req: QueryRequest):
    """C-MOVE retrieve from a remote PACS into Orthanc."""
    # First query, then retrieve matching
    query_result = await orthanc.post_json(f"/modalities/{name}/query", {
        "Level": req.level, "Query": req.query
    })
    query_id = query_result.get("ID", "")
    if not query_id:
        raise HTTPException(400, "Query returned no results")

    # Retrieve all matches
    result = await orthanc.post_json(f"/queries/{query_id}/retrieve", {"TargetAet": "MANTHANA"})
    return {"status": "retrieving", "query_id": query_id, "result": result}


@app.post("/pacs/modalities/{name}/echo")
async def echo_modality(name: str):
    """C-ECHO — test connectivity to a remote PACS."""
    try:
        result = await orthanc.post_json(f"/modalities/{name}/echo", {})
        return {"status": "ok", "name": name}
    except Exception as e:
        return {"status": "failed", "name": name, "error": str(e)}


# ══════════════════════════════════════════════════════════
# PACS CONFIG
# ══════════════════════════════════════════════════════════

@app.get("/pacs/config")
async def get_pacs_config():
    """Get current PACS configuration summary."""
    system = await orthanc.get_json("/system")
    modalities = await orthanc.get_json("/modalities")
    return {
        "orthanc_version": system.get("Version", "unknown"),
        "dicom_aet": system.get("DicomAet", "MANTHANA"),
        "dicom_port": system.get("DicomPort", 4242),
        "plugins": system.get("Plugins", []),
        "studies_count": system.get("TotalDiskSizeMB", 0),
        "modalities": modalities,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT_PACS_BRIDGE", 8030))
    uvicorn.run(app, host="0.0.0.0", port=port)
