"""
ZeroClaw / Kimi triage agent — tool registry and executors.

Tools are registered by name for ReAct-style callers. Executors return JSON-serializable
dicts; unavailability must not raise (avoid agent retry loops).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from typing import Any, Callable

logger = logging.getLogger("manthana.zeroclaw_tools")

# Stable module names so tests can patch `manthana_spine_service.inference.run_*`
# without colliding with other services that also expose `inference.py`.
_SPINE_INF_MODULE = "manthana_zeroclaw_spine_inference"
_MAMMO_INF_MODULE = "manthana_zeroclaw_mammography_inference"

_BACKEND_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_ORAL_SERVICE = os.path.join(_BACKEND_ROOT, "services", "14_oral_cancer")
_BRAIN_MRI = os.path.join(_BACKEND_ROOT, "services", "02_brain_mri")
_SPINE_NEURO = os.path.join(_BACKEND_ROOT, "services", "10_spine_neuro")
_LAB_REPORT = os.path.join(_BACKEND_ROOT, "services", "15_lab_report")
_DERM = os.path.join(_BACKEND_ROOT, "services", "16_dermatology")
_PATHOLOGY = os.path.join(_BACKEND_ROOT, "services", "05_pathology")
_CYTOLOGY = os.path.join(_BACKEND_ROOT, "services", "11_cytology")
_MAMMOGRAPHY = os.path.join(_BACKEND_ROOT, "services", "12_mammography")
_BODY_XRAY = os.path.join(_BACKEND_ROOT, "services", "01_body_xray")
_ABDOMINAL_CT = os.path.join(_BACKEND_ROOT, "services", "08_abdominal_ct")
_ECG = os.path.join(_BACKEND_ROOT, "services", "13_ecg")
_USG = os.path.join(_BACKEND_ROOT, "services", "09_ultrasound")
_SHARED = os.path.normpath(os.path.join(os.path.dirname(__file__)))


def _ensure_oral_paths() -> None:
    for p in (_SHARED, _ORAL_SERVICE):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_brain_mri_paths() -> None:
    for p in (_SHARED, _BRAIN_MRI):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_spine_paths() -> None:
    # Insert service first, then shared, so `import preprocessing` resolves to shared/preprocessing/.
    for p in (_SPINE_NEURO, _SHARED):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_lab_report_paths() -> None:
    for p in (_SHARED, _LAB_REPORT):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_derm_paths() -> None:
    for p in (_SHARED, _DERM):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_pathology_paths() -> None:
    # Insert service dir first, then shared, so final order is [shared, service, ...]
    # and `import preprocessing` resolves to shared/preprocessing/ (not service/preprocessing.py).
    for p in (_PATHOLOGY, _SHARED):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_cytology_paths() -> None:
    for p in (_CYTOLOGY, _SHARED):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_mammography_paths() -> None:
    for p in (_SHARED, _MAMMOGRAPHY):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_body_xray_paths() -> None:
    for p in (_SHARED, _BODY_XRAY):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_abdominal_ct_paths() -> None:
    for p in (_SHARED, _ABDOMINAL_CT):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_ecg_paths() -> None:
    for p in (_SHARED, _ECG):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_usg_paths() -> None:
    for p in (_USG, _SHARED):
        if p not in sys.path:
            sys.path.insert(0, p)


def _exec_analyze_abdominal_ct(
    image_b64: str,
    patient_context_json: str = "{}",
    filename_hint: str = "",
    series_dir: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Abdominal CT (TotalSegmentator + Comp2Comp when series available); never raises."""
    _ = kwargs
    _ensure_abdominal_ct_paths()
    try:
        from inference import run_pipeline_b64
    except ImportError as e:
        return {
            "available": False,
            "reason": "import_error",
            "message": str(e),
        }
    try:
        return run_pipeline_b64(
            image_b64,
            patient_context=patient_context_json,
            job_id="zeroclaw_abdominal_ct",
            filename_hint=filename_hint or "study.dcm",
            series_dir=series_dir or "",
        )
    except ValueError as e:
        logger.warning("analyze_abdominal_ct invalid input: %s", e)
        return {
            "available": False,
            "reason": "invalid_input",
            "message": str(e),
        }
    except Exception as e:
        logger.exception("analyze_abdominal_ct failed: %s", e)
        return {
            "available": False,
            "reason": "inference_error",
            "message": str(e),
        }


def _exec_analyze_ecg(
    image_b64: str,
    patient_context_json: str = "{}",
    filename_hint: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """12-lead ECG from CSV/EDF/DICOM/PNG base64 via run_ecg_pipeline_b64; never raises."""
    _ = kwargs
    _ensure_ecg_paths()
    try:
        from inference import run_ecg_pipeline_b64
    except ImportError as e:
        return {
            "available": False,
            "reason": "import_error",
            "message": str(e),
        }
    try:
        return run_ecg_pipeline_b64(
            image_b64,
            patient_context=patient_context_json,
            job_id="zeroclaw_ecg",
            filename_hint=filename_hint or "",
        )
    except ValueError as e:
        logger.warning("analyze_ecg invalid input: %s", e)
        return {
            "available": False,
            "reason": "invalid_input",
            "message": str(e),
        }


def _exec_analyze_usg(tool_input: Any) -> dict[str, Any]:
    """
    ZeroClaw executor for analyze_usg.
    Handles both dict and JSON-string tool_input.
    """
    _ensure_usg_paths()
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except Exception:
            return {
                "available": False,
                "reason": "bad_input",
                "findings": "Invalid tool input.",
            }
    if not isinstance(tool_input, dict):
        return {
            "available": False,
            "reason": "bad_input",
            "findings": "Invalid tool input.",
        }

    image_b64 = str(tool_input.get("image_b64") or "")
    patient_context_json = tool_input.get("patient_context_json")
    job_id = tool_input.get("job_id")

    if isinstance(patient_context_json, dict):
        patient_context_json = json.dumps(patient_context_json)

    try:
        from inference import run_usg_pipeline_b64, enrich_usg_pipeline_output
    except ImportError as e:
        return {
            "available": False,
            "reason": "import_error",
            "message": str(e),
        }

    result = run_usg_pipeline_b64(
        image_b64=image_b64,
        patient_context_json=patient_context_json,
        job_id=job_id,
    )
    if result.get("available") is False:
        return result
    return enrich_usg_pipeline_output(result)


def _exec_analyze_chest_xray(
    image_b64: str,
    patient_context_json: str = "{}",
    **kwargs: Any,
) -> dict[str, Any]:
    """Chest X-ray (TorchXRayVision); never raises."""
    _ = kwargs
    _ensure_body_xray_paths()
    try:
        from inference import run_cxr_pipeline_b64
    except ImportError as e:
        return {
            "available": False,
            "reason": "import_error",
            "message": str(e),
        }
    try:
        return run_cxr_pipeline_b64(
            image_b64,
            patient_context=patient_context_json,
            job_id="zeroclaw_cxr",
        )
    except ValueError as e:
        logger.warning("analyze_chest_xray invalid input: %s", e)
        return {
            "available": False,
            "reason": "invalid_input",
            "message": str(e),
        }
    except Exception as e:
        logger.exception("analyze_chest_xray failed: %s", e)
        return {
            "available": False,
            "reason": "inference_error",
            "message": str(e),
        }


def _exec_analyze_oral(
    image_b64: str,
    clinical_notes: str = "",
    filename_hint: str = "",
    patient_context_json: str = "",
    input_type: str = "",
) -> dict[str, Any]:
    """Run oral cancer b64 pipeline; never raises — returns available:false on gate/errors."""
    _ensure_oral_paths()
    try:
        from inference import (
            OralClassificationFailed,
            OralServiceUnavailableError,
            run_oral_cancer_pipeline_b64,
        )
    except ImportError as e:
        return {
            "available": False,
            "reason": "import_error",
            "message": str(e),
        }
    ctx: dict[str, Any] = {}
    if patient_context_json and str(patient_context_json).strip():
        try:
            parsed = json.loads(patient_context_json)
            if isinstance(parsed, dict):
                ctx = parsed
        except json.JSONDecodeError:
            ctx = {}
    it = (input_type or "").strip()
    if it not in ("clinical_photo", "histopathology", "mixed", "unknown"):
        it = ""
    try:
        return run_oral_cancer_pipeline_b64(
            image_b64,
            clinical_notes=clinical_notes,
            filename_hint=filename_hint,
            patient_context=ctx,
            input_type=it or None,
        )
    except OralServiceUnavailableError:
        return {
            "available": False,
            "reason": "oral_cancer_model_not_deployed",
            "message": "Oral cancer screening checkpoint not yet deployed. Skip this tool.",
        }
    except OralClassificationFailed as e:
        return {
            "available": False,
            "reason": "classification_failed",
            "message": str(e),
        }
    except Exception as e:
        logger.exception("analyze_oral failed: %s", e)
        return {
            "available": False,
            "reason": "inference_error",
            "message": str(e),
        }


def _exec_analyze_oral_cancer(
    image_b64: str,
    patient_context_json: str = "{}",
    image_type: str = "clinical_photo",
    **kwargs: Any,
) -> dict[str, Any]:
    """Primary ZeroClaw entry for oral screening; never raises."""
    _ = kwargs
    _ensure_oral_paths()
    try:
        from inference import (
            OralServiceUnavailableError,
            run_oral_cancer_pipeline_b64,
        )
    except ImportError as e:
        return {
            "available": False,
            "reason": "import_error",
            "message": str(e),
        }
    try:
        return run_oral_cancer_pipeline_b64(
            image_b64,
            patient_context_json=patient_context_json or "{}",
            image_type=image_type or "clinical_photo",
        )
    except OralServiceUnavailableError:
        return {
            "available": False,
            "reason": "oral_cancer_service_disabled",
            "message": "Oral cancer service disabled (ORAL_CANCER_ENABLED=false).",
        }
    except Exception as e:
        logger.exception("analyze_oral_cancer failed: %s", e)
        return {
            "available": False,
            "reason": "inference_error",
            "message": str(e),
            "modality": "oral_cancer",
            "findings": [],
            "pathology_scores": {},
            "structures": {
                "biopsy_recommended": False,
                "emergency_flags": [],
                "narrative_report": "",
                "india_note": (
                    "Any persistent oral lesion >2 weeks "
                    "with tobacco/betel habit should be "
                    "referred for biopsy."
                ),
            },
        }


def _exec_analyze_brain_mri(
    image_b64: str,
    clinical_notes: str = "",
    patient_context_json: str = "",
    filename_hint: str = "",
    series_dir: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run brain MRI b64 pipeline; never raises."""
    _ = kwargs
    _ensure_brain_mri_paths()
    try:
        from inference import run_brain_mri_pipeline_b64
    except ImportError as e:
        return {
            "available": False,
            "reason": "import_error",
            "message": str(e),
        }
    try:
        return run_brain_mri_pipeline_b64(
            image_b64,
            clinical_notes=clinical_notes,
            patient_context_json=patient_context_json or "",
            filename_hint=filename_hint,
            series_dir=series_dir or "",
        )
    except Exception as e:
        logger.exception("analyze_brain_mri failed: %s", e)
        return {
            "available": False,
            "reason": "inference_error",
            "message": str(e),
        }


def _exec_analyze_spine_neuro(
    image_b64: str,
    patient_context: str = "",
    patient_context_json: str = "",
    filename_hint: str = "",
    series_dir: str = "",
    **_kwargs: Any,
) -> dict[str, Any]:
    """Run spine/neuro b64 pipeline; never raises."""
    _ensure_spine_paths()
    inf_path = os.path.join(_SPINE_NEURO, "inference.py")
    try:
        if _SPINE_INF_MODULE in sys.modules:
            mod = sys.modules[_SPINE_INF_MODULE]
        else:
            spec = importlib.util.spec_from_file_location(_SPINE_INF_MODULE, inf_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {inf_path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[_SPINE_INF_MODULE] = mod
            spec.loader.exec_module(mod)
        run_pipeline_b64 = mod.run_pipeline_b64
    except Exception as e:
        return {
            "available": False,
            "reason": "import_error",
            "message": str(e),
        }
    ctx = patient_context_json if (patient_context_json or "").strip() else patient_context
    try:
        return run_pipeline_b64(
            image_b64=image_b64,
            patient_context_json=ctx if ctx else "{}",
            filename_hint=filename_hint,
            series_dir=series_dir or "",
        )
    except Exception as e:
        logger.exception("analyze_spine_neuro failed: %s", e)
        return {
            "available": False,
            "reason": "inference_error",
            "message": str(e),
        }


def _exec_analyze_lab_report(
    document_b64: Any = "",
    clinical_notes: str = "",
    patient_context_json: str = "",
    filename_hint: str = "report.pdf",
    **extra: Any,
) -> dict[str, Any]:
    """Run lab report pipeline from base64; never raises."""
    if isinstance(document_b64, dict):
        d = document_b64
        document_b64 = str(d.get("document_b64") or "")
        clinical_notes = str(d.get("clinical_notes") or clinical_notes or "")
        patient_context_json = str(d.get("patient_context_json") or patient_context_json or "")
        filename_hint = str(d.get("filename_hint") or filename_hint or "report.pdf")
    document_b64 = str(document_b64 or "")
    clinical_notes = str(clinical_notes or "")
    patient_context_json = str(patient_context_json or "")
    filename_hint = str(filename_hint or "report.pdf")
    _ = extra

    _ensure_lab_report_paths()
    try:
        from inference import run_lab_report_pipeline_b64
    except ImportError as e:
        return {
            "available": False,
            "reason": "import_error",
            "message": str(e),
        }
    try:
        return run_lab_report_pipeline_b64(
            document_b64,
            patient_context_json=patient_context_json or "",
            clinical_notes=clinical_notes or "",
            filename_hint=filename_hint or "report.pdf",
        )
    except Exception as e:
        logger.exception("analyze_lab_report failed: %s", e)
        return {
            "available": False,
            "reason": "inference_error",
            "message": str(e),
        }


def _exec_analyze_dermatology(
    image_b64: str,
    patient_context: Any = None,
    filename_hint: str = "",
) -> dict[str, Any]:
    """Run dermatology vision pipeline; never raises."""
    _ensure_derm_paths()
    try:
        from analyzer import run_dermatology_pipeline_b64
    except ImportError as e:
        return {
            "available": False,
            "reason": "import_error",
            "message": str(e),
        }
    ctx: dict[str, Any] = {}
    if isinstance(patient_context, dict):
        ctx = patient_context
    elif isinstance(patient_context, str) and patient_context.strip():
        try:
            ctx = json.loads(patient_context)
            if not isinstance(ctx, dict):
                ctx = {}
        except json.JSONDecodeError:
            ctx = {}
    try:
        return run_dermatology_pipeline_b64(
            image_b64,
            patient_context=ctx,
            job_id="zeroclaw",
            filename_hint=filename_hint or "",
        )
    except Exception as e:
        logger.exception("analyze_dermatology failed: %s", e)
        return {
            "available": False,
            "reason": "inference_error",
            "message": str(e),
        }




def _exec_analyze_pathology(
    image_b64: str,
    patient_context_json: str = "{}",
    **_kwargs: Any,
) -> dict[str, Any]:
    """Histopathology / WSI tile; never raises."""
    _ensure_pathology_paths()
    try:
        from inference import run_pathology_pipeline_b64
    except ImportError as e:
        return {"available": False, "reason": "import_error", "message": str(e)}
    try:
        ctx: dict[str, Any] = {}
        if patient_context_json and str(patient_context_json).strip():
            ctx = json.loads(patient_context_json)
        if not isinstance(ctx, dict):
            ctx = {}
    except json.JSONDecodeError:
        ctx = {}
    try:
        return run_pathology_pipeline_b64(
            image_b64=image_b64,
            patient_context=ctx,
        )
    except Exception as e:
        logger.exception("analyze_pathology failed: %s", e)
        return {
            "available": False,
            "reason": "inference_error",
            "message": str(e),
        }


def _exec_analyze_cytology(
    image_b64: str,
    patient_context_json: str = "{}",
    **_kwargs: Any,
) -> dict[str, Any]:
    """Cytology smear; never raises."""
    _ensure_cytology_paths()
    try:
        from inference import run_cytology_pipeline_b64
    except ImportError as e:
        return {"available": False, "reason": "import_error", "message": str(e)}
    try:
        ctx: dict[str, Any] = {}
        if patient_context_json and str(patient_context_json).strip():
            ctx = json.loads(patient_context_json)
        if not isinstance(ctx, dict):
            ctx = {}
    except json.JSONDecodeError:
        ctx = {}
    try:
        return run_cytology_pipeline_b64(
            image_b64=image_b64,
            patient_context=ctx,
        )
    except Exception as e:
        logger.exception("analyze_cytology failed: %s", e)
        return {
            "available": False,
            "reason": "inference_error",
            "message": str(e),
        }


def _exec_analyze_mammography(
    image_b64: str,
    patient_context_json: str = "{}",
    **_kwargs: Any,
) -> dict[str, Any]:
    """Mammography: Mirai when 4 views in patient_context; never raises."""
    _ensure_mammography_paths()
    inf_path = os.path.join(_MAMMOGRAPHY, "inference.py")
    try:
        if _MAMMO_INF_MODULE in sys.modules:
            mod = sys.modules[_MAMMO_INF_MODULE]
        else:
            spec = importlib.util.spec_from_file_location(_MAMMO_INF_MODULE, inf_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {inf_path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[_MAMMO_INF_MODULE] = mod
            spec.loader.exec_module(mod)
        run_mammography_pipeline_b64 = mod.run_mammography_pipeline_b64
    except Exception as e:
        return {"available": False, "reason": "import_error", "message": str(e)}
    try:
        ctx: dict[str, Any] = {}
        if patient_context_json and str(patient_context_json).strip():
            ctx = json.loads(patient_context_json)
        if not isinstance(ctx, dict):
            ctx = {}
    except json.JSONDecodeError:
        ctx = {}
    try:
        return run_mammography_pipeline_b64(
            image_b64=image_b64,
            patient_context=ctx,
        )
    except Exception as e:
        logger.exception("analyze_mammography failed: %s", e)
        return {
            "available": False,
            "reason": "inference_error",
            "message": str(e),
        }


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "analyze_oral",
            "description": (
                "Oral cavity / intraoral screening (India: gutka, paan, areca, OSMF). "
                "Uses local EfficientNet weights when deployed, optional UNI for histopathology crops, "
                "else Kimi or Claude vision JSON — never requires a checkpoint to return a structured result. "
                "Biopsy and specialist review are still required for tissue diagnosis."
            ),
            "parameters": {
                "type": "object",
                "required": ["image_b64"],
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": "Base64-encoded oral cavity photo (JPEG/PNG). Phone photos accepted.",
                    },
                    "clinical_notes": {
                        "type": "string",
                        "description": (
                            "Optional clinical context string. Include tobacco/betel use if known. "
                            "Example: 'tobacco_chewing: yes, duration_years: 15, gutka: yes'"
                        ),
                    },
                    "patient_context_json": {
                        "type": "string",
                        "description": (
                            "Optional JSON string: duration_weeks, lesion_site, input_type, habits, etc. "
                            "Merged with clinical_notes for biopsy flags and India-aware narrative."
                        ),
                    },
                    "input_type": {
                        "type": "string",
                        "description": (
                            "Optional hint: clinical_photo | histopathology | mixed | unknown. "
                            "Select histopathology for H&E / biopsy crop to prefer UNI encoder path."
                        ),
                    },
                    "filename_hint": {
                        "type": "string",
                        "description": "Original filename e.g. photo.jpg — used for temp file extension.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_usg",
            "description": (
                "Analyze an ultrasound (sonography) image or cine loop using the Manthana "
                "Ultrasound Engine. Supports abdominal, obstetric, cardiac, and point-of-care "
                "(FAST) ultrasound. Returns organ echogenicity scores, free-fluid indicator, "
                "anomaly proxy score, and a full narrative report with India-specific clinical "
                "context (gallstones, fatty liver, TB peritonitis, hydatid cyst, nephrolithiasis)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": "Base64-encoded ultrasound image (JPEG/PNG/DICOM) or short cine video (MP4).",
                    },
                    "patient_context_json": {
                        "type": "string",
                        "description": (
                            "Optional JSON string with patient context: "
                            '{"age": 45, "sex": "M", "complaint": "RUQ pain", '
                            '"history": "alcohol use disorder", "study_type": "abdominal"}'
                        ),
                    },
                    "job_id": {
                        "type": "string",
                        "description": "Optional job ID for tracing.",
                    },
                },
                "required": ["image_b64"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_oral_cancer",
            "description": (
                "Analyzes oral cavity images for signs of oral cancer, "
                "premalignant lesions (OPMD), oral submucous fibrosis, "
                "and other suspicious oral mucosal conditions. "
                "Accepts clinical photos of the oral cavity or "
                "histopathology / biopsy slide images. "
                "Returns OSCC confidence, OPMD confidence, biopsy "
                "recommendation, lesion location, habit risk assessment, "
                "and India-specific clinical context including tobacco, "
                "betel nut, and gutka exposure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": (
                            "Base64-encoded oral cavity photo or "
                            "histopathology image."
                        ),
                    },
                    "patient_context_json": {
                        "type": "string",
                        "description": (
                            "JSON string with patient context. "
                            "Important fields: age, sex, "
                            "tobacco_habit (gutka/khaini/paan/betel/none), "
                            "habit_duration_years, "
                            "lesion_duration_weeks, "
                            "lesion_site (buccal/tongue/floor/other), "
                            "clinical_history."
                        ),
                    },
                    "image_type": {
                        "type": "string",
                        "enum": [
                            "clinical_photo",
                            "histopathology",
                            "intraoral",
                            "unknown",
                        ],
                        "description": (
                            "Type of image uploaded. "
                            "Default: clinical_photo."
                        ),
                    },
                },
                "required": ["image_b64"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_brain_mri",
            "description": (
                "Brain MRI analysis: TotalSegmentator total_mr whole-body MR volumes (when TotalSeg is installed). "
                "SynthSeg brain parcellation appears in pathology_scores only when SYNTHSEG_SCRIPT exists and the "
                "subprocess succeeds — otherwise findings note SynthSeg unavailable. "
                "Optional Prima diagnostic scores only when PRIMA_CONFIG_YAML and weights are deployed. "
                "Expects base64-encoded NIfTI (.nii/.nii.gz) or DICOM-derived volume; "
                "returns structured findings (list), pathology_scores, structures.*_available flags, and models_used."
            ),
            "parameters": {
                "type": "object",
                "required": ["image_b64"],
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": "Base64-encoded brain MRI volume (e.g. NIfTI).",
                    },
                    "clinical_notes": {
                        "type": "string",
                        "description": "Optional clinical context.",
                    },
                    "patient_context_json": {
                        "type": "string",
                        "description": (
                            "Optional JSON string of patient context "
                            "(age, sex, clinical_history, mri_sequence) merged into clinical_notes."
                        ),
                    },
                    "filename_hint": {
                        "type": "string",
                        "description": "Original filename e.g. brain.nii.gz — sets temp extension.",
                    },
                    "series_dir": {
                        "type": "string",
                        "description": "Optional DICOM series path on shared volume for NIfTI conversion.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_spine_neuro",
            "description": (
                "Spine imaging: TotalSegmentator vertebrae_mr (MRI) or vertebrae_body (CT) per volume. "
                "Returns level-wise segmentation metrics and findings. "
                "Base64-encoded volume (NIfTI or DICOM suffix) required."
            ),
            "parameters": {
                "type": "object",
                "required": ["image_b64"],
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": "Base64-encoded spine CT/MR volume.",
                    },
                    "patient_context": {
                        "type": "string",
                        "description": "Optional clinical context string (legacy).",
                    },
                    "patient_context_json": {
                        "type": "string",
                        "description": "JSON object string, e.g. {\"age\":50,\"clinical_history\":\"...\"}.",
                    },
                    "filename_hint": {
                        "type": "string",
                        "description": "Original filename e.g. spine.nii.gz or study.dcm.",
                    },
                    "series_dir": {
                        "type": "string",
                        "description": "Optional DICOM series directory on shared volume.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_lab_report",
            "description": (
                "Extract and interpret laboratory reports from PDF, image, or text (base64). "
                "Uses optional Parrotlet vision parsing plus LLM interpretation. "
                "Returns findings, pathology_scores, labs, critical_values, models_used. "
                "clinical_notes uses semicolon-separated key:value pairs (e.g. age:45; fasting:yes)."
            ),
            "parameters": {
                "type": "object",
                "required": ["document_b64"],
                "properties": {
                    "document_b64": {
                        "type": "string",
                        "description": "Base64-encoded PDF, JPEG, PNG, TXT, or CSV lab report.",
                    },
                    "patient_context_json": {
                        "type": "string",
                        "description": (
                            "JSON string, e.g. "
                            '{"age":28,"sex":"M","clinical_history":"fever 2 weeks"}.'
                        ),
                    },
                    "clinical_notes": {
                        "type": "string",
                        "description": "Optional semicolon-separated key:value context (legacy).",
                    },
                    "filename_hint": {
                        "type": "string",
                        "description": "Original filename for format detection (e.g. labs.pdf).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_dermatology",
            "description": (
                "Analyse a skin or dermatology image: 12 India-relevant conditions "
                "(tinea, vitiligo, psoriasis, melasma, acne, eczema, scabies, urticaria, "
                "BCC, SCC, melanoma, normal/benign), malignancy screening, and narrative report. "
                "Requires ANTHROPIC_API_KEY. Returns structured findings and pathology_scores."
            ),
            "parameters": {
                "type": "object",
                "required": ["image_b64", "patient_context"],
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": "Base64-encoded skin image (JPEG/PNG/WebP).",
                    },
                    "patient_context": {
                        "type": "object",
                        "description": "Clinical context for interpretation.",
                        "properties": {
                            "age": {"type": "integer"},
                            "sex": {"type": "string", "enum": ["M", "F", "Unknown"]},
                            "duration_weeks": {"type": "number"},
                            "location_body": {"type": "string"},
                            "symptoms": {"type": "string"},
                            "history": {"type": "string"},
                            "prior_treatment": {"type": "string"},
                        },
                    },
                    "filename_hint": {
                        "type": "string",
                        "description": "Original filename for format hints.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_pathology",
            "description": (
                "Analyse histopathology / whole-slide tile or H&E tissue image using Virchow + DSMIL. "
                "Returns structured findings, pathology_scores, structures (dict), and optional narrative. "
                "Use for biopsy, surgical specimen, tissue screening (India: oral, cervical, GI, lung context)."
            ),
            "parameters": {
                "type": "object",
                "required": ["image_b64"],
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": "Base64-encoded JPEG/PNG/TIFF histopathology image or WSI tile.",
                    },
                    "patient_context_json": {
                        "type": "string",
                        "description": (
                            'JSON string with keys such as tissue_source (recommended), stain, age, sex, '
                            "clinical_history, biopsy_site."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_cytology",
            "description": (
                "Analyse cytology smear (Pap, sputum, FNAC, urine, CSF). Virchow + DSMIL with specimen-specific "
                "labels; returns adequacy, Bethesda category when Pap, pathology_scores, structures dict."
            ),
            "parameters": {
                "type": "object",
                "required": ["image_b64"],
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": "Base64 JPEG/PNG cytology image.",
                    },
                    "patient_context_json": {
                        "type": "string",
                        "description": (
                            'JSON string with specimen_type (pap_smear|sputum|fnac|urine|csf|ascitic|other), '
                            "age, sex, fnac_site, clinical_history."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_mammography",
            "description": (
                "Analyse mammography imaging. When patient_context includes four server-side view paths "
                "(L-CC, L-MLO, R-CC, R-MLO), runs Mirai for 1–5 year breast cancer risk probabilities. "
                "Single-image base64 upload yields qualitative assessment only — no fabricated risk scores. "
                "India: breast cancer is common; younger presentation than Western averages — Mirai not validated in India."
            ),
            "parameters": {
                "type": "object",
                "required": ["image_b64"],
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": "Base64 mammogram image (JPEG/PNG).",
                    },
                    "patient_context_json": {
                        "type": "string",
                        "description": (
                            'JSON string: age, birads_density (1–4), family_history, menopausal_status, weight_kg, '
                            'height_cm. For 4-view Mirai: views={"L-CC": path, "L-MLO": path, "R-CC": path, "R-MLO": path}'
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_abdominal_ct",
            "description": (
                "Analyse an abdominal CT volume (DICOM slice or NIfTI). TotalSegmentator organ segmentation "
                "plus optional Comp2Comp body-composition metrics when a full DICOM series directory is available. "
                "Returns findings (list), pathology_scores (organ volumes, aorta, BMD proxies), structures dict."
            ),
            "parameters": {
                "type": "object",
                "required": ["image_b64"],
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": "Base64-encoded CT DICOM or NIfTI (.nii/.nii.gz).",
                    },
                    "patient_context_json": {
                        "type": "string",
                        "description": "JSON string: age, sex, clinical_history, indication (optional; for audit).",
                    },
                    "filename_hint": {
                        "type": "string",
                        "description": "Original filename e.g. study.dcm or volume.nii.gz.",
                    },
                    "series_dir": {
                        "type": "string",
                        "description": "Optional path to full DICOM series on shared volume for Comp2Comp.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_chest_xray",
            "description": (
                "Analyse a chest X-ray (PA/AP portable). TorchXRayVision dual-model ensemble; returns "
                "pathology_scores, structured findings, impression, and optional narrative_report in structures. "
                "India context: TB endemicity, silicosis in occupational exposure when clinically relevant."
            ),
            "parameters": {
                "type": "object",
                "required": ["image_b64"],
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": "Base64-encoded chest radiograph (JPEG/PNG).",
                    },
                    "patient_context_json": {
                        "type": "string",
                        "description": 'JSON string: age, sex, clinical_history, comorbidities.',
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_ecg",
            "description": (
                "Analyse a 12-lead ECG from base64-encoded CSV, EDF, DICOM, or photo (PNG/JPEG) of a printout. "
                "Returns rhythm scores (pathology_scores), interval dict (structures: hr_bpm, pr_ms, qtc_ms, etc.), "
                "and structured findings. CPU signal pipeline (neurokit2 + heuristics). "
                "Use filename_hint e.g. ecg.csv for waveform files."
            ),
            "parameters": {
                "type": "object",
                "required": ["image_b64"],
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": "Base64-encoded ECG file (CSV, EDF, DICOM, or image).",
                    },
                    "patient_context_json": {
                        "type": "string",
                        "description": 'JSON string: age, sex, clinical_history (optional; for future narrative wiring).',
                    },
                    "filename_hint": {
                        "type": "string",
                        "description": "Original filename e.g. ecg.csv or trace.png for format detection.",
                    },
                },
            },
        },
    },
]

TOOL_EXECUTORS: dict[str, Callable[..., dict[str, Any]]] = {
    "analyze_oral": _exec_analyze_oral,
    "analyze_oral_cancer": _exec_analyze_oral_cancer,
    "analyze_brain_mri": _exec_analyze_brain_mri,
    "analyze_spine_neuro": _exec_analyze_spine_neuro,
    "analyze_lab_report": _exec_analyze_lab_report,
    "analyze_dermatology": _exec_analyze_dermatology,
    "analyze_pathology": _exec_analyze_pathology,
    "analyze_cytology": _exec_analyze_cytology,
    "analyze_mammography": _exec_analyze_mammography,
    "analyze_abdominal_ct": _exec_analyze_abdominal_ct,
    "analyze_chest_xray": _exec_analyze_chest_xray,
    "analyze_ecg": _exec_analyze_ecg,
    "analyze_usg": _exec_analyze_usg,
}


def run_tool(name: str, **kwargs: Any) -> dict[str, Any]:
    """Dispatch by tool name (for agents that resolve by string)."""
    fn = TOOL_EXECUTORS.get(name)
    if fn is None:
        return {"available": False, "reason": "unknown_tool", "message": name}
    return fn(**kwargs)
