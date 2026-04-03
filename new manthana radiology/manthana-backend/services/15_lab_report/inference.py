"""
Lab report E2E pipeline: base64 decode → analyze_lab_report → structured enrichment,
deterministic pathology score hints, Kimi K2.5 narrative only.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import re
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("manthana.lab_inference")

_FLAGS = frozenset({"H", "L", "HH", "LL", "CRITICAL"})


def _make_openai_client(api_key: str, base_url: str = None):
    """Safe OpenAI client factory — strips proxies kwarg for openai>=1.0."""
    import httpx
    from openai import OpenAI

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    kwargs["http_client"] = httpx.Client(timeout=60.0, follow_redirects=True)
    return OpenAI(**kwargs)


def _find_shared() -> Optional[str]:
    """Resolve shared package path for Docker (/app/shared) or local dev."""
    for p in ("/app/shared",):
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "schemas.py")):
            return p
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", "shared"))
    if os.path.isdir(root) and os.path.isfile(os.path.join(root, "schemas.py")):
        return root
    return None


def _merge_patient_context(clinical_notes: str, patient_context_json: str) -> Dict[str, Any]:
    from analyzer import parse_clinical_notes

    ctx: Dict[str, Any] = {}
    if clinical_notes and clinical_notes.strip():
        ctx.update(parse_clinical_notes(clinical_notes))
    if patient_context_json and str(patient_context_json).strip():
        try:
            j = json.loads(patient_context_json)
            if isinstance(j, dict):
                ctx.update(j)
        except json.JSONDecodeError:
            logger.warning("patient_context_json not valid JSON; ignoring")
    return ctx


def _first_value_token_index(parts: List[str]) -> int:
    for i, p in enumerate(parts):
        if p in ("Present", "Absent", "REACTIVE", "Non-Reactive"):
            return i
        if re.match(r"^[\d.,]+(?:-[0-9.,]+)?$", p.replace(",", "")):
            return i
    return -1


def extract_test_results_from_text(raw_text: str) -> List[Dict[str, Any]]:
    """Heuristic row parse for Indian-style table text (ReportLab / PyMuPDF)."""
    rows: List[Dict[str, Any]] = []
    if not raw_text or not raw_text.strip():
        return rows
    skip_prefixes = (
        "test ",
        "patient ",
        "name:",
        "age/",
        "ref dr",
        "sample:",
        "collected:",
        "report:",
        "nabl ",
        "results interpreted",
        "for queries:",
    )
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or len(line) < 4:
            continue
        low = line.lower()
        if any(low.startswith(s) for s in skip_prefixes):
            continue
        if "REFERENCE" in line and "RESULT" in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        idx = _first_value_token_index(parts)
        if idx < 1:
            continue
        name = " ".join(parts[:idx]).strip()
        if len(name) < 2 or name.isdigit():
            continue
        value = parts[idx]
        rest = parts[idx + 1 :]
        flag = ""
        if rest and rest[-1] in _FLAGS:
            flag = rest[-1]
            rest = rest[:-1]
        unit = rest[0] if rest else ""
        ref = " ".join(rest[1:]) if len(rest) > 1 else ""
        is_crit = flag == "CRITICAL" or flag == "HH"
        rows.append(
            {
                "test_name": name[:120],
                "value": value,
                "unit": unit[:40],
                "reference_range": ref[:80],
                "flag": flag,
                "is_critical": is_crit,
            }
        )
    return rows


def _report_type_from_text(text: str) -> str:
    if not text:
        return "GENERAL_LAB"
    u = text.upper()
    if "CBC" in u or "COMPLETE BLOOD COUNT" in u:
        return "CBC"
    if "LFT" in u or "LIVER FUNCTION" in u:
        return "LFT"
    if "RFT" in u or "RENAL FUNCTION" in u or "EGFR" in u:
        return "RFT"
    if "IRON" in u or "FERRITIN" in u or "TIBC" in u:
        return "IRON_STUDIES"
    if "TB" in u or "ADA" in u:
        return "TB_WORKUP"
    if "HBA1C" in u or "GLYCATED" in u:
        return "DIABETES_PANEL"
    return "GENERAL_LAB"


def _float_val(s: Any) -> Optional[float]:
    if s is None:
        return None
    t = str(s).replace(",", "").strip()
    m = re.match(r"^([\d.]+)", t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _compute_extended_pathology_scores(
    test_results: List[Dict[str, Any]],
    base_ps: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic 0–1 hints for correlation_engine (merged with LLM scores)."""
    out: Dict[str, Any] = {}
    tn_blob = " ".join((r.get("test_name") or "") for r in test_results).lower()

    tb_hits = 0
    hep_hits = 0
    renal_hits = 0
    dm_hits = 0
    anaemia_hits = 0
    crit_names: List[str] = []
    abnormal = 0
    critical_c = 0

    for r in test_results:
        tn = (r.get("test_name") or "").lower()
        fl = str(r.get("flag") or "").upper()
        val = _float_val(r.get("value"))
        if fl in _FLAGS and fl != "":
            abnormal += 1
        if fl == "CRITICAL" or r.get("is_critical"):
            critical_c += 1
            crit_names.append(str(r.get("test_name") or "unknown")[:60])
        if ("esr" in tn or "crp" in tn or "c-reactive" in tn or "ada" in tn) and fl in (
            "H",
            "HH",
            "L",
            "LL",
            "CRITICAL",
        ):
            tb_hits += 1
        if "lymphocyte" in tn and fl in ("H", "HH"):
            tb_hits += 1
        if "wbc" in tn and "total" in tn.replace(" ", "") and fl in ("L", "LL"):
            tb_hits += 1
        if any(x in tn for x in ("sgot", "ast", "sgpt", "alt", "bilirubin")) and fl in (
            "H",
            "HH",
            "CRITICAL",
        ):
            hep_hits += 1
        if any(x in tn for x in ("creatinine", "urea", "egfr", "bun")) and fl in (
            "H",
            "HH",
            "L",
            "LL",
            "CRITICAL",
        ):
            renal_hits += 1
        if any(x in tn for x in ("hba1c", "glucose", "sugar", "rbs", "fbs")) and fl in (
            "H",
            "HH",
            "CRITICAL",
        ):
            dm_hits += 1
        if ("haemoglobin" in tn or "hemoglobin" in tn or tn.startswith("hb ")) and fl in (
            "L",
            "LL",
            "CRITICAL",
        ):
            anaemia_hits += 1

    out["tb_pattern_confidence"] = min(1.0, 0.12 * tb_hits + 0.08)
    out["hepatic_injury_score"] = min(1.0, 0.2 * hep_hits + 0.05)
    out["renal_impairment_score"] = min(1.0, 0.22 * renal_hits + 0.05)
    out["diabetes_control_score"] = min(1.0, 0.18 * dm_hits + 0.05)
    out["anaemia_severity_score"] = min(1.0, 0.25 * anaemia_hits + 0.05)

    # Boost TB when comment mentions TB / RNTCP
    if "tb" in tn_blob or "rntcp" in tn_blob or "nikshay" in tn_blob:
        out["tb_pattern_confidence"] = min(1.0, float(out["tb_pattern_confidence"]) + 0.25)

    # Prefer max of LLM vs deterministic for these keys (stronger signal wins)
    for k in list(out.keys()):
        try:
            b = float(base_ps.get(k, 0) or 0)
            out[k] = max(float(out[k]), b)
        except (TypeError, ValueError):
            pass

    out["critical_value_present"] = bool(critical_c) or bool(crit_names)
    out["_abnormal_count"] = abnormal
    out["_critical_count"] = critical_c
    out["_critical_names"] = crit_names
    return out


def _patterns_detected(test_results: List[Dict[str, Any]], raw_text: str) -> List[str]:
    t = (raw_text or "").lower()
    pats: List[str] = []
    if "iron deficiency" in t or ("ferritin" in t and "deplet" in t):
        pats.append("iron_deficiency_anaemia")
    if "pancytop" in t or ("lymphocytosis" in t and "esr" in t):
        pats.append("pancytopaenia_tb_pattern")
    if "sgpt" in t or "sgot" in t or "hepatitis" in t or "hbsag" in t:
        pats.append("acute_hepatocellular_injury")
    if "nephropathy" in t or "egfr" in t or ("creatinine" in t and "hba1c" in t):
        pats.append("diabetic_nephropathy_ckd4")
    if "microcytic" in t or "hypochromic" in t:
        pats.append("microcytic_hypochromic")
    if not pats:
        blob = " ".join((r.get("test_name") or "").lower() for r in test_results)
        if "haemoglobin" in blob and not any(x in blob for x in ("sgpt", "esr")):
            pats.append("cbc_review")
    return list(dict.fromkeys(pats))


def _india_context(
    report_type: str,
    patterns: List[str],
    patient_ctx: Dict[str, Any],
    test_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    hist = str(patient_ctx.get("clinical_history") or "").lower()
    ctx: Dict[str, Any] = {
        "report_type": report_type,
        "iron_deficiency_common_india": True,
        "hbsag_prevalence_note": (
            "HBV surface antigen positivity is seen in ~2–4% of Indian adults; "
            "correlate with vaccination history and risk factors."
        ),
        "tb_risk_score": 0.15,
        "malnutrition_flag": "weight loss" in hist or "fever" in hist,
    }
    if "bihar" in hist or "tb contact" in hist:
        ctx["tb_risk_score"] = min(1.0, float(ctx["tb_risk_score"]) + 0.5)
    if "pancytopaenia_tb_pattern" in patterns:
        ctx["tb_risk_score"] = min(1.0, max(float(ctx["tb_risk_score"]), 0.72))
    for r in test_results:
        if "hbsag" in (r.get("test_name") or "").lower() and "reactive" in str(
            r.get("value") or ""
        ).lower():
            ctx["hbsag_reactive_flag"] = True
    return ctx


def _kimi_extra_body(model: str) -> dict | None:
    m = (model or "").lower()
    if "kimi-k2" in m:
        return {"thinking": {"type": "disabled"}}
    return None


def _call_lab_narrative(
    text_summary: str,
    image_b64: str = None,
    patient_context: dict = None,
) -> tuple[str, list]:
    from config import KIMI_API_KEY, KIMI_BASE_URL, KIMI_LAB_MODEL

    kimi_key = (KIMI_API_KEY or "").strip()
    kimi_base = (KIMI_BASE_URL or "https://api.moonshot.ai/v1").strip()
    kimi_model = (KIMI_LAB_MODEL or "kimi-k2.5").strip()
    extra = _kimi_extra_body(kimi_model)

    system_prompt = (
        "You are a senior clinical pathologist in India. "
        "Analyze the lab report and provide a structured clinical interpretation. "
        "Highlight critical values, likely diagnoses, and recommended follow-up. "
        "Include Indian epidemiological context (TB, dengue, anaemia prevalence). "
        "End with: 'This is an AI-assisted interpretation. Clinical correlation required.'"
    )
    _ = patient_context

    if kimi_key and image_b64:
        try:
            from PIL import Image as _PIL

            raw = base64.b64decode(image_b64)
            img = _PIL.open(io.BytesIO(raw)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            jpeg_b64 = base64.b64encode(buf.getvalue()).decode()

            client = _make_openai_client(kimi_key, kimi_base)
            create_kw: dict = {
                "model": kimi_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"},
                            },
                            {"type": "text", "text": text_summary},
                        ],
                    },
                ],
                "max_tokens": 1500,
                "temperature": 0.2,
            }
            if extra is not None:
                create_kw["extra_body"] = extra
            resp = client.chat.completions.create(**create_kw)
            narrative = (resp.choices[0].message.content or "").strip()
            if narrative:
                return narrative, ["Kimi-vision-Lab"]
        except Exception as e:
            logging.getLogger("manthana.lab").warning(f"Kimi vision lab failed: {e}")

    if kimi_key:
        try:
            client = _make_openai_client(kimi_key, kimi_base)
            create_kw = {
                "model": kimi_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_summary},
                ],
                "max_tokens": 1500,
                "temperature": 0.2,
            }
            if extra is not None:
                create_kw["extra_body"] = extra
            resp = client.chat.completions.create(**create_kw)
            narrative = (resp.choices[0].message.content or "").strip()
            if narrative:
                return narrative, ["Kimi-text-Lab"]
        except Exception as e:
            logging.getLogger("manthana.lab").warning(f"Kimi text lab failed: {e}")

    return "", []


def enrich_lab_pipeline_output(
    raw: Dict[str, Any], patient_ctx: Dict[str, Any], image_b64: str = None
) -> Dict[str, Any]:
    """Attach E2E structures, merged pathology_scores, narrative, is_critical."""
    out = dict(raw)
    out.setdefault("status", "complete")
    st_in = out.get("structures")
    st: Dict[str, Any] = dict(st_in) if isinstance(st_in, dict) else {}
    raw_text = st.get("raw_text") or ""
    if not raw_text and isinstance(st_in, dict):
        raw_text = str(st_in.get("raw_text") or "")
    test_results = extract_test_results_from_text(raw_text)
    if not test_results and isinstance(out.get("labs"), dict):
        for k, v in (out.get("labs") or {}).items():
            if isinstance(v, dict) and v.get("value") is not None:
                test_results.append(
                    {
                        "test_name": str(k),
                        "value": str(v.get("value")),
                        "unit": str(v.get("unit") or ""),
                        "reference_range": "",
                        "flag": "",
                        "is_critical": False,
                    }
                )

    base_ps = dict(out.get("pathology_scores") or {})
    ext = _compute_extended_pathology_scores(test_results, base_ps)
    abnormal_count = int(ext.pop("_abnormal_count", 0))
    critical_count = int(ext.pop("_critical_count", 0))
    crit_names = ext.pop("_critical_names", [])
    merged_ps = {**base_ps, **ext}
    out["pathology_scores"] = merged_ps

    report_type = _report_type_from_text(raw_text) or st.get("report_type") or "GENERAL_LAB"
    patterns = _patterns_detected(test_results, raw_text)
    india = _india_context(report_type, patterns, patient_ctx, test_results)

    critical_vals = [r["test_name"] for r in test_results if r.get("is_critical")]
    abnormal_vals = [
        f"{r['test_name']} {r['value']} {r.get('unit', '')} ({r.get('flag', '')})"
        for r in test_results
        if r.get("flag")
    ][:14]
    text_summary = (
        f"Report type: {report_type}\n"
        f"Patient context: {json.dumps(patient_ctx, default=str)[:1200]}\n"
        f"Abnormal values:\n{chr(10).join(abnormal_vals)}\n"
        f"Critical values: {critical_vals}\n"
        f"Patterns detected: {patterns}\n"
        f"India context: {json.dumps(india, default=str)[:1200]}"
    )
    narrative, tags = _call_lab_narrative(
        text_summary=text_summary,
        image_b64=image_b64,
        patient_context=patient_ctx,
    )
    if not narrative.strip():
        narrative = (out.get("impression") or "").strip()

    critical_values = list(
        dict.fromkeys([*crit_names, *(out.get("critical_values") or [])])
    )

    st_out = {
        **st,
        "report_type": report_type,
        "test_results": test_results,
        "critical_values": critical_values,
        "abnormal_count": abnormal_count,
        "critical_count": critical_count,
        "patterns_detected": patterns,
        "india_context": india,
        "narrative_report": narrative,
        "ocr_confidence": st.get("ocr_confidence"),
        "page_count": int(st.get("page_count") or out.get("pages_processed") or 1),
    }
    out["structures"] = st_out
    out["is_critical"] = bool(merged_ps.get("critical_value_present")) or critical_count > 0

    mu = list(out.get("models_used") or [])
    for t in tags:
        if t and t not in mu:
            mu.append(t)
    out["models_used"] = mu

    return out


def run_lab_report_pipeline_b64(
    document_b64: str,
    patient_context_json: str = None,
    clinical_notes: str = None,
    filename_hint: str = "report.pdf",
    job_id: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Decode base64 document, run analysis, return enriched dict.
    On invalid base64 returns {"available": False, ...} for ZeroClaw.
    """
    _ = kwargs
    _ = job_id
    try:
        if not isinstance(document_b64, str):
            return {
                "available": False,
                "reason": "invalid_document_b64",
                "message": "document_b64 must be a base64 string",
                "findings": [],
                "impression": "",
                "pathology_scores": {},
                "structures": {},
                "modality": "lab_report",
            }
        try:
            data = base64.b64decode(document_b64.strip(), validate=True)
        except (binascii.Error, ValueError, TypeError) as e:
            return {
                "available": False,
                "reason": "bad_b64",
                "message": str(e),
                "findings": [],
                "impression": "",
                "pathology_scores": {},
                "structures": {},
                "modality": "lab_report",
            }

        ext = os.path.splitext(filename_hint)[1] or ".pdf"
        job = str(uuid.uuid4())
        fd, path = tempfile.mkstemp(prefix=f"lab_{job}_", suffix=ext)
        try:
            os.write(fd, data)
            os.close(fd)
            from analyzer import analyze_lab_report

            ctx = _merge_patient_context(clinical_notes or "", patient_context_json or "")
            raw = analyze_lab_report(path, patient_context=ctx)
            is_image_input = filename_hint and any(
                filename_hint.lower().endswith(e)
                for e in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
            )
            img_b64 = document_b64 if is_image_input else None
            return enrich_lab_pipeline_output(raw, ctx, image_b64=img_b64)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception as e:
        import traceback

        logging.getLogger("manthana.lab").error(
            f"Lab pipeline failed: {e}\n{traceback.format_exc()}"
        )
        return {
            "modality": "lab_report",
            "status": "complete",
            "findings": [
                {
                    "label": "Lab report parsing incomplete",
                    "severity": "warning",
                    "confidence": 0.0,
                    "region": "lab",
                    "description": (
                        "Could not fully parse this report. "
                        "Please ensure the file is readable text or a clear photo."
                    ),
                }
            ],
            "impression": "Lab report could not be fully parsed. Manual review required.",
            "pathology_scores": {},
            "structures": {
                "narrative_report": "",
                "test_results": [],
                "critical_values": [],
                "abnormal_count": 0,
                "critical_count": 0,
                "india_context": {},
                "parse_error": str(e),
            },
            "labs": {},
            "critical_values": [],
            "is_critical": False,
            "models_used": ["Manthana Lab Engine"],
            "confidence": "low",
            "disclaimer": (
                "AI-assisted tool only. All findings require "
                "clinical correlation and pathologist verification."
            ),
        }
