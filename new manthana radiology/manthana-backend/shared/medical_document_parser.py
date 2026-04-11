"""
Manthana — Medical Document Parser
Purpose-built medical records parsing using Google MedGemma-4B-IT.

Capabilities:
- Lab report parsing (structured extraction)
- Digital prescription parsing
- Document classification
- PII extraction

Architecture:
- Lazy-loaded via ManagedModel pattern (auto-downloads on first use, never at startup)
- MedGemma uses apply_chat_template + AutoModelForImageTextToText (Gemma 3 architecture)
- GPU-accelerated with CPU fallback
- Integrated with ModelMemoryManager for VRAM management
- Returns structured JSON compatible with correlation_engine
"""

import os
import json
import logging
import threading
import re

from typing import Union
from pathlib import Path

from model_loader import ManagedModel, get_hf_token

logger = logging.getLogger("manthana.medical_parser")

_DEVICE = os.getenv("DEVICE", "cuda")

# ─────────────────────────────────────────────────────────────────────────────
# Model configuration — Google MedGemma-4B-IT
# Medically pre-trained on radiology, histopathology, dermatology, EHR data.
# Same 4B footprint as Parrotlet; unrestricted download (accept HF terms once).
# ─────────────────────────────────────────────────────────────────────────────
MEDGEMMA_MODEL_ID  = "google/medgemma-4b-it"
MEDGEMMA_CACHE_NAME = "medgemma-4b-it"
MEDGEMMA_VRAM_GB    = 9.0   # bfloat16 ≈ 8 GB + processor overhead

# Lazy model container — weights NOT loaded at import time
medgemma_model = ManagedModel(
    model_id=MEDGEMMA_MODEL_ID,
    cache_name=MEDGEMMA_CACHE_NAME,
    device=_DEVICE,
    vram_gb=MEDGEMMA_VRAM_GB,
    priority=6,
    # MedGemma uses AutoModelForImageTextToText, not AutoModel
    model_class=None,          # set dynamically in _load_medgemma
    extra_kwargs={
        "torch_dtype": "auto",
        "device_map": "auto",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe download + processor singletons
# ─────────────────────────────────────────────────────────────────────────────
_download_lock        = threading.Lock()
_weights_ready        = False
_snapshot_attempted   = False

_model_lock           = threading.Lock()
_medgemma_model_obj   = None   # actual loaded model instance

_processor_lock       = threading.Lock()
_medgemma_processor   = None


def ensure_weights_downloaded() -> bool:
    """
    Download MedGemma files to the cache dir once, under a process-wide lock.
    At most one snapshot attempt per process; failure falls back gracefully.

    Download order:
      1. ModelScope (modelscope.cn) — no gating, no terms to accept, open download.
      2. HuggingFace (huggingface.co) — fallback if ModelScope unavailable;
         requires accepting MedGemma terms once at huggingface.co/google/medgemma-4b-it
    """
    global _weights_ready, _snapshot_attempted
    if _weights_ready:
        return True
    with _download_lock:
        if _weights_ready:
            return True
        if _snapshot_attempted:
            return False
        _snapshot_attempted = True
        os.makedirs(medgemma_model.cache_dir, exist_ok=True)

        # ── 1. ModelScope (preferred — no gating) ────────────────────────────
        try:
            from modelscope.hub.snapshot_download import snapshot_download as ms_download

            logger.info("Downloading MedGemma from ModelScope …")
            ms_download(
                model_id=MEDGEMMA_MODEL_ID,
                cache_dir=os.path.dirname(medgemma_model.cache_dir),
            )
            _weights_ready = True
            logger.info("MedGemma weights ready (ModelScope) → %s", medgemma_model.cache_dir)
            return True
        except Exception as exc:
            logger.warning("ModelScope download failed (%s), trying HuggingFace …", exc)

        # ── 2. HuggingFace fallback ───────────────────────────────────────────
        try:
            from huggingface_hub import snapshot_download as hf_download

            tok = get_hf_token()
            kwargs: dict = {
                "repo_id": MEDGEMMA_MODEL_ID,
                "local_dir": medgemma_model.cache_dir,
            }
            if tok:
                kwargs["token"] = tok
            logger.info("Downloading MedGemma from HuggingFace …")
            hf_download(**kwargs)
            _weights_ready = True
            logger.info("MedGemma weights ready (HuggingFace) → %s", medgemma_model.cache_dir)
            return True
        except Exception as exc:
            logger.warning("HuggingFace download also failed: %s", exc)
            return False


def _get_model():
    """Lazy-load the MedGemma model (thread-safe; first call triggers download+load)."""
    global _medgemma_model_obj
    if _medgemma_model_obj is not None:
        return _medgemma_model_obj
    with _model_lock:
        if _medgemma_model_obj is not None:
            return _medgemma_model_obj
        ensure_weights_downloaded()
        try:
            import torch
            from transformers import AutoModelForImageTextToText

            tok = get_hf_token()
            kwargs: dict = {
                "cache_dir": medgemma_model.cache_dir,
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            }
            if tok:
                kwargs["token"] = tok

            logger.info("Loading MedGemma model from %s …", medgemma_model.cache_dir)
            _medgemma_model_obj = AutoModelForImageTextToText.from_pretrained(
                MEDGEMMA_MODEL_ID, **kwargs
            )
            logger.info("MedGemma model loaded.")
        except Exception as exc:
            logger.warning("MedGemma model load failed: %s", exc)
            _medgemma_model_obj = None
    return _medgemma_model_obj


def _get_processor():
    """Lazy-load the MedGemma processor (thread-safe singleton)."""
    global _medgemma_processor
    if _medgemma_processor is not None:
        return _medgemma_processor
    with _processor_lock:
        if _medgemma_processor is not None:
            return _medgemma_processor
        try:
            from transformers import AutoProcessor

            tok = get_hf_token()
            kwargs: dict = {"cache_dir": medgemma_model.cache_dir}
            if tok:
                kwargs["token"] = tok
            logger.info("Loading MedGemma processor …")
            _medgemma_processor = AutoProcessor.from_pretrained(
                MEDGEMMA_MODEL_ID, **kwargs
            )
            logger.info("MedGemma processor loaded.")
        except Exception as exc:
            logger.warning("MedGemma processor load failed: %s", exc)
            _medgemma_processor = None
    return _medgemma_processor


def is_loaded() -> bool:
    """True only when MedGemma model is already resident in memory."""
    return _medgemma_model_obj is not None


# ─────────────────────────────────────────────────────────────────────────────
# India-focused prompts for MedGemma
# ─────────────────────────────────────────────────────────────────────────────

_LAB_SYSTEM = (
    "You are an expert clinical pathologist working in India. "
    "You will receive an image of a lab report from an Indian diagnostic laboratory. "
    "Indian labs commonly abbreviate: Hb=Haemoglobin, TLC=Total Leucocyte Count, "
    "DLC=Differential Leucocyte Count, ESR=Erythrocyte Sedimentation Rate, "
    "Sr.=Serum, U/E=Urea/Electrolytes, LFT=Liver Function Test, "
    "RFT=Renal Function Test, TSH=Thyroid Stimulating Hormone, "
    "HbA1c=Glycated Haemoglobin, SGPT/ALT, SGOT/AST, Sr.Creatinine, "
    "Sr.Uric Acid, Sr.Cholesterol, VLDL, LDL, HDL, TG=Triglycerides. "
    "Reference ranges may use Indian population norms (slightly differ from Western). "
    "Extract ALL test results precisely, including flagged values (H, L, HH, LL, *)."
)

_LAB_USER = (
    "Parse this Indian lab report image completely. Extract every test result with its value, unit, "
    "reference range, and abnormal flag. Return ONLY valid JSON:\n"
    '{\n'
    '  "report_date": "DD/MM/YYYY or null",\n'
    '  "lab_name": "string or null",\n'
    '  "patient": {"name": "...", "age": "...", "gender": "..."},\n'
    '  "tests": [\n'
    '    {"name": "test name", "value": "numeric or string", "unit": "unit", '
    '"ref_range": "low-high or text", "flag": "H/L/HH/LL/CRITICAL or null"}\n'
    '  ],\n'
    '  "document_type": "lab_report"\n'
    '}'
)

_RX_USER = (
    "Parse this Indian prescription image completely. Return ONLY valid JSON:\n"
    '{\n'
    '  "doctor": {"name": "...", "reg_no": "...", "clinic": "...", "speciality": "..."},\n'
    '  "patient": {"name": "...", "age": "...", "gender": "..."},\n'
    '  "date": "DD/MM/YYYY or null",\n'
    '  "medicines": [\n'
    '    {"name": "...", "dosage": "...", "frequency": "...", "duration": "...", "route": "oral/topical/etc"}\n'
    '  ],\n'
    '  "instructions": "...",\n'
    '  "diagnosis": "... or null"\n'
    '}'
)

_DISCHARGE_USER = (
    "Parse this Indian hospital discharge summary completely. Return ONLY valid JSON:\n"
    '{\n'
    '  "hospital": "...",\n'
    '  "admission_date": "DD/MM/YYYY or null",\n'
    '  "discharge_date": "DD/MM/YYYY or null",\n'
    '  "diagnoses": {"primary": "...", "secondary": []},\n'
    '  "procedures": [],\n'
    '  "discharge_medications": [\n'
    '    {"name": "...", "dosage": "...", "frequency": "...", "duration": "..."}\n'
    '  ],\n'
    '  "follow_up": "...",\n'
    '  "condition_at_discharge": "..."\n'
    '}'
)


def _build_messages(document_type: str, image) -> list:
    """Build MedGemma chat-template messages for the given doc type."""
    system_text = _LAB_SYSTEM
    if document_type == "lab_report":
        user_text = _LAB_USER
    elif document_type == "prescription":
        user_text = _RX_USER
    elif document_type == "discharge_summary":
        user_text = _DISCHARGE_USER
    else:
        user_text = (
            "Parse this medical document image completely and return all information as valid JSON."
        )

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_text},
            ],
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Core parse function
# ─────────────────────────────────────────────────────────────────────────────

def parse_medical_document(
    file_path: Union[str, Path],
    document_type: str = "auto",
    extract_pii: bool = False,
) -> dict:
    """
    Parse a medical document (PDF or image) using Google MedGemma-4B-IT.

    Lazy: MedGemma loads only when this function is first called.
    Falls back to PyMuPDF text extraction if model unavailable.

    Returns dict compatible with correlation_engine:
        document_type, structured, labs, entities, confidence,
        raw_text, pages_processed, models_used
    """
    from PIL import Image

    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    # ── Text / CSV → no vision needed ────────────────────────────────────────
    if ext in (".txt", ".csv", ".tsv"):
        return _parse_text_fallback(file_path, document_type)

    # ── Load images ──────────────────────────────────────────────────────────
    if ext == ".pdf":
        images = _pdf_to_images(file_path)
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        images = [Image.open(file_path).convert("RGB")]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not images:
        # PDF→image conversion failed; try text extraction
        raw_text = _extract_pdf_text_fallback(file_path)
        return _make_fallback_response(document_type, raw_text, "pdf_to_image_failed")

    # ── Resolve document type from first page if "auto" ──────────────────────
    if document_type == "auto":
        document_type = _guess_doc_type_from_filename(file_path)

    # ── Lazy-load model + processor ──────────────────────────────────────────
    logger.info("Parsing medical document: %s (type=%s)", file_path.name, document_type)
    model     = _get_model()
    processor = _get_processor()

    if model is None or processor is None:
        raw_text = _extract_pdf_text_fallback(file_path) if ext == ".pdf" else ""
        return _make_fallback_response(document_type, raw_text, "medgemma_unavailable")

    # ── Run inference per page ────────────────────────────────────────────────
    import torch

    all_results: list[str] = []
    for img in images:
        try:
            messages = _build_messages(document_type, img)
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )
            generation = generation[0][input_len:]
            result_text = processor.decode(generation, skip_special_tokens=True)
            all_results.append(result_text)
        except Exception as exc:
            logger.warning("MedGemma inference error on page: %s", exc)
            all_results.append("")

    merged_result = _merge_page_results(all_results)
    structured    = _extract_json_from_response(merged_result)

    return {
        "document_type": structured.get("document_type", document_type),
        "structured":    structured,
        "labs":          structured.get("tests") or structured.get("labs") or {},
        "entities":      structured.get("entities", []),
        "confidence":    0.88,
        "raw_text":      merged_result,
        "pages_processed": len(images),
        "models_used":   [MEDGEMMA_MODEL_ID],
    }


def parse_lab_report(file_path: Union[str, Path]) -> dict:
    """
    Specialized lab report parser — returns structured data for correlation_engine.
    Entry point called by analyzer.py.
    """
    result = parse_medical_document(file_path, document_type="lab_report")

    labs = result.get("labs") or {}
    if not isinstance(labs, dict):
        labs = {}

    # Normalise tests list → dict keyed by test name for correlation rules
    tests_list = result.get("structured", {}).get("tests") or []
    if isinstance(tests_list, list) and tests_list:
        labs_dict: dict = {}
        for t in tests_list:
            if isinstance(t, dict) and t.get("name"):
                key = t["name"].strip().lower().replace(" ", "_")
                labs_dict[key] = {
                    "value":     t.get("value"),
                    "unit":      t.get("unit"),
                    "ref_range": t.get("ref_range"),
                    "flag":      t.get("flag"),
                }
        if labs_dict:
            labs = labs_dict

    # Flatten numeric values for correlation_engine
    flattened: dict = {}
    for k, v in labs.items():
        if isinstance(v, dict):
            raw_val = v.get("value")
        else:
            raw_val = v
        try:
            flattened[k] = float(str(raw_val).replace(",", ""))
        except (ValueError, TypeError):
            flattened[k] = raw_val

    result["structured"]    = labs
    result["flattened_labs"] = flattened
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _guess_doc_type_from_filename(path: Path) -> str:
    name = path.stem.lower()
    if any(k in name for k in ("lab", "report", "test", "blood", "urine", "cbc", "lft", "rft")):
        return "lab_report"
    if any(k in name for k in ("rx", "prescription", "presc")):
        return "prescription"
    if any(k in name for k in ("discharge", "summary", "disch")):
        return "discharge_summary"
    return "lab_report"   # safest default for Manthana's primary use-case


def _pdf_to_images(pdf_path: Path) -> list:
    """Convert PDF pages to PIL Images at 200 DPI (better for small text in lab reports)."""
    try:
        import fitz
        from PIL import Image

        images = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                mat = fitz.Matrix(200 / 72, 200 / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
        return images
    except Exception as exc:
        logger.error("PDF→image conversion failed: %s", exc)
        return []


def _extract_pdf_text_fallback(file_path: Path) -> str:
    try:
        import fitz
        with fitz.open(file_path) as doc:
            return "\n".join(p.get_text("text") for p in doc).strip()
    except Exception:
        return ""


def _parse_text_fallback(file_path: Path, document_type: str) -> dict:
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = ""
    return {
        "document_type": document_type,
        "structured":    {"raw_text": text},
        "labs":          {},
        "entities":      [],
        "confidence":    0.4,
        "raw_text":      text,
        "pages_processed": 1,
        "models_used":   ["text_fallback"],
        "note":          "Text file — vision model not used.",
    }


def _make_fallback_response(document_type: str, raw_text: str, reason: str) -> dict:
    return {
        "document_type": document_type,
        "structured":    {"raw_text": raw_text},
        "labs":          {},
        "entities":      [],
        "confidence":    0.3,
        "raw_text":      raw_text,
        "pages_processed": 1,
        "models_used":   ["text_fallback"],
        "note":          f"MedGemma unavailable ({reason}); used PyMuPDF text fallback.",
    }


def _merge_page_results(results: list) -> str:
    if not results:
        return ""
    if len(results) == 1:
        return results[0]
    return "\n\n".join(f"--- Page {i+1} ---\n{r}" for i, r in enumerate(results))


def _extract_json_from_response(text: str) -> dict:
    """Extract first valid JSON object from model response (handles markdown fences)."""
    # 1. markdown fences
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. first { … } block
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1

    return {"raw_parsed": text[:2000], "document_type": "unknown"}


def classify_document(file_path: Union[str, Path]) -> str:
    result = parse_medical_document(file_path, document_type="auto")
    return result.get("document_type", "unknown")


def medgemma_multimodal_generate(
    messages: list,
    *,
    max_new_tokens: int = 2048,
) -> str:
    """
    Run Google MedGemma-4B-IT on a multimodal chat (same template path as lab parsing).

    ``messages`` must follow the Gemma chat format used elsewhere in this module, e.g.::

        [
          {"role": "system", "content": [{"type": "text", "text": "..."}]},
          {"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": "..."}]},
        ]

    Returns decoded assistant text (not JSON-validated).
    """
    import torch

    model = _get_model()
    processor = _get_processor()
    if model is None or processor is None:
        raise RuntimeError("MedGemma model or processor unavailable")

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generation = generation[0][input_len:]
    return processor.decode(generation, skip_special_tokens=True)
