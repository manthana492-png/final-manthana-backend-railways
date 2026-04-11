"""MedGemma CXR middle-layer: TXRV scores + image + patient context → structured JSON + follow-up questions."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from PIL import Image

_CXR_SYSTEM = (
    "You are an expert thoracic radiologist. You see a chest X-ray image together with "
    "TorchXRayVision / CheXpert-style ensemble probability scores (0–1) and optional clinical context. "
    "You do NOT replace the final signed radiology report. You prepare a structured draft and "
    "3–5 focused questions the clinician or patient should answer before a definitive narrative is written. "
    "Be concise, safety-aware, and avoid over-calling findings not supported by the image or scores."
)


def _scores_block(pathology_scores: Dict[str, Any]) -> str:
    try:
        return json.dumps(pathology_scores, indent=2, ensure_ascii=False)[:12000]
    except Exception:
        return str(pathology_scores)[:12000]


def _context_block(patient_context: Dict[str, Any]) -> str:
    if not patient_context:
        return "(none provided)"
    try:
        return json.dumps(patient_context, indent=2, ensure_ascii=False)[:8000]
    except Exception:
        return str(patient_context)[:8000]


def _parse_json_loose(text: str) -> Dict[str, Any]:
    from medical_document_parser import _extract_json_from_response  # type: ignore

    return _extract_json_from_response(text)


def build_stage1_user_prompt(pathology_scores: Dict[str, Any], patient_context: Dict[str, Any]) -> str:
    return (
        "Below are automated chest-X-ray model scores (probabilities / logits summary) and optional patient context.\n\n"
        "## Pathology / model scores (JSON)\n"
        f"{_scores_block(pathology_scores)}\n\n"
        "## Patient / clinical context (JSON or text)\n"
        f"{_context_block(patient_context)}\n\n"
        "Study the attached chest radiograph carefully in light of the scores and context.\n\n"
        "Return ONLY valid JSON with this exact shape:\n"
        "{\n"
        '  "impression_draft": "short bullet-style draft (not a formal report)",\n'
        '  "key_observations": ["...", "..."],\n'
        '  "uncertainties": ["what you cannot decide from this single view"],\n'
        '  "safety_flags": ["urgent patterns to rule out clinically if suspected — empty if none"],\n'
        '  "follow_up_questions": [\n'
        '    {"id": "q1", "question": "plain language question", "why_needed": "one sentence"},\n'
        "    ... 3 to 5 objects total, ids q1..q5\n"
        "  ]\n"
        "}\n"
        "Questions must be the minimum set needed to refine differential and reporting (e.g. symptoms, "
        "fever, cough duration, TB exposure, smoking, prior films, pregnancy, occupational exposure, "
        "travel, immunosuppression). Do not ask for identifying information beyond what is already in context."
    )


def run_medgemma_stage1(
    image: Image.Image,
    pathology_scores: Dict[str, Any],
    patient_context: Dict[str, Any],
    *,
    max_new_tokens: int = 1536,
) -> Dict[str, Any]:
    from medical_document_parser import medgemma_multimodal_generate  # type: ignore

    user_text = build_stage1_user_prompt(pathology_scores, patient_context)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": _CXR_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    raw = medgemma_multimodal_generate(messages, max_new_tokens=max_new_tokens)
    parsed = _parse_json_loose(raw)
    if not isinstance(parsed, dict):
        parsed = {"raw": raw[:4000]}
    parsed["_raw_medgemma"] = raw[:8000]
    return parsed


def build_kimi_user_payload(
    pathology_scores: Dict[str, Any],
    patient_context: Dict[str, Any],
    medgemma_stage1: Dict[str, Any],
    answers: Dict[str, str],
) -> str:
    return (
        "You are writing the final patient-facing chest X-ray interpretation for Manthana Labs.\n"
        "Ground everything in: (1) the attached image, (2) the TorchXRayVision-style scores JSON, "
        "(3) patient context, (4) MedGemma structured draft + follow-up Q&A.\n"
        "Use clear sections: Summary, Findings, Impression, Recommendations, Limitations.\n"
        "If the user skipped questions, state reasonable assumptions and encourage clinical follow-up.\n"
        "Do not invent priors not supported by the image or scores. Non-diagnostic disclaimer at end.\n\n"
        "## Pathology scores (JSON)\n"
        f"{_scores_block(pathology_scores)}\n\n"
        "## Patient context\n"
        f"{_context_block(patient_context)}\n\n"
        "## MedGemma structured output (JSON)\n"
        f"{json.dumps({k: v for k, v in medgemma_stage1.items() if not str(k).startswith('_')}, indent=2, ensure_ascii=False)[:12000]}\n\n"
        "## User answers (question id → answer; empty string = skipped)\n"
        f"{json.dumps(answers, indent=2, ensure_ascii=False)[:8000]}\n\n"
        "Produce the final report as plain Markdown text (not JSON)."
    )


def normalize_answers_from_payload(
    questions: List[Dict[str, Any]],
    answers_payload: Any,
    skip_all: bool,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    ids = []
    for q in questions:
        if isinstance(q, dict) and q.get("id"):
            ids.append(str(q["id"]))
    if skip_all:
        return {i: "" for i in ids}
    if isinstance(answers_payload, dict):
        for i in ids:
            v = answers_payload.get(i)
            if v is None:
                out[i] = ""
            else:
                out[i] = str(v).strip()
        return out
    return {i: "" for i in ids}
