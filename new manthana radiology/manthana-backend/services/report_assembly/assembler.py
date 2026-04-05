"""
Manthana — Report Assembler
OpenRouter-backed report generation (SSOT: config/cloud_inference.yaml, roles report_assembly / unified_report).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Tuple

from prompts import (
    SUPPORTED_LANGUAGES,
    build_report_prompt,
    build_system_prompt,
    build_unified_report_prompt,
    get_language_config,
)

logger = logging.getLogger("manthana.report_assembly")

try:
    import sys

    sys.path.insert(0, "/app/shared")
    from llm_router import llm_router

    LLM_ROUTER_AVAILABLE = True
except ImportError:
    LLM_ROUTER_AVAILABLE = False
    llm_router = None
    logger.warning("LLM router not available — report generation will fail without it")


async def generate_report(
    modality: str,
    findings: dict,
    structures: list = None,
    detected_region: str = None,
    language: str = "en",
) -> Tuple[str, str, str]:
    """Generate narrative report via OpenRouter (role: report_assembly)."""
    if language not in SUPPORTED_LANGUAGES:
        logger.warning("Unsupported language '%s', falling back to English", language)
        language = "en"

    lang_cfg = get_language_config(language)
    system_prompt = build_system_prompt(language)
    user_prompt = build_report_prompt(modality, findings, structures, detected_region, language)

    logger.info("Generating %s report for modality=%s via OpenRouter", lang_cfg["name"], modality)

    if not LLM_ROUTER_AVAILABLE or llm_router is None:
        return json.dumps(findings, indent=2), "AI report generation unavailable.", "none"

    async def _complete(lang: str, sys_p: str, usr_p: str):
        def _sync():
            return llm_router.complete(
                prompt=usr_p,
                system_prompt=sys_p,
                task_type="report_assembly",
                temperature=0.3,
                max_tokens=2000,
            )

        return await asyncio.to_thread(_sync)

    try:
        result = await _complete(language, system_prompt, user_prompt)
        narrative, impression = _parse_report(result["content"], language)
        if _is_valid_output(narrative, language):
            return narrative, impression, str(result.get("model_used", "openrouter"))
        logger.warning("OpenRouter output failed language validation for %s", language)
    except Exception as e:
        logger.warning("OpenRouter report failed: %s", e)

    if language != "en":
        logger.warning("Falling back to English for modality=%s", modality)
        en_system = build_system_prompt("en")
        en_prompt = build_report_prompt(modality, findings, structures, detected_region, "en")
        try:
            result = await _complete("en", en_system, en_prompt)
            narrative, impression = _parse_report(result["content"], "en")
            return (
                narrative + "\n\n[Report generated in English — translation unavailable]",
                impression,
                str(result.get("model_used", "openrouter")),
            )
        except Exception as e:
            logger.warning("English fallback failed: %s", e)

    logger.error("All report generation attempts failed.")
    return json.dumps(findings, indent=2), "AI report generation unavailable.", "none"


def _parse_report(text: str, language: str = "en") -> Tuple[str, str]:
    """Parse LLM output into (findings, impression) sections."""
    lang_cfg = get_language_config(language)
    findings_hdr = lang_cfg["findings_header"]
    impression_hdr = lang_cfg["impression_header"]

    impression_markers = [
        impression_hdr + ":",
        impression_hdr,
        "IMPRESSION:",
        "IMPRESSION\n",
        "**IMPRESSION**",
        "Summary:",
    ]
    findings_markers = [
        findings_hdr + ":",
        findings_hdr,
        "FINDINGS:",
        "FINDINGS\n",
        "**FINDINGS**",
    ]

    impression_start = -1
    impression_marker_len = 0
    for marker in impression_markers:
        pos = text.find(marker)
        if pos != -1:
            impression_start = pos
            impression_marker_len = len(marker)
            break

    if impression_start != -1:
        findings_raw = text[:impression_start].strip()
        impression = text[impression_start + impression_marker_len :].strip()
        for hdr in findings_markers:
            if findings_raw.startswith(hdr):
                findings_raw = findings_raw[len(hdr) :].strip()
                break
        return findings_raw.strip(), impression.strip()

    findings = text.strip()
    parts = re.split(r"\n\n|(?<=\.)\s+", text, maxsplit=1)
    impression = parts[0].strip() if parts else text[:200]
    return findings, impression


def _is_valid_output(text: str, language: str) -> bool:
    if not text or len(text.strip()) < 50:
        return False
    if language == "en":
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / max(len(text), 1)
    if ratio < 0.05:
        logger.warning("Output for language=%s appears mostly ASCII (ratio=%.2f)", language, ratio)
        return False
    return True


async def generate_unified_report(
    individual_reports: list,
    language: str = "en",
    correlations_block: str = "",
) -> dict:
    """Generate unified cross-modality report via OpenRouter (unified_report role)."""
    if language not in SUPPORTED_LANGUAGES:
        language = "en"

    if not LLM_ROUTER_AVAILABLE or llm_router is None:
        return {
            "unified_diagnosis": "",
            "unified_findings": "",
            "cross_modality_correlations": "",
            "risk_assessment": "",
            "treatment_recommendations": "",
            "prognosis": "",
            "confidence": "low",
            "models_used": [],
            "language": language,
            "error": "llm_router_unavailable",
        }

    prompt = build_unified_report_prompt(individual_reports, language, correlations_block)
    system = build_system_prompt(language)

    def _sync():
        return llm_router.complete(
            prompt=prompt,
            system_prompt=system,
            task_type="unified_report",
            temperature=0.3,
            max_tokens=4096,
        )

    try:
        response = await asyncio.to_thread(_sync)
        full_text = response["content"]
        llm_used = response["model_used"]
        logger.info("Unified report generated with %s", llm_used)
        result = _parse_unified_report(full_text)
        result["models_used"] = [llm_used]
        result["language"] = language
        return result
    except Exception as e:
        logger.error("Unified report failed: %s", e)
        return {
            "unified_diagnosis": "AI unified analysis unavailable.",
            "unified_findings": str(e)[:500],
            "cross_modality_correlations": "",
            "risk_assessment": "",
            "treatment_recommendations": "",
            "prognosis": "",
            "confidence": "low",
            "models_used": [],
            "language": language,
        }


def _parse_unified_report(text: str) -> dict:
    sections = {
        "unified_diagnosis": "",
        "unified_findings": "",
        "cross_modality_correlations": "",
        "risk_assessment": "",
        "treatment_recommendations": "",
        "prognosis": "",
        "confidence": "moderate",
    }
    section_markers = {
        "UNIFIED_DIAGNOSIS:": "unified_diagnosis",
        "UNIFIED_FINDINGS:": "unified_findings",
        "CROSS_MODALITY_CORRELATIONS:": "cross_modality_correlations",
        "RISK_ASSESSMENT:": "risk_assessment",
        "TREATMENT_RECOMMENDATIONS:": "treatment_recommendations",
        "PROGNOSIS:": "prognosis",
        "CONFIDENCE:": "confidence",
    }
    text_upper = text.upper()
    positions = []
    for marker, key in section_markers.items():
        pos = text_upper.find(marker)
        if pos != -1:
            positions.append((pos, pos + len(marker), key))
    positions.sort(key=lambda x: x[0])
    for i, (start_pos, content_start, key) in enumerate(positions):
        end_pos = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        sections[key] = text[content_start:end_pos].strip()
    if not any(sections[k] for k in sections if k != "confidence"):
        sections["unified_diagnosis"] = text.strip()
    return sections
