"""
Manthana — Report Assembler
Multi-LLM report generation with cascading fallback.
Primary: Gemini 2.0 Flash Lite → Groq (GROQ_MODEL, default llama-3.3-70b-versatile) → DeepSeek V3 → Qwen / English fallback.
Supports 10 Indian languages natively.
"""

import os
import re
import json
import logging
import httpx
from typing import Tuple, Optional

from prompts import (
    build_report_prompt, build_system_prompt, build_unified_report_prompt,
    get_language_config, SUPPORTED_LANGUAGES
)

logger = logging.getLogger("manthana.report_assembly")

# Import LLM router for intelligent routing
try:
    import sys
    sys.path.insert(0, "/app/shared")
    from llm_router import llm_router, generate_unified_report as _router_unified_report
    LLM_ROUTER_AVAILABLE = True
except ImportError:
    LLM_ROUTER_AVAILABLE = False
    llm_router = None
    logger.warning("LLM router not available, using legacy cascading fallback")

# Legacy LLM Configuration (fallback)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.5")
# Groq production models on free/dev tier (see https://console.groq.com/docs/models):
#   llama-3.3-70b-versatile (default, strong for reports)
#   llama-3.1-8b-instant (faster, higher throughput)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Dravidian languages that benefit most from Gemini (better training data)
GEMINI_PREFERRED_LANGS = {"ta", "te", "kn", "ml"}


async def generate_report(
    modality: str,
    findings: dict,
    structures: list = None,
    detected_region: str = None,
    language: str = "en",
) -> Tuple[str, str, str]:
    """Generate narrative report with language-aware cascading LLM fallback.

    Args:
        modality: Imaging modality (xray, brain_mri, etc.)
        findings: AI analysis findings dict
        structures: Detected structures list
        detected_region: Auto-detected body region
        language: ISO language code — en, hi, ta, te, kn, ml, mr, bn, gu, pa

    Returns:
        (narrative_text, impression_text, llm_model_used)
    """
    # Validate language
    if language not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language '{language}', falling back to English")
        language = "en"

    lang_cfg = get_language_config(language)
    system_prompt = build_system_prompt(language)
    user_prompt = build_report_prompt(modality, findings, structures, detected_region, language)

    logger.info(f"Generating {lang_cfg['name']} report for modality={modality}")

    # ── Gemini 2.0 Flash Lite — primary (all languages) ──
    if GEMINI_API_KEY:
        try:
            narrative, impression = await _call_gemini(
                system=system_prompt,
                prompt=user_prompt,
                language=language,
            )
            if _is_valid_output(narrative, language):
                return narrative, impression, "gemini-2.0-flash-lite"
            logger.warning(f"Gemini output failed language validation for {language}")
        except Exception as e:
            logger.warning(f"Gemini 2.0 Flash Lite failed: {e}")

    # ── Groq — immediate fallback if Gemini fails (free tier: llama-3.3-70b-versatile, llama-3.1-8b-instant, …) ──
    if GROQ_API_KEY:
        try:
            narrative, impression = await _call_openai_compatible(
                base_url=GROQ_BASE_URL,
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                system=system_prompt,
                prompt=user_prompt,
                language=language,
            )
            if _is_valid_output(narrative, language):
                return narrative, impression, f"groq-{GROQ_MODEL.replace('/', '-')}"
        except Exception as e:
            logger.warning(f"Groq failed: {e}")

    # ── DeepSeek V3 — after Groq ──
    if DEEPSEEK_API_KEY:
        try:
            narrative, impression = await _call_openai_compatible(
                base_url=DEEPSEEK_BASE_URL,
                api_key=DEEPSEEK_API_KEY,
                model="deepseek-chat",
                system=system_prompt,
                prompt=user_prompt,
                language=language,
            )
            if _is_valid_output(narrative, language):
                return narrative, impression, "deepseek-v3"
            logger.warning(f"DeepSeek output failed language validation for {language}")
        except Exception as e:
            logger.warning(f"DeepSeek failed: {e}")

    # ── Qwen-Max — strong Indic language support ──
    if QWEN_API_KEY:
        try:
            narrative, impression = await _call_openai_compatible(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=QWEN_API_KEY,
                model="qwen-max",
                system=system_prompt,
                prompt=user_prompt,
                language=language,
            )
            if _is_valid_output(narrative, language):
                return narrative, impression, "qwen-2.5-max"
        except Exception as e:
            logger.warning(f"Qwen failed: {e}")

    # ── Final fallback: Generate in English if non-English failed ──
    if language != "en":
        logger.warning(f"All providers failed for {language}, falling back to English")
        en_system = build_system_prompt("en")
        en_prompt = build_report_prompt(modality, findings, structures, detected_region, "en")
        for provider_fn, kwargs in [
            (_call_openai_compatible, {"base_url": GROQ_BASE_URL, "api_key": GROQ_API_KEY, "model": GROQ_MODEL}) if GROQ_API_KEY else (None, None),
            (_call_openai_compatible, {"base_url": DEEPSEEK_BASE_URL, "api_key": DEEPSEEK_API_KEY, "model": "deepseek-chat"}) if DEEPSEEK_API_KEY else (None, None),
        ]:
            if provider_fn is None:
                continue
            try:
                narrative, impression = await provider_fn(
                    system=en_system, prompt=en_prompt, language="en", **kwargs
                )
                return (
                    narrative + "\n\n[Report generated in English — translation unavailable]",
                    impression,
                    "fallback-en",
                )
            except Exception:
                continue

    logger.error("All LLM providers failed. Returning raw findings.")
    return json.dumps(findings, indent=2), "AI report generation unavailable.", "none"


async def _call_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    prompt: str,
    language: str = "en",
) -> Tuple[str, str]:
    """Call any OpenAI-compatible API (DeepSeek, Groq, Qwen, etc.)"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
        )
    response.raise_for_status()
    data = response.json()
    full_text = data["choices"][0]["message"]["content"]
    return _parse_report(full_text, language)


async def _call_gemini(
    system: str,
    prompt: str,
    language: str = "en",
) -> Tuple[str, str]:
    """Call Google Gemini API (default: gemini-2.0-flash-lite)."""
    combined = f"{system}\n\n{prompt}"
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{url}?key={GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [{"text": combined}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2000},
            },
        )
    response.raise_for_status()
    data = response.json()
    full_text = data["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_report(full_text, language)


def _parse_report(text: str, language: str = "en") -> Tuple[str, str]:
    """Parse LLM output into (findings, impression) sections.

    Tries native-language headers first, then English fallback.
    """
    lang_cfg = get_language_config(language)
    findings_hdr = lang_cfg["findings_header"]
    impression_hdr = lang_cfg["impression_header"]

    # Build candidate markers (native + English fallbacks)
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

    # Find impression section
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
        impression = text[impression_start + impression_marker_len:].strip()

        # Strip findings header from findings_raw
        for hdr in findings_markers:
            if findings_raw.startswith(hdr):
                findings_raw = findings_raw[len(hdr):].strip()
                break

        return findings_raw.strip(), impression.strip()
    else:
        # No clear section split — use full text as findings, first "sentence" as impression
        findings = text.strip()
        # Split on first double newline or period
        parts = re.split(r"\n\n|(?<=\.)\s+", text, maxsplit=1)
        impression = parts[0].strip() if parts else text[:200]
        return findings, impression


def _is_valid_output(text: str, language: str) -> bool:
    """Basic validation: check the output isn't empty or wrong script.

    For non-English, verify at least some non-ASCII characters are present
    (crude check that the model didn't just respond in English).
    """
    if not text or len(text.strip()) < 50:
        return False
    if language == "en":
        return True

    # Count non-ASCII characters — for Indic scripts, expect >10%
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / max(len(text), 1)
    if ratio < 0.05:
        logger.warning(f"Output for language={language} appears to be mostly ASCII (ratio={ratio:.2f})")
        return False
    return True


# ══════════════════════════════════════════════════════════
# Unified Report (multi-modality)
# ══════════════════════════════════════════════════════════

async def generate_unified_report(
    individual_reports: list,
    language: str = "en",
    correlations_block: str = "",
) -> dict:
    """Generate a unified cross-modality report with intelligent LLM routing.

    Args:
        individual_reports: List of dicts with modality, findings_summary, impression
        language: Target language code
        correlations_block: Pre-matched correlation patterns (from correlation engine)

    Returns:
        Dict with structured unified analysis sections
    """
    if language not in SUPPORTED_LANGUAGES:
        language = "en"

    # Use LLM router if available (intelligent routing to Kimi for long context)
    if LLM_ROUTER_AVAILABLE and llm_router:
        try:
            logger.info("Using LLM router for unified report (Gemini → Groq → DeepSeek → …)")
            
            # Build the prompt
            prompt = build_unified_report_prompt(individual_reports, language, correlations_block)
            system = build_system_prompt(language)
            
            # Use router - will select Kimi for long prompts (unified reports)
            response = llm_router.complete(
                prompt=prompt,
                system_prompt=system,
                task_type="unified_report",
                temperature=0.3,
                max_tokens=4096,
            )
            
            full_text = response["content"]
            llm_used = response["model_used"]
            
            logger.info(f"Unified report generated with {llm_used}")
            
            result = _parse_unified_report(full_text)
            result["models_used"] = [llm_used]
            result["language"] = language
            return result
            
        except Exception as e:
            logger.warning(f"LLM router failed for unified report: {e}. Falling back to legacy.")
    
    # Legacy cascading fallback: Gemini → Groq → DeepSeek → Kimi → Qwen
    prompt = build_unified_report_prompt(individual_reports, language, correlations_block)
    system = build_system_prompt(language)

    full_text = None
    llm_used = "none"

    # Gemini 2.0 Flash Lite first
    if GEMINI_API_KEY:
        try:
            full_text = await _call_llm_for_unified_gemini(system, prompt)
            if _is_valid_output(full_text, language):
                llm_used = "gemini-2.0-flash-lite"
            else:
                full_text = None
        except Exception as e:
            logger.warning(f"Gemini failed for unified report: {e}")

    # Groq second (same tier as Gemini fallback in llm_router)
    if full_text is None and GROQ_API_KEY:
        try:
            full_text = await _call_llm_for_unified(
                GROQ_BASE_URL, GROQ_API_KEY, GROQ_MODEL, system, prompt
            )
            if _is_valid_output(full_text, language):
                llm_used = f"groq-{GROQ_MODEL.replace('/', '-')}"
            else:
                full_text = None
        except Exception as e:
            logger.warning(f"Groq failed for unified report: {e}")

    # DeepSeek V3 third
    if full_text is None and DEEPSEEK_API_KEY:
        try:
            full_text = await _call_llm_for_unified(
                DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, "deepseek-chat", system, prompt
            )
            if _is_valid_output(full_text, language):
                llm_used = "deepseek-v3"
            else:
                full_text = None
        except Exception as e:
            logger.warning(f"DeepSeek failed for unified report: {e}")

    # Kimi 2.5 (long context)
    if full_text is None and KIMI_API_KEY:
        try:
            full_text = await _call_llm_for_unified(
                KIMI_BASE_URL,
                KIMI_API_KEY,
                KIMI_MODEL,
                system,
                prompt,
            )
            if _is_valid_output(full_text, language):
                llm_used = "kimi-2.5"
            else:
                full_text = None
        except Exception as e:
            logger.warning(f"Kimi failed for unified report: {e}")

    # Qwen
    if full_text is None and QWEN_API_KEY:
        try:
            full_text = await _call_llm_for_unified(
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                QWEN_API_KEY, "qwen-max", system, prompt
            )
            llm_used = "qwen-2.5-max"
        except Exception as e:
            logger.warning(f"Qwen failed for unified report: {e}")

    if full_text is None:
        logger.error("All LLM providers failed for unified report.")
        full_text = "AI unified analysis unavailable. Please review individual reports."

    result = _parse_unified_report(full_text)
    result["models_used"] = [llm_used]
    result["language"] = language
    return result


async def _call_llm_for_unified(
    base_url: str, api_key: str, model: str, system: str, prompt: str
) -> str:
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4000,
    }
    if "kimi-k2" in model.lower():
        payload["thinking"] = {"type": "disabled"}
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


async def _call_llm_for_unified_gemini(system: str, prompt: str) -> str:
    combined = f"{system}\n\n{prompt}"
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            f"{url}?key={GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [{"text": combined}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4000},
            },
        )
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def _parse_unified_report(text: str) -> dict:
    """Parse LLM output into structured unified report sections."""
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
