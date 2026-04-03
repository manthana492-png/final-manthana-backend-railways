"""
Manthana — Intelligent LLM Router
Primary: Google Gemini 2.0 Flash Lite (direct REST).
If Gemini fails: Groq (default llama-3.3-70b-versatile — on Groq free tier), then DeepSeek V3, then Kimi.

Usage:
    from llm_router import llm_router
    response = llm_router.complete(prompt="...", system_prompt="...", task_type="lab_report")
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Callable, List, Tuple

logger = logging.getLogger("manthana.llm_router")


def _valid_gemini_key(key: str) -> bool:
    if not key or len(key) < 30:
        return False
    kl = key.lower()
    if "your-" in kl or kl.startswith("aiza-placeholder"):
        return False
    return key.startswith("AIza")


def _valid_sk_key(key: str) -> bool:
    if not key or key in ("", "sk-xxx", "sk-placeholder"):
        return False
    if "your-" in key.lower():
        return False
    return True


class ModelType:
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    KIMI = "kimi"
    GROQ = "groq"


class TaskType:
    LAB_REPORT = "lab_report"
    UNIFIED_REPORT = "unified_report"
    CLINICAL_QA = "clinical_qa"
    SUMMARIZATION = "summarization"
    CORRELATION = "correlation"
    FALLBACK = "fallback"


class LLMRouter:
    """
    Primary: Gemini 2.0 Flash Lite (GEMINI_MODEL, default gemini-2.0-flash-lite).
    Fallback chain: Groq (GROQ_MODEL, default llama-3.3-70b-versatile) → DeepSeek V3 → Kimi.
    """

    def __init__(self):
        self.strategy = os.getenv("LLM_ROUTING_STRATEGY", "smart")

        self.config = {
            ModelType.GEMINI: {
                "api_key": os.getenv("GEMINI_API_KEY", ""),
                "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite"),
                "context_window": 1000000,
                "strengths": ["fast", "multimodal-ready", "cost", "json"],
            },
            ModelType.DEEPSEEK: {
                "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
                "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                "context_window": 64000,
                "strengths": ["json", "structured", "fallback"],
            },
            ModelType.KIMI: {
                "api_key": os.getenv("KIMI_API_KEY", ""),
                "base_url": os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1"),
                "model": os.getenv("KIMI_MODEL", "kimi-k2.5"),
                "context_window": 256000,
                "strengths": ["long_context", "synthesis"],
            },
            ModelType.GROQ: {
                "api_key": os.getenv("GROQ_API_KEY", ""),
                "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                "context_window": 128000,
                "strengths": ["speed"],
            },
        }

        self.available_models = self._check_availability()

        logger.info("LLM Router: strategy=%s available=%s", self.strategy, self.available_models)

    def _check_availability(self) -> List[str]:
        out: List[str] = []
        if _valid_gemini_key(self.config[ModelType.GEMINI]["api_key"]):
            out.append(ModelType.GEMINI)
        if _valid_sk_key(self.config[ModelType.DEEPSEEK]["api_key"]):
            out.append(ModelType.DEEPSEEK)
        if _valid_sk_key(self.config[ModelType.KIMI]["api_key"]):
            out.append(ModelType.KIMI)
        if _valid_sk_key(self.config[ModelType.GROQ]["api_key"]):
            out.append(ModelType.GROQ)
        return out

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        task_type: str = "clinical_qa",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        requires_json: bool = False,
        force_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = self._build_messages(system_prompt, prompt)

        if force_model:
            fm = force_model.lower().strip()
            if fm == "gemini" and ModelType.GEMINI in self.available_models:
                return self._call_gemini(self.config[ModelType.GEMINI], messages, temperature, max_tokens)
            if fm == "deepseek" and ModelType.DEEPSEEK in self.available_models:
                return self._call_deepseek(self.config[ModelType.DEEPSEEK], messages, temperature, max_tokens)
            if fm == "kimi" and ModelType.KIMI in self.available_models:
                return self._call_kimi(self.config[ModelType.KIMI], messages, temperature, max_tokens)
            if fm == "groq" and ModelType.GROQ in self.available_models:
                return self._call_groq(self.config[ModelType.GROQ], messages, temperature, max_tokens)

        # Default chain: Gemini → Groq (fast free-tier fallback) → DeepSeek → Kimi
        chain: List[Tuple[str, Callable]] = []
        if ModelType.GEMINI in self.available_models:
            chain.append(
                (ModelType.GEMINI, lambda: self._call_gemini(self.config[ModelType.GEMINI], messages, temperature, max_tokens))
            )
        if ModelType.GROQ in self.available_models:
            chain.append(
                (ModelType.GROQ, lambda: self._call_groq(self.config[ModelType.GROQ], messages, temperature, max_tokens))
            )
        if ModelType.DEEPSEEK in self.available_models:
            chain.append(
                (ModelType.DEEPSEEK, lambda: self._call_deepseek(self.config[ModelType.DEEPSEEK], messages, temperature, max_tokens))
            )
        if ModelType.KIMI in self.available_models:
            chain.append(
                (ModelType.KIMI, lambda: self._call_kimi(self.config[ModelType.KIMI], messages, temperature, max_tokens))
            )

        if not chain:
            raise ValueError("No LLM providers configured. Set GEMINI_API_KEY, GROQ_API_KEY, and/or DEEPSEEK_API_KEY in .env")

        last_err: Optional[Exception] = None
        for name, fn in chain:
            try:
                return fn()
            except Exception as e:
                last_err = e
                logger.warning("LLM %s failed: %s", name, e)

        raise RuntimeError(f"All LLM providers failed. Last error: {last_err}")

    @staticmethod
    def _build_messages(system_prompt: str, prompt: str) -> List[Dict[str, str]]:
        msgs = []
        if system_prompt and system_prompt.strip():
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _call_gemini(self, config: dict, messages: list, temperature: float, max_tokens: int) -> dict:
        """Google Generative Language API — Gemini 2.0 Flash Lite (direct)."""
        import httpx

        system = ""
        user_texts = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_texts.append(m["content"])
        user = "\n\n".join(user_texts)
        combined = f"{system}\n\n{user}" if system.strip() else user

        model = config["model"]
        api_key = config["api_key"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        payload = {
            "contents": [{"role": "user", "parts": [{"text": combined}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        with httpx.Client(timeout=120.0) as client:
            r = client.post(f"{url}?key={api_key}", json=payload)
            r.raise_for_status()
            data = r.json()

        cand = data.get("candidates") or [{}]
        parts = (cand[0].get("content") or {}).get("parts") or []
        text = parts[0].get("text", "") if parts else ""
        if not text and data.get("promptFeedback", {}).get("blockReason"):
            raise RuntimeError(f"Gemini blocked: {data['promptFeedback']}")

        usage = data.get("usageMetadata") or {}
        return {
            "content": text,
            "model_used": "gemini-2.0-flash-lite",
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
            "finish_reason": "stop",
        }

    def _call_deepseek(self, config: dict, messages: list, temperature: float, max_tokens: int) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        u = response.usage
        return {
            "content": response.choices[0].message.content,
            "model_used": "deepseek-v3",
            "usage": {
                "prompt_tokens": u.prompt_tokens if u else 0,
                "completion_tokens": u.completion_tokens if u else 0,
                "total_tokens": u.total_tokens if u else 0,
            },
            "finish_reason": response.choices[0].finish_reason,
        }

    def _call_kimi(self, config: dict, messages: list, temperature: float, max_tokens: int) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        u = response.usage
        return {
            "content": response.choices[0].message.content,
            "model_used": "kimi-2.5",
            "usage": {
                "prompt_tokens": u.prompt_tokens if u else 0,
                "completion_tokens": u.completion_tokens if u else 0,
                "total_tokens": u.total_tokens if u else 0,
            },
            "finish_reason": response.choices[0].finish_reason,
        }

    def _call_groq(self, config: dict, messages: list, temperature: float, max_tokens: int) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        u = response.usage
        mid = config.get("model", "llama-3.3-70b-versatile")
        return {
            "content": response.choices[0].message.content,
            "model_used": f"groq:{mid}",
            "usage": {
                "prompt_tokens": u.prompt_tokens if u else 0,
                "completion_tokens": u.completion_tokens if u else 0,
                "total_tokens": u.total_tokens if u else 0,
            },
            "finish_reason": response.choices[0].finish_reason,
        }

    def get_model_info(self) -> dict:
        return {
            "strategy": self.strategy,
            "primary": "gemini-2.0-flash-lite",
            "fallback_after_gemini": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "then": ["deepseek-v3", "kimi"],
            "available_models": self.available_models,
        }


llm_router = LLMRouter()


def analyze_lab_report(structured_data: dict, raw_text: str) -> dict:
    prompt = f"""Analyze this structured lab report data:

STRUCTURED DATA:
{json.dumps(structured_data, indent=2, default=str)[:6000]}

RAW TEXT:
{raw_text[:3000]}

Provide clinical interpretation with severity classification."""
    return llm_router.complete(
        prompt=prompt,
        system_prompt="You are a senior clinical pathologist. Identify abnormalities and classify severity.",
        task_type=TaskType.LAB_REPORT,
        requires_json=True,
    )


def generate_unified_report(individual_results: list, patient_id: str) -> dict:
    results_json = json.dumps(individual_results, indent=2, default=str)
    prompt = f"""Synthesize these individual modality analyses into a unified clinical report:

PATIENT ID: {patient_id}

INDIVIDUAL RESULTS:
{results_json[:15000]}

Generate a comprehensive unified report that:
1. Identifies cross-modal correlations
2. Highlights critical findings
3. Provides integrated impression
4. Suggests follow-up recommendations"""
    return llm_router.complete(
        prompt=prompt,
        system_prompt="You are a senior radiologist synthesizing multi-modal findings.",
        task_type=TaskType.UNIFIED_REPORT,
        temperature=0.2,
        max_tokens=8192,
    )


def clinical_qa(question: str, context: dict) -> dict:
    context_json = json.dumps(context, indent=2, default=str)
    prompt = f"""Context:
{context_json[:4000]}

Question: {question}

Provide a clear, evidence-based answer."""
    return llm_router.complete(
        prompt=prompt,
        system_prompt="You are a knowledgeable medical AI assistant.",
        task_type=TaskType.CLINICAL_QA,
    )
