from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

from manthana_inference.loader import build_extra_headers
from manthana_inference.schema import CloudInferenceConfig, RoleConfig

logger = logging.getLogger("manthana_inference.client")


def build_openrouter_sync_client(
    api_key: str,
    config: CloudInferenceConfig,
) -> OpenAI:
    if not (api_key or "").strip():
        raise ValueError("OPENROUTER_API_KEY is empty")
    headers = build_extra_headers(config)
    base = config.openrouter_base_url.rstrip("/")
    return OpenAI(
        base_url=base,
        api_key=api_key.strip(),
        default_headers=headers or None,
    )


def build_openrouter_async_client(
    api_key: str,
    config: CloudInferenceConfig,
) -> AsyncOpenAI:
    if not (api_key or "").strip():
        raise ValueError("OPENROUTER_API_KEY is empty")
    headers = build_extra_headers(config)
    base = config.openrouter_base_url.rstrip("/")
    return AsyncOpenAI(
        base_url=base,
        api_key=api_key.strip(),
        default_headers=headers or None,
    )


def chat_complete_sync(
    client: OpenAI,
    config: CloudInferenceConfig,
    role: str,
    messages: List[Dict[str, Any]],
    *,
    role_cfg: Optional[RoleConfig] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> tuple[str, str]:
    """Return (content, model_used). Tries primary + fallbacks."""
    from manthana_inference.loader import resolve_role

    rc = role_cfg or resolve_role(config, role)
    models = [rc.model] + [m for m in rc.fallback_models if m]
    last_err: Optional[Exception] = None
    for model in models:
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": rc.max_tokens,
                "temperature": rc.temperature,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            comp = client.chat.completions.create(**kwargs)
            text = (comp.choices[0].message.content or "").strip()
            return text, model
        except Exception as e:
            last_err = e
            logger.debug("openrouter model %s failed: %s", model, e)
    raise RuntimeError(f"All models failed for role {role!r}: {last_err}")


async def chat_complete_async(
    client: AsyncOpenAI,
    config: CloudInferenceConfig,
    role: str,
    messages: List[Dict[str, Any]],
    *,
    role_cfg: Optional[RoleConfig] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> tuple[str, str]:
    from manthana_inference.loader import resolve_role

    rc = role_cfg or resolve_role(config, role)
    models = [rc.model] + [m for m in rc.fallback_models if m]
    last_err: Optional[Exception] = None
    for model in models:
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": rc.max_tokens,
                "temperature": rc.temperature,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            comp = await client.chat.completions.create(**kwargs)
            text = (comp.choices[0].message.content or "").strip()
            return text, model
        except Exception as e:
            last_err = e
            logger.debug("openrouter model %s failed: %s", model, e)
    raise RuntimeError(f"All models failed for role {role!r}: {last_err}")


async def stream_chat_async(
    client: AsyncOpenAI,
    config: CloudInferenceConfig,
    role: str,
    messages: List[Dict[str, Any]],
    *,
    role_cfg: Optional[RoleConfig] = None,
) -> AsyncIterator[tuple[str, str]]:
    """Yield (delta_text, model_id) from first successful model in chain."""
    from manthana_inference.loader import resolve_role

    rc = role_cfg or resolve_role(config, role)
    models = [rc.model] + [m for m in rc.fallback_models if m]
    last_err: Optional[Exception] = None
    for model in models:
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=rc.max_tokens,
                temperature=rc.temperature,
                stream=True,
            )
            async for chunk in stream:
                ch = chunk.choices[0]
                if ch.delta and ch.delta.content:
                    yield ch.delta.content, model
            return
        except Exception as e:
            last_err = e
            logger.warning("openrouter stream model %s failed: %s", model, e)
    raise RuntimeError(f"All stream models failed for role {role!r}: {last_err}")
