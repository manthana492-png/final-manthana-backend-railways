from __future__ import annotations

import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

from manthana_inference.loader import build_extra_headers
from manthana_inference.schema import CloudInferenceConfig, RoleConfig

logger = logging.getLogger("manthana_inference.client")

_ONLINE_SUFFIX = re.compile(r":online$")


def _strip_openrouter_online_suffix(model: str) -> str:
    """Remove deprecated :online suffix when using openrouter:web_search server tool."""
    return _ONLINE_SUFFIX.sub("", model) if model else model


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


def _openrouter_web_search_tools(web_search_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"type": "openrouter:web_search", "parameters": dict(web_search_parameters)}]


_MD_HTTP_LINK = re.compile(r"\[([^\]]{0,800})\]\((https?://[^)\s]+)\)", re.MULTILINE)
_MAX_WEB_LINKS = 14


def _safe_http_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def _normalize_cited_url(url: str) -> str:
    return (url or "").strip().rstrip(").,;]'\"»")


def web_links_from_markdown(text: str) -> List[Dict[str, str]]:
    """Extract [label](https://...) links from assistant text (streaming path)."""
    if not (text or "").strip():
        return []
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for m in _MD_HTTP_LINK.finditer(text):
        title = (m.group(1) or "").strip() or m.group(2)
        url = _normalize_cited_url(m.group(2))
        if not _safe_http_url(url) or url in seen:
            continue
        seen.add(url)
        out.append({"title": title[:400], "url": url})
        if len(out) >= _MAX_WEB_LINKS:
            break
    return out


def web_links_from_message_annotations(annotations: Any) -> List[Dict[str, str]]:
    """Parse Chat Completions message.annotations (url_citation) when present."""
    if not annotations:
        return []
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for a in annotations:
        try:
            if getattr(a, "type", None) != "url_citation":
                continue
            uc = getattr(a, "url_citation", None)
            if uc is None:
                continue
            url = _normalize_cited_url(getattr(uc, "url", "") or "")
            if not _safe_http_url(url) or url in seen:
                continue
            title = (getattr(uc, "title", None) or "").strip() or url
            seen.add(url)
            out.append({"title": title[:400], "url": url})
            if len(out) >= _MAX_WEB_LINKS:
                break
        except Exception:
            continue
    return out


def merge_web_search_links(
    assistant_text: str,
    annotations: Any,
) -> List[Dict[str, str]]:
    """Prefer API annotations; add markdown links not already listed. Cap total."""
    merged: List[Dict[str, str]] = []
    seen: set[str] = set()
    for item in web_links_from_message_annotations(annotations):
        u = item["url"]
        if u not in seen:
            seen.add(u)
            merged.append(item)
    for item in web_links_from_markdown(assistant_text):
        u = item["url"]
        if u not in seen:
            seen.add(u)
            merged.append(item)
        if len(merged) >= _MAX_WEB_LINKS:
            break
    return merged[:_MAX_WEB_LINKS]


def chat_complete_sync(
    client: OpenAI,
    config: CloudInferenceConfig,
    role: str,
    messages: List[Dict[str, Any]],
    *,
    role_cfg: Optional[RoleConfig] = None,
    response_format: Optional[Dict[str, Any]] = None,
    web_search_parameters: Optional[Dict[str, Any]] = None,
) -> tuple[str, str, Dict[str, Any]]:
    """Return (content, model_used, usage_info). Tries primary + fallbacks."""
    from manthana_inference.loader import resolve_role

    rc = role_cfg or resolve_role(config, role)
    models = [rc.model] + [m for m in rc.fallback_models if m]
    last_err: Optional[Exception] = None
    for model in models:
        try:
            model_eff = _strip_openrouter_online_suffix(model) if web_search_parameters else model
            kwargs: Dict[str, Any] = {
                "model": model_eff,
                "messages": messages,
                "max_tokens": rc.max_tokens,
                "temperature": rc.temperature,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            if web_search_parameters is not None:
                kwargs["tools"] = _openrouter_web_search_tools(web_search_parameters)
            comp = client.chat.completions.create(**kwargs)
            text = (comp.choices[0].message.content or "").strip()
            usage = getattr(comp, "usage", None)
            usage_info: Dict[str, Any] = {}
            if usage is not None:
                try:
                    usage_info = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    }
                except Exception:
                    usage_info = {}
            return text, model_eff, usage_info
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
    web_search_parameters: Optional[Dict[str, Any]] = None,
) -> tuple[str, str, Dict[str, Any]]:
    """Return (content, model_used, usage_info). Tries primary + fallbacks."""
    from manthana_inference.loader import resolve_role

    rc = role_cfg or resolve_role(config, role)
    models = [rc.model] + [m for m in rc.fallback_models if m]
    last_err: Optional[Exception] = None
    for model in models:
        try:
            model_eff = _strip_openrouter_online_suffix(model) if web_search_parameters else model
            kwargs: Dict[str, Any] = {
                "model": model_eff,
                "messages": messages,
                "max_tokens": rc.max_tokens,
                "temperature": rc.temperature,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            if web_search_parameters is not None:
                kwargs["tools"] = _openrouter_web_search_tools(web_search_parameters)
            comp = await client.chat.completions.create(**kwargs)
            text = (comp.choices[0].message.content or "").strip()
            usage = getattr(comp, "usage", None)
            usage_info: Dict[str, Any] = {}
            if usage is not None:
                try:
                    usage_info = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    }
                except Exception:
                    usage_info = {}
            return text, model_eff, usage_info
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
    web_search_parameters: Optional[Dict[str, Any]] = None,
    strip_online_suffix: bool = False,
    completion_meta: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[tuple[str, str]]:
    """Yield (delta_text, model_id) from first successful model in chain."""
    from manthana_inference.loader import resolve_role

    rc = role_cfg or resolve_role(config, role)
    models = [rc.model] + [m for m in rc.fallback_models if m]
    last_err: Optional[Exception] = None
    for model in models:
        try:
            model_eff = (
                _strip_openrouter_online_suffix(model)
                if (web_search_parameters is not None or strip_online_suffix)
                else model
            )
            kwargs: Dict[str, Any] = {
                "model": model_eff,
                "messages": messages,
                "max_tokens": rc.max_tokens,
                "temperature": rc.temperature,
                "stream": True,
            }
            if web_search_parameters is not None:
                kwargs["tools"] = _openrouter_web_search_tools(web_search_parameters)
            stream = await client.chat.completions.create(**kwargs)
            saw_content = False
            accumulated: List[str] = []
            async for chunk in stream:
                ch = chunk.choices[0]
                if ch.delta and ch.delta.content:
                    saw_content = True
                    accumulated.append(ch.delta.content)
                    yield ch.delta.content, model_eff
            if saw_content:
                if web_search_parameters is not None and completion_meta is not None:
                    full_text = "".join(accumulated)
                    completion_meta["web_links"] = merge_web_search_links(full_text, None)
                return
            # Server-side web search may not stream assistant text; fall back to one completion.
            if web_search_parameters is not None:
                comp = await client.chat.completions.create(
                    model=model_eff,
                    messages=messages,
                    max_tokens=rc.max_tokens,
                    temperature=rc.temperature,
                    stream=False,
                    tools=_openrouter_web_search_tools(web_search_parameters),
                )
                msg = comp.choices[0].message
                text = (msg.content or "").strip()
                if completion_meta is not None:
                    completion_meta["web_links"] = merge_web_search_links(
                        text,
                        getattr(msg, "annotations", None),
                    )
                if text:
                    yield text, model_eff
                return
            return
        except Exception as e:
            last_err = e
            logger.warning("openrouter stream model %s failed: %s", model, e)
    raise RuntimeError(f"All stream models failed for role {role!r}: {last_err}")
