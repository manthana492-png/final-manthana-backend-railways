"""Manthana OpenRouter SSOT — config load, role resolution, chat helpers."""

from manthana_inference.client import (
    build_openrouter_async_client,
    build_openrouter_sync_client,
    chat_complete_async,
    chat_complete_sync,
    stream_chat_async,
)
from manthana_inference.loader import load_cloud_inference_config, resolve_role
from manthana_inference.schema import CloudInferenceConfig, RoleConfig

__all__ = [
    "CloudInferenceConfig",
    "RoleConfig",
    "load_cloud_inference_config",
    "resolve_role",
    "build_openrouter_sync_client",
    "build_openrouter_async_client",
    "chat_complete_sync",
    "chat_complete_async",
    "stream_chat_async",
]
