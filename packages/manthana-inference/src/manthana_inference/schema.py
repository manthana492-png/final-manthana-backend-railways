from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RoleConfig(BaseModel):
    """provider: openrouter (default) or nim — see cloud_inference.yaml + NVIDIA_NIM_API_KEY."""

    model: str
    max_tokens: int = 4096
    temperature: float = 0.2
    fallback_models: List[str] = Field(default_factory=list)
    provider: str = "openrouter"


class CloudInferenceConfig(BaseModel):
    """SSOT for OpenRouter + NIM roles; optional orch_chains for gateway multi-step routing."""

    schema_version: str = "1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    header_env: Dict[str, str] = Field(default_factory=dict)
    defaults: Dict[str, Any] = Field(default_factory=dict)
    roles: Dict[str, RoleConfig]
    orch_chains: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Logical stage keys → ordered role names (each must exist in roles).",
    )
    model_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Model slug → pinned version string for audit display.",
    )

    def models_to_try(self, role: str) -> List[str]:
        r = self.roles[role]
        out = [r.model]
        out.extend(m for m in r.fallback_models if m and m not in out)
        return out
