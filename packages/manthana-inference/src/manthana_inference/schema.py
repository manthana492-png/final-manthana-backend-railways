from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RoleConfig(BaseModel):
    model: str
    max_tokens: int = 4096
    temperature: float = 0.2
    fallback_models: List[str] = Field(default_factory=list)


class CloudInferenceConfig(BaseModel):
    schema_version: str = "1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    header_env: Dict[str, str] = Field(default_factory=dict)
    defaults: Dict[str, Any] = Field(default_factory=dict)
    roles: Dict[str, RoleConfig]

    def models_to_try(self, role: str) -> List[str]:
        r = self.roles[role]
        out = [r.model]
        out.extend(m for m in r.fallback_models if m and m not in out)
        return out
