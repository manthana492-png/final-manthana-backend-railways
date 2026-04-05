from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import yaml

from manthana_inference.schema import CloudInferenceConfig, RoleConfig

PathLike = Union[str, Path]


def _default_config_path() -> Path:
    env = (os.environ.get("CLOUD_INFERENCE_CONFIG_PATH") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Repo layout: .../this_studio/packages/manthana-inference/src/manthana_inference/loader.py
    here = Path(__file__).resolve()
    repo_root = here.parents[4]
    return (repo_root / "config" / "cloud_inference.yaml").resolve()


def load_cloud_inference_config(path: Optional[PathLike] = None) -> CloudInferenceConfig:
    """Load and validate cloud_inference.yaml."""
    p = Path(path) if path else _default_config_path()
    if not p.is_file():
        raise FileNotFoundError(f"Cloud inference config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("cloud_inference.yaml must be a mapping at root")
    roles_raw = raw.get("roles") or {}
    roles: dict[str, RoleConfig] = {}
    defaults = raw.get("defaults") or {}
    default_max = int(defaults.get("max_tokens", 4096))
    default_temp = float(defaults.get("temperature", 0.2))
    for name, spec in roles_raw.items():
        if not isinstance(spec, dict):
            continue
        roles[name] = RoleConfig(
            model=str(spec.get("model") or "").strip(),
            max_tokens=int(spec.get("max_tokens", default_max)),
            temperature=float(spec.get("temperature", default_temp)),
            fallback_models=list(spec.get("fallback_models") or []),
        )
    return CloudInferenceConfig(
        schema_version=str(raw.get("schema_version", "1")),
        openrouter_base_url=str(
            raw.get("openrouter_base_url") or "https://openrouter.ai/api/v1"
        ).rstrip("/"),
        header_env=dict(raw.get("header_env") or {}),
        defaults=dict(defaults),
        roles=roles,
    )


def resolve_role(config: CloudInferenceConfig, role: str) -> RoleConfig:
    if role not in config.roles:
        raise KeyError(f"Unknown inference role: {role!r}. Defined roles: {sorted(config.roles)}")
    r = config.roles[role]
    if not r.model:
        raise ValueError(f"Role {role!r} has empty model in cloud_inference.yaml")
    return r


def build_extra_headers(config: CloudInferenceConfig) -> dict[str, str]:
    """Resolve optional OpenRouter attribution headers from env names in YAML."""
    out: dict[str, str] = {}
    for header_name, env_name in config.header_env.items():
        val = (os.environ.get(env_name) or "").strip()
        if val:
            out[header_name] = val
    return out
