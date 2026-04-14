"""AI orchestration prompts (interrogator / interpreter)."""
from .interrogator_base import interrogator_system_prompt
from .interpreter_base import interpreter_system_prompt
from .modality_prompts import group_specialization_for

__all__ = [
    "interrogator_system_prompt",
    "interpreter_system_prompt",
    "group_specialization_for",
]
