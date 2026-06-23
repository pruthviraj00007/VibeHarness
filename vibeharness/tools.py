"""Tool interface and the value objects around it.

A Tool is the single source of truth for: how it runs, its plain-text docs, and
its JSON schema. Docs and schema are *derived* from `parameters` so they can
never drift apart.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Param:
    """One tool parameter, described once and rendered as both docs and schema."""
    name: str
    type: str                       # JSON Schema scalar type: string | integer | boolean
    description: str
    required: bool = True
    enum: tuple[str, ...] | None = None
    default: Any = None

    def schema(self) -> dict:
        s: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            s["enum"] = list(self.enum)
        return s

    def doc(self) -> str:
        if self.required:
            qualifier = "required"
        elif self.default is not None:
            qualifier = f"optional, default {self.default!r}"
        else:
            qualifier = "optional"
        choices = f" Allowed values: {', '.join(self.enum)}." if self.enum else ""
        return f"  - `{self.name}` ({self.type}, {qualifier}) — {self.description}{choices}"


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a tool run. `observation` is a natural-language, past-tense
    sentence (starting with 'you ...') that gets woven into the narrative."""
    ok: bool
    observation: str
    is_final: bool = False          # True only for the finish tool


class Tool(ABC):
    name: str = ""
    description: str = ""

    @property
    @abstractmethod
    def parameters(self) -> list[Param]:
        ...

    @abstractmethod
    def run(self, args: dict) -> ToolResult:
        ...

    # ---- derived: schema (for constrained decoding) ----
    def _args_schema(self) -> dict:
        props = {p.name: p.schema() for p in self.parameters}
        required = [p.name for p in self.parameters if p.required]
        schema: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        return schema

    def call_schema(self) -> dict:
        """Schema for one call to THIS tool: {"tool": <const>, "args": {...}}."""
        return {
            "type": "object",
            "properties": {"tool": {"const": self.name}, "args": self._args_schema()},
            "required": ["tool", "args"],
        }

    # ---- derived: human docs for the system prompt ----
    def doc(self) -> str:
        lines = [f"### `{self.name}`", self.description]
        if self.parameters:
            lines.append("Parameters:")
            lines.extend(p.doc() for p in self.parameters)
        else:
            lines.append("Parameters: none.")
        return "\n".join(lines)
