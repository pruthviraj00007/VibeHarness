"""Persistent user settings.

A tiny JSON store at ~/.vibeharness/settings.json that overrides the built-in
Config defaults. Lets the user change e.g. the default temperature once and have
it stick, while per-run CLI flags still take precedence.

Resolution order (lowest to highest): Config defaults < saved settings < CLI flags.
"""
from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

from .config import Config

SETTINGS_DIR = Path.home() / ".vibeharness"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"

# Friendly CLI key -> (Config field name, value parser). Only these are settable.
_SETTABLE: dict[str, tuple[str, type]] = {
    "temp": ("temperature", float),
    "temperature": ("temperature", float),
    "model": ("model", str),
    "max-steps": ("max_steps", int),
    "max_steps": ("max_steps", int),
    "top-p": ("top_p", float),
    "top_k": ("top_k", int),
}


def settable_keys() -> list[str]:
    # de-duplicate to the canonical friendly names
    return ["temp", "model", "max-steps", "top-p", "top_k"]


class Settings:
    """Load/save persistent overrides and merge them into a Config."""

    @staticmethod
    def load() -> dict:
        if SETTINGS_PATH.exists():
            try:
                data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    @staticmethod
    def save(data: dict) -> None:
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def set(key: str, raw_value: str) -> tuple[str, object]:
        """Persist one setting. Returns (config_field, parsed_value)."""
        if key not in _SETTABLE:
            raise KeyError(key)
        field_name, caster = _SETTABLE[key]
        value = caster(raw_value)            # may raise ValueError on bad input
        data = Settings.load()
        data[field_name] = value
        Settings.save(data)
        return field_name, value

    @staticmethod
    def reset() -> bool:
        """Delete saved settings. Returns True if a file was removed."""
        if SETTINGS_PATH.exists():
            SETTINGS_PATH.unlink()
            return True
        return False

    @staticmethod
    def apply(base: Config) -> Config:
        """Return a Config with saved overrides applied to known fields only."""
        valid = {f.name for f in fields(Config)}
        overrides = {k: v for k, v in Settings.load().items() if k in valid}
        return replace(base, **overrides) if overrides else base
