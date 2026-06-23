"""Prompt construction.

The system prompt is deliberately concise and is assembled from the registry so
the documented tools and the enforced schema can never disagree. It follows the
common system-prompt convention: describe each tool and its parameters in plain
English, then give the formal JSON schema.
"""
from __future__ import annotations

import json

from .registry import ToolRegistry

_SYSTEM_TEMPLATE = """\
You are a capable task-execution agent operating a computer through a small set \
of tools. You work in a loop: on each turn you read what you have done so far, \
then choose exactly one tool to make progress. Keep going until the task is fully \
done, then call `finish`.

# How the loop works
- You are given the task and a plain-English account of the actions you have \
already taken and what each returned.
- Each turn, output exactly ONE action as a single JSON object: \
{{"tool": <tool name>, "args": {{ ... }}}}.
- Do not output anything except that one JSON object. Do not invent tools or \
parameters. Only use the tools listed below.
- After each action you will see its result described in the account. Use it to \
decide your next action. If an action returns an error, adapt — do not repeat the \
same failing call.
- When the task is complete, call `finish` with a short summary.

# Tools
{docs}

# Action schema
Every action you emit must validate against this JSON schema:
{schema}

# Guidance
- Prefer the simplest tool that accomplishes the step.
- Verify your work: after writing a file, read it back before finishing.
- Use relative paths unless an absolute path is required.
"""


class SystemPromptBuilder:
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def build(self) -> str:
        schema = json.dumps(self._registry.action_schema(), indent=2)
        return _SYSTEM_TEMPLATE.format(docs=self._registry.docs(), schema=schema)


def build_turn_prompt(task: str, narrative: str) -> str:
    """The per-turn user message: the task plus the natural-language account."""
    return (
        f"# Task\n{task}\n\n"
        f"# What you have done so far\n{narrative}\n\n"
        f"# Your next action\n"
        f"Choose the single best next tool call to make progress on the task "
        f"(or call `finish` if it is already complete). Respond with one JSON action."
    )
