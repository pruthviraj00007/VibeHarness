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
- Each turn, output a JSON ARRAY of one or more actions, each of the form \
{{"tool": <tool name>, "args": {{ ... }}}}. The actions run in order.
- Batch several actions in one turn when they are independent or you are confident \
of the outcome (e.g. write a file then read it back). Emit a single action when \
you must see its result before deciding the next move.
- Output nothing except that JSON array. Do not invent tools or parameters. \
Only use the tools listed below.
- After your actions run you will see each result described in the account. Use it \
to decide your next turn. If an action returns an error, adapt — do not repeat the \
same failing call.
- When the task is complete, end with a `finish` action and a short summary. \
Anything after `finish` in the array is ignored.

# Tools
{docs}

# Action schema
Your output each turn must validate against this JSON schema (an array of actions):
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
        f"Choose the next action (or several, as a batch) to make progress on the "
        f"task, ending with `finish` once it is complete. Respond with a JSON array "
        f"of one or more actions."
    )
