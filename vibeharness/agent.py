"""The Ralph loop.

Each turn: render the task + narrative -> ask the model for one or more
constrained actions (a JSON array) -> execute them in order -> append a
natural-language observation for each. Repeat until `finish` or the step budget.

Batching several actions in one turn is allowed (the model decides them together,
without seeing intermediate results); a turn that needs a result before deciding
the next move simply emits a single action.

The agent depends only on abstractions: an LLMClient, a ToolRegistry, a
NarrativeMemory and a Reporter.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .config import Config
from .llm import LLMClient
from .memory import NarrativeMemory
from .prompt import build_turn_prompt
from .registry import ToolRegistry
from .reporting import NullReporter, Reporter


@dataclass
class Action:
    """One executed tool call within a turn."""
    tool: str | None
    args: dict
    observation: str
    ok: bool
    final: bool = False


@dataclass
class Turn:
    """One model turn: its reasoning, the raw action payload, and the executed actions."""
    index: int
    reasoning: str
    raw_action: str
    actions: list[Action] = field(default_factory=list)


@dataclass
class RunResult:
    task: str
    turns: list[Turn] = field(default_factory=list)
    finished: bool = False
    final_summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def transcript(self) -> str:
        out = [f"TASK: {self.task}", ""]
        for turn in self.turns:
            out.append(f"--- Turn {turn.index} ---")
            if turn.reasoning.strip():
                out.append(f"reasoning:\n{turn.reasoning.strip()}")
            for a in turn.actions:
                out.append(f"action: {a.tool} {json.dumps(a.args, ensure_ascii=False)}")
                out.append(f"result: {a.observation}")
            out.append("")
        out.append(f"FINISHED: {self.finished}")
        if self.final_summary:
            out.append(f"SUMMARY: {self.final_summary}")
        return "\n".join(out)


class RalphAgent:
    def __init__(self, client: LLMClient, registry: ToolRegistry,
                 system_prompt: str, config: Config, reporter: Reporter | None = None):
        self._client = client
        self._registry = registry
        self._system = system_prompt
        self._cfg = config
        self._reporter = reporter or NullReporter()

    def run(self, task: str) -> RunResult:
        memory = NarrativeMemory()
        result = RunResult(task=task)
        schema = self._registry.action_schema()

        for i in range(1, self._cfg.max_steps + 1):
            self._reporter.turn_start(i)
            user = build_turn_prompt(task, memory.render())
            decision = self._client.decide(
                self._system, user, schema,
                on_reason=self._reporter.reasoning_token,
                on_action=self._reporter.action_token,
            )
            turn = Turn(index=i, reasoning=decision.reasoning, raw_action=decision.action_json)
            result.turns.append(turn)

            actions, error = self._parse(decision.action_json)
            if error is not None:
                self._record(turn, Action(None, {}, f"your last response was invalid and "
                                          f"could not be run: {error}.", ok=False), memory)
                continue

            for tool_name, args in actions:
                action = self._execute(tool_name, args)
                self._record(turn, action, memory)
                if action.final:
                    result.finished = True
                    result.final_summary = args.get("summary", "")
                    break

            if result.finished:
                break

        return result

    # ---- helpers ----
    def _execute(self, tool_name: str | None, args: dict) -> Action:
        tool = self._registry.get(tool_name) if tool_name else None
        if tool is None:
            return Action(tool_name, args,
                          f"you tried to use '{tool_name}', which is not a real tool.", ok=False)
        result = tool.run(args)
        return Action(tool_name, args, result.observation, ok=result.ok, final=result.is_final)

    def _record(self, turn: Turn, action: Action, memory: NarrativeMemory) -> None:
        turn.actions.append(action)
        memory.record(action.observation)
        self._reporter.action_result(action)

    @staticmethod
    def _parse(action_json: str):
        """Parse the turn's payload into a list of (tool, args). A lone object is
        accepted as a one-element batch. Returns (actions, error_message)."""
        try:
            obj = json.loads(action_json)
        except json.JSONDecodeError as e:
            return None, f"not valid JSON ({e})"
        if isinstance(obj, dict):
            obj = [obj]
        if not isinstance(obj, list) or not obj:
            return None, "expected a non-empty JSON array of actions"
        actions = []
        for item in obj:
            if not isinstance(item, dict) or "tool" not in item:
                return None, "each action must be an object with a 'tool' field"
            args = item.get("args", {})
            if not isinstance(args, dict):
                return None, "'args' must be an object"
            actions.append((item["tool"], args))
        return actions, None
