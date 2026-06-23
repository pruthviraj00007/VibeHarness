"""CLI entrypoint for the vibeharness Ralph-loop agent.

Examples:
  python run.py "Create notes.txt in the workspace containing 'hello hello hello', then read it back."
  python run.py --model vibethinker --temp 0.3 --max-steps 12 --workdir workspace "List the workspace."
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path

from vibeharness.agent import RalphAgent, Step
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.llm import OllamaClient
from vibeharness.prompt import SystemPromptBuilder
from vibeharness.registry import ToolRegistry

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vibeharness Ralph-loop agent")
    p.add_argument("task", help="the task for the agent to accomplish")
    p.add_argument("--model", default=Config.model)
    p.add_argument("--temp", type=float, default=Config.temperature)
    p.add_argument("--max-steps", type=int, default=Config.max_steps)
    p.add_argument("--workdir", default=str(REPO / "workspace"),
                   help="working directory for relative paths (default: ./workspace)")
    p.add_argument("--print-system", action="store_true", help="print the system prompt and exit")
    return p.parse_args()


def make_step_logger():
    def log(step: Step) -> None:
        status = "ok " if step.ok else "ERR"
        print(f"[step {step.index:>2}] [{status}] {step.tool or '?'}  ->  {step.observation}", flush=True)
    return log


def main() -> int:
    args = parse_args()
    config = Config(model=args.model, temperature=args.temp, max_steps=args.max_steps)

    fs = FileSystem()
    registry = ToolRegistry(build_default_tools(fs, config.observation_char_limit))
    system_prompt = SystemPromptBuilder(registry).build()

    if args.print_system:
        print(system_prompt)
        return 0

    # Resolve relative paths under the chosen working directory.
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    import os
    os.chdir(workdir)

    print(f"model={config.model} temp={config.temperature} max_steps={config.max_steps}")
    print(f"workdir={workdir}")
    print(f"task: {args.task}\n")

    agent = RalphAgent(OllamaClient(config), registry, system_prompt, config,
                       on_step=make_step_logger())
    result = agent.run(args.task)

    print("\n" + "=" * 70)
    print(f"finished={result.finished}  steps={len(result.steps)}")
    if result.final_summary:
        print(f"summary: {result.final_summary}")

    # Save transcript (timestamped; never overwrites).
    runs = REPO / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = runs / f"run_t{config.temperature}_{stamp}.txt"
    out.write_text(result.transcript(), encoding="utf-8")
    print(f"saved transcript: {out}")
    return 0 if result.finished else 2


if __name__ == "__main__":
    raise SystemExit(main())
