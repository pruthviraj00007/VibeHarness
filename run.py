"""CLI entrypoint for the vibeharness Ralph-loop agent.

`vibe` runs this. The workspace is the current terminal directory unless
--workdir is given. Each turn streams live to the console.

Run `vibe --help` to see every command and parameter.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from vibeharness.agent import RalphAgent
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.llm import OllamaClient
from vibeharness.prompt import SystemPromptBuilder
from vibeharness.registry import ToolRegistry
from vibeharness.reporting import ConsoleReporter
from vibeharness.settings import SETTINGS_PATH, Settings, settable_keys

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    saved_temp = Settings.apply(Config()).temperature
    epilog = f"""\
examples:
  vibe "create a README and fill in a project overview"
  vibe --temp 1.0 "draft notes.txt"        run once at a different temperature
  vibe --max-steps 30 "refactor this dir"  allow more steps for a big task

manage persistent defaults (saved to ~/.vibeharness/settings.json):
  vibe --set temp 0.5                      change the default temperature
  vibe --set max-steps 25                  change the default step budget
  vibe --show-config                       show current settings
  vibe --reset-config                      restore built-in defaults

settable keys: {', '.join(settable_keys())}
current default temperature: {saved_temp}
"""
    p = argparse.ArgumentParser(
        prog="vibe", description="vibe — a tiny local coding agent (VibeThinker via Ollama)",
        epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
    )
    p.add_argument("task", nargs="*", help="the task for the agent to perform (quote it)")
    # per-run overrides (None = fall back to saved setting / built-in default)
    p.add_argument("--temp", type=float, default=None, metavar="T",
                   help="sampling temperature for this run only")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="Ollama model name for this run only")
    p.add_argument("--max-steps", type=int, default=None, metavar="N",
                   help="max tool-call steps for this run only")
    p.add_argument("--workdir", default=None, metavar="DIR",
                   help="working directory (default: current terminal directory)")
    p.add_argument("--no-color", action="store_true", help="disable colored output")
    # management commands (run and exit, no task needed)
    p.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"),
                   help="persist a default, e.g. --set temp 0.5")
    p.add_argument("--show-config", action="store_true", help="print current settings and exit")
    p.add_argument("--reset-config", action="store_true", help="clear saved settings and exit")
    p.add_argument("--print-system", action="store_true", help="print the system prompt and exit")
    return p


def resolve_config(args: argparse.Namespace) -> Config:
    """Config defaults < saved settings < CLI flags (only those provided)."""
    cfg = Settings.apply(Config())
    overrides = {}
    if args.temp is not None:
        overrides["temperature"] = args.temp
    if args.model is not None:
        overrides["model"] = args.model
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    return replace(cfg, **overrides) if overrides else cfg


def cmd_show_config() -> int:
    saved = Settings.load()
    effective = Settings.apply(Config())
    print(f"settings file: {SETTINGS_PATH}")
    print(f"saved overrides: {saved or '(none)'}")
    print("effective defaults:")
    print(f"  model       = {effective.model}")
    print(f"  temperature = {effective.temperature}")
    print(f"  max_steps   = {effective.max_steps}")
    print(f"  top_p       = {effective.top_p}")
    print(f"  top_k       = {effective.top_k}")
    return 0


def cmd_set(key: str, value: str) -> int:
    try:
        field, parsed = Settings.set(key, value)
    except KeyError:
        print(f"error: '{key}' is not settable. Settable keys: {', '.join(settable_keys())}")
        return 2
    except ValueError:
        print(f"error: '{value}' is not a valid value for '{key}'.")
        return 2
    print(f"saved: {field} = {parsed}")
    return 0


def run_agent(args: argparse.Namespace) -> int:
    task = " ".join(args.task)
    config = resolve_config(args)

    fs = FileSystem()
    registry = ToolRegistry(build_default_tools(fs, config.observation_char_limit))
    system_prompt = SystemPromptBuilder(registry).build()

    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        os.chdir(workdir)
    workdir = Path.cwd()

    reporter = ConsoleReporter(color=not args.no_color)
    reporter.run_start(task, str(workdir), config)

    agent = RalphAgent(OllamaClient(config), registry, system_prompt, config, reporter=reporter)
    result = agent.run(task)
    reporter.run_end(result)

    runs = REPO / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = runs / f"run_t{config.temperature}_{stamp}.txt"
    out.write_text(result.transcript(), encoding="utf-8")
    print(f" transcript: {out}")
    return 0 if result.finished else 2


def main() -> int:
    argv = sys.argv[1:]
    parser = build_parser()

    # Friendly help: bare `vibe`, `vibe help`, or `vibe -help` all show help.
    if not argv or argv[0] in ("help", "-help"):
        parser.print_help()
        return 0

    args = parser.parse_args()

    # management commands (exit without running the agent)
    if args.show_config:
        return cmd_show_config()
    if args.reset_config:
        removed = Settings.reset()
        print("settings cleared." if removed else "no saved settings to clear.")
        return 0
    if args.set:
        return cmd_set(args.set[0], args.set[1])
    if args.print_system:
        fs = FileSystem()
        registry = ToolRegistry(build_default_tools(fs, Config().observation_char_limit))
        print(SystemPromptBuilder(registry).build())
        return 0

    if not args.task:
        print("error: no task given.\n")
        parser.print_help()
        return 2

    return run_agent(args)


if __name__ == "__main__":
    raise SystemExit(main())
