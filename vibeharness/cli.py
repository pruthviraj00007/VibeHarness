"""Command-line interface for vibeharness.

Exposed as the ``vibe`` console script (see pyproject.toml) and runnable via
``python -m vibeharness`` or ``python run.py``. The workspace is the current
terminal directory unless ``--workdir`` is given; each turn streams live.

Run ``vibe --help`` to see every command and parameter.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .agent import RalphAgent
from .config import Config
from .filesystem import FileSystem
from .fs_tools import build_default_tools
from .llm import OllamaClient, OllamaUnavailable
from .prompt import SystemPromptBuilder
from .registry import ToolRegistry
from .reporting import ConsoleReporter
from .runlog import RunLogger
from .settings import Settings, settable_keys


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
        prog="vibe",
        description="vibe - a tiny local coding agent (VibeThinker via Ollama)",
        epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("task", nargs="*", help="the task for the agent to perform (quote it)")
    p.add_argument("--temp", type=float, default=None, metavar="T",
                   help="sampling temperature for this run only")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="Ollama model name for this run only")
    p.add_argument("--max-steps", type=int, default=None, metavar="N",
                   help="max tool-call steps for this run only")
    p.add_argument("--workdir", default=None, metavar="DIR",
                   help="working directory (default: current terminal directory)")
    p.add_argument("--no-color", action="store_true", help="disable colored output")
    p.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"),
                   help="persist a default, e.g. --set temp 0.5")
    p.add_argument("--show-config", action="store_true", help="print current settings and exit")
    p.add_argument("--reset-config", action="store_true", help="clear saved settings and exit")
    p.add_argument("--print-system", action="store_true", help="print the system prompt and exit")
    return p


def resolve_config(args: argparse.Namespace) -> Config:
    """Config defaults < saved settings < CLI flags (only those provided)."""
    cfg = Settings.apply(Config())
    overrides: dict[str, object] = {}
    if args.temp is not None:
        overrides["temperature"] = args.temp
    if args.model is not None:
        overrides["model"] = args.model
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    return replace(cfg, **overrides) if overrides else cfg


def _make_registry(config: Config) -> ToolRegistry:
    return ToolRegistry(build_default_tools(FileSystem(), config.observation_char_limit))


def cmd_show_config() -> int:
    saved = Settings.load()
    effective = Settings.apply(Config())
    print(f"settings file: {Settings.path()}")
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
    registry = _make_registry(config)
    system_prompt = SystemPromptBuilder(registry).build()

    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        os.chdir(workdir)
    workdir = Path.cwd()

    reporter = ConsoleReporter(color=not args.no_color)
    reporter.run_start(task, str(workdir), config)

    agent = RalphAgent(OllamaClient(config), registry, system_prompt, config, reporter=reporter)
    started = datetime.now()
    try:
        result = agent.run(task)
    except OllamaUnavailable as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    reporter.run_end(result)

    log_path = RunLogger(workdir).write(task, config, result, started)
    print(f" log: {log_path}")
    return 0 if result.finished else 2


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # robust unicode on Windows consoles
    except AttributeError:
        pass

    parser = build_parser()
    # Friendly help: bare `vibe`, `vibe help`, or `vibe -help` all show help.
    if not argv or argv[0] in ("help", "-help"):
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.show_config:
        return cmd_show_config()
    if args.reset_config:
        print("settings cleared." if Settings.reset() else "no saved settings to clear.")
        return 0
    if args.set:
        return cmd_set(args.set[0], args.set[1])
    if args.print_system:
        print(SystemPromptBuilder(_make_registry(Config())).build())
        return 0
    if not args.task:
        print("error: no task given.\n")
        parser.print_help()
        return 2

    return run_agent(args)
