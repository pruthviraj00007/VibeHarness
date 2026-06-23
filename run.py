"""Thin wrapper so the agent can be launched without installing the package:

    python run.py "<task>"

All logic lives in ``vibeharness.cli`` (also exposed as the ``vibe`` console
script and via ``python -m vibeharness``).
"""
from vibeharness.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
