"""vibeharness - a minimal Ralph-loop agent harness for small local models.

A small, SOLID set of pieces:
  - tools / fs_tools : the Tool interface and concrete filesystem tools
  - registry         : single source of truth -> docs + action JSON schema
  - prompt           : builds the system prompt and per-turn task prompt
  - memory           : natural-language narrative of past actions
  - llm              : LLM client interface + Ollama two-phase implementation
  - agent            : the Ralph loop orchestrator
"""

__version__ = "0.1.0"
