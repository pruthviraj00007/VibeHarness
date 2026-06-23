"""Runtime configuration. One immutable value object passed where needed (DIP-friendly)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # model / sampling
    model: str = "vibethinker"
    temperature: float = 0.3          # 0.3 gave clean content in our experiments
    top_p: float = 0.95
    top_k: int = 0
    num_ctx: int = 16384
    num_gpu: int = 99                 # force full GPU offload (NVIDIA via CUDA)

    # loop
    max_steps: int = 15

    # per-turn token budgets
    reason_tokens: int = 2048         # phase 1 (free reasoning, discarded)
    action_tokens: int = 1024         # phase 2 (constrained JSON action)

    # observation rendering
    observation_char_limit: int = 1500  # truncate big tool outputs in the narrative

    # backend
    ollama_url: str = "http://127.0.0.1:11434"
    request_timeout: int = 600
