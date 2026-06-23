"""LLM client.

The agent depends on the `LLMClient` abstraction (DIP); `OllamaClient` is one
implementation. It performs the two-phase generation that converts noperator's
vLLM structural-tag idea to Ollama:
  phase 1 - free reasoning, stopped at </think>  (discarded by the caller)
  phase 2 - raw continuation prefilled past </think>, constrained by a JSON
            schema via Ollama's `format` field -> a guaranteed-valid action.
"""
from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import Config


@dataclass(frozen=True)
class Decision:
    reasoning: str        # phase-1 text (kept only for logging/inspection)
    action_json: str      # phase-2 constrained JSON (the actual action)


class LLMClient(ABC):
    @abstractmethod
    def decide(self, system: str, user: str, action_schema: dict) -> Decision:
        ...


class OllamaClient(LLMClient):
    def __init__(self, config: Config):
        self._cfg = config

    # ---- public ----
    def decide(self, system: str, user: str, action_schema: dict) -> Decision:
        reasoning = self._reason(system, user)
        action = self._act(system, user, reasoning, action_schema)
        return Decision(reasoning=reasoning, action_json=action)

    # ---- phase 1: free reasoning, stop at </think> ----
    def _reason(self, system: str, user: str) -> str:
        data = self._post("/api/chat", {
            "model": self._cfg.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {**self._options(), "num_predict": self._cfg.reason_tokens,
                        "stop": ["</think>"]},
        })
        return data["message"]["content"]

    # ---- phase 2: constrained action via raw continuation ----
    def _act(self, system: str, user: str, reasoning: str, action_schema: dict) -> str:
        prompt = self._render_chatml(system, user) + self._continue_after_reasoning(reasoning)
        data = self._post("/api/generate", {
            "model": self._cfg.model,
            "stream": False,
            "raw": True,
            "prompt": prompt,
            "format": action_schema,
            "options": {**self._options(), "num_predict": self._cfg.action_tokens,
                        "stop": ["<|im_end|>"]},
        })
        return data["response"].strip()

    # ---- helpers ----
    def _options(self) -> dict:
        c = self._cfg
        return {"temperature": c.temperature, "top_p": c.top_p, "top_k": c.top_k,
                "num_ctx": c.num_ctx, "num_gpu": c.num_gpu}

    @staticmethod
    def _render_chatml(system: str, user: str) -> str:
        # Matches the Qwen2.5 ChatML template Ollama applies for this model.
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @staticmethod
    def _continue_after_reasoning(reasoning: str) -> str:
        """Re-open the assistant turn at the point right after </think>, so the
        constrained JSON is generated as the post-reasoning answer."""
        if "<think>" in reasoning and "</think>" not in reasoning:
            return f"{reasoning}</think>\n"
        if not reasoning.strip():
            return ""           # model produced no reasoning; just constrain directly
        return f"{reasoning}\n"

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self._cfg.ollama_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._cfg.request_timeout) as resp:
            return json.loads(resp.read())
