# VibeHarness

**A tiny, dependency-free Ralph-loop agent harness for small local models.**

VibeHarness turns a 3B local model ([VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B), served by [Ollama](https://ollama.com)) into a basic command-line coding agent. You type a task; it works on your current directory one step at a time — reading, writing, searching, and managing files — and streams its reasoning and actions live to the terminal.

```powershell
cd C:\my\project
vibe "Create a CHANGELOG.md and seed it with an Unreleased section."
```

```
 vibe (vibethinker, temp 0.3)
 workspace: C:\my\project
 task: Create a CHANGELOG.md and seed it with an Unreleased section.

┌─ step 1 ────────────────────────────────────────
│ thinking: <think>No changelog exists yet, so I'll create one…</think>
│ action: {"tool":"write_file","args":{"path":"CHANGELOG.md","content":"# Changelog\n\n## [Unreleased]\n"}}
└ ✓ you wrote the file 'CHANGELOG.md' (38 characters).

┌─ step 2 ────────────────────────────────────────
│ action: {"tool":"finish","args":{"summary":"Created CHANGELOG.md with an Unreleased section."}}
└ ✓ you finished the task: Created CHANGELOG.md with an Unreleased section.

 done in 2 steps — Created CHANGELOG.md with an Unreleased section.
```

> **Status:** working prototype. VibeThinker-3B is a math/reasoning specialist and is *not* tuned for tool use, so treat this as a research toy for studying small-model agentic behaviour — not a production agent.

---

## Why it's interesting

- **Constrained actions.** Each turn the model emits a JSON array of one or more tool calls, validated against a JSON schema *at decode time* (Ollama's `format` grammar), so it can't produce malformed actions — even at high temperature, where small models otherwise drift into garbage.
- **Two-phase turns.** The model first reasons freely, then emits the action(s) under the schema constraint. (This is an Ollama adaptation of [noperator's vLLM structural-tag trick](https://gist.github.com/noperator/6c711ab19027ea8056442df839f2d7e6).) The reasoning is dropped from the running context — but kept on disk (see below).
- **Natural-language memory.** Instead of a JSON/ChatML transcript, the agent's past is a plain-English narrative ("First, you… Then, you…"), which is what a small model follows most reliably.
- **Full reasoning logs.** Every run is written in full — *including* each turn's reasoning trace — to a hidden `.vibe/` folder in the workspace, so you can mine the traces to improve the prompt and model.
- **Small but powerful tools.** Six tools cover the filesystem; behaviour is widened with optional parameters (e.g. `write_file.mode` = overwrite/append/prepend) rather than by adding more tools.
- **Zero runtime dependencies.** Pure Python standard library.

---

## Prerequisites

You need three things: Python, Ollama, and a VibeThinker model registered in Ollama.

### 1. Python ≥ 3.10
Check with `python --version`. (Windows, macOS, and Linux are all supported; the harness is pure Python. Developed and tested on Windows 10.)

### 2. Ollama
Install from [ollama.com/download](https://ollama.com/download) and make sure the server is running:
```bash
ollama --version
ollama serve        # usually already running as a background service / tray app
```

### 3. The `vibethinker` model
VibeThinker ships as safetensors; Ollama needs GGUF. Pull a community GGUF and register it under the name `vibethinker` using the included [`Modelfile`](./Modelfile):
```bash
ollama pull hf.co/mradermacher/VibeThinker-3B-GGUF:Q8_0
ollama create vibethinker -f Modelfile
ollama run vibethinker "hi"      # quick sanity check
```
The `Modelfile` only points at the weights — the harness sets all sampling parameters per request.

### Hardware
The Q8_0 quant of this 3B model needs **~3.5 GB of VRAM** (or runs on CPU, just slower). Any modern GPU with ≥4 GB, or a CPU with ≥8 GB RAM, is fine.

---

## Install

### Option A — install the package (recommended; cross-platform `vibe` command)
```bash
git clone https://github.com/NickalasLight/VibeHarness.git
cd VibeHarness
pip install -e .          # creates the `vibe` console command
vibe "list this folder and tell me what's here"
```

### Option B — no install
```bash
git clone https://github.com/NickalasLight/VibeHarness.git
cd VibeHarness
python run.py "list this folder and tell me what's here"
# or:  python -m vibeharness "..."
```

### Windows convenience launcher
`bin\vibe.cmd` is a launcher that works in both PowerShell and CMD without `pip install`. Add the `bin` folder to your PATH (open a **new** terminal afterwards):
```powershell
[Environment]::SetEnvironmentVariable('Path',
  [Environment]::GetEnvironmentVariable('Path','User') + ';' + (Resolve-Path .\bin),
  'User')
```

---

## Usage

`vibe` runs **in your current terminal directory** — that directory is the agent's workspace.

```powershell
vibe "Create notes.txt containing 'hello hello hello', then read it back to verify."
vibe --temp 1.0 "draft a haiku into poem.txt"     # override temperature for one run
vibe --max-steps 30 "tidy up this folder"         # raise the step budget
vibe --workdir C:\some\other\dir "summarise it"   # operate elsewhere
```

### Commands & settings
```powershell
vibe --help                 # list every command and parameter
vibe --show-config          # show effective defaults + saved overrides
vibe --set temp 0.5         # persist a new default temperature
vibe --set max-steps 25     # persist a new default step budget
vibe --reset-config         # restore built-in defaults
vibe --print-system "x"     # print the generated system prompt
```
Persistent defaults live in `~/.vibeharness/settings.json` (override the location with the `VIBEHARNESS_HOME` env var). Resolution order is **built-in defaults < saved settings < per-run flags**. Settable keys: `temp`, `model`, `max-steps`, `top-p`, `top_k`. The built-in default temperature is `0.3`.

Each run is logged (with reasoning traces) to a hidden `.vibe/` folder in the workspace — see [Run logs](#run-logs-vibe).

---

## How it works

Each turn is two model calls — reason, then act under a schema constraint:

```
 task + natural-language narrative of past actions
        │
        ▼
 ┌─ phase 1: free reasoning ──────────┐   /api/chat, stop at </think>
 │  <think> … </think>                │   (streamed live, then discarded)
 └────────────────────────────────────┘
        │
        ▼
 ┌─ phase 2: constrained action(s) ───┐   /api/generate, raw continuation,
 │  [{"tool":"...","args":{...}}, …]   │   format = tools JSON schema (an array)
 └────────────────────────────────────┘
        │
        ▼
 parse → execute each action in order via ToolRegistry → append a
 plain-English observation per action
        │
        └──────────► repeat until `finish` or the step budget
```

A turn may **batch several actions** (e.g. write a file and read it back) when the
model is confident of the outcomes, or emit a single action when it needs the
result before deciding the next move.

### Run logs (`.vibe/`)
Each run writes two timestamped files into a hidden `.vibe/` folder in the
workspace:
- `<stamp>.json` — the complete structured log, **including every turn's reasoning
  trace**, the actions, results, and the config used.
- `<stamp>.md` — a readable transcript.

These are intended for analysis — diffing reasoning across runs, spotting where a
small model goes wrong, and tuning the prompt.

### Tools
| tool | purpose | key params |
|------|---------|-----------|
| `list_directory` | list a folder | `path?`, `recursive?` |
| `read_file` | read a file | `path` |
| `write_file` | create/modify a file | `path`, `content`, `mode?` (overwrite/append/prepend) |
| `search` | find text or filenames | `query`, `path?`, `target?` (content/filename/both), `max_results?` |
| `manage_path` | mkdir / delete / move | `action`, `path`, `destination?` |
| `finish` | end the task | `summary` |

The system prompt documents each tool in plain English **and** embeds the formal JSON schema — both generated from the single `ToolRegistry` source of truth, so docs and the enforced grammar can never drift apart.

### Architecture
```
vibeharness/
  tools.py       Tool interface + Param/ToolResult (docs & schema derived from params)
  filesystem.py  FileSystem service — the only code that touches the OS (SRP)
  fs_tools.py    concrete tools wrapping FileSystem, each self-describing
  registry.py    ToolRegistry — builds docs + action schema (OCP: add tools freely)
  prompt.py      system prompt + per-turn prompt builders
  memory.py      NarrativeMemory — the English account of past actions
  llm.py         LLMClient interface + OllamaClient two-phase streaming (DIP)
  reporting.py   Reporter interface + ConsoleReporter (live, colored output)
  agent.py       RalphAgent — the loop orchestrator
  settings.py    persistent user settings
  cli.py         argument parsing and command dispatch
run.py           no-install entrypoint   |   bin/vibe.cmd  Windows launcher
```
The design leans on small interfaces: the agent depends on `LLMClient` and `Reporter` abstractions, so the whole loop is testable with a fake client and a null reporter — no model required.

---

## Testing

```bash
python -m unittest discover -s tests -v     # standard library, no install needed
# or, if you prefer pytest:
pip install -e ".[dev]" && pytest -q
```
The suite (58 tests, runs in <0.1s) covers the filesystem service, every tool, schema generation, the settings store, the narrative memory, prompt building, the LLM helper functions, run logging, and the full agent loop (single- and multi-action turns) driven by a fake LLM client.

---

## Troubleshooting

- **`Could not reach Ollama …`** — the server isn't running. Start it with `ollama serve` (or launch the Ollama app).
- **It's slow / not using my GPU** — confirm with `ollama ps` (`PROCESSOR` should say `100% GPU`) and `nvidia-smi` (VRAM should be in use). On laptops with both an NVIDIA dGPU *and* an integrated GPU, Ollama's Vulkan backend may pick the iGPU; force CUDA with `setx OLLAMA_VULKAN 0` and restart the Ollama server.
- **Garbled / non-English tokens in output** — small models drift at high temperature; the *action* is always valid (schema-constrained), but lower `--temp` (e.g. 0.3) for cleaner reasoning and content.
- **No colors on Windows** — colors use ANSI; pass `--no-color` if your console doesn't render them.

---

## Acknowledgements

- [VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B) by WeiboAI (MIT).
- The constrained-decoding-after-reasoning idea is adapted from [noperator's structural-tag gist](https://gist.github.com/noperator/6c711ab19027ea8056442df839f2d7e6).
- [Ollama](https://ollama.com) for painless local model serving.

## License

[MIT](./LICENSE) © 2026 Nickalas Light
