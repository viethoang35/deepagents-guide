# Deep Agents Hands-on Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![LangChain](https://img.shields.io/badge/built%20with-LangChain%20%2F%20LangGraph-1C3C3C)
![Status](https://img.shields.io/badge/status-actively%20verified-brightgreen)

A hands-on, **actually-run** guide to [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) —
not slides, not theory. Every script here has been executed against a real model API and its
output is documented as-is, including the times it didn't work.

It goes from a single agent with one tool, to the things a real deployment needs and tutorials
usually skip: a durable checkpointer, human-in-the-loop approval gates, a real filesystem/shell
agent that fixes real bugs, cross-thread long-term memory, and a regression eval that caught a
real difference in behavior between models. Then it applies that hardened base to 3 real
problems: reviewing a git commit, triaging support tickets against a knowledge base, and
triaging ops/infra logs.

**Read `README.md` top to bottom** — Part 1 (B1–B8) builds the harness step by step; Part 2
(B9–B11) applies it to real problems. Verified on macOS aarch64, Python 3.12, uv 0.11.

## Structure
```
deepagents-guide/
├── pyproject.toml          # deps: deepagents, langchain, langgraph, provider packages
├── .env.example            # copy to .env, fill in one model key
├── scripts/
│   ├── B2_quickstart.py    # single agent + 1 custom tool (verified through a real API call)
│   ├── B2_advanced.py      # sub-agent (dict spec) + filesystem + streaming
│   ├── B3_nemotron_deep_agent.py  # multi-model: orchestrator + researcher sub-agent (OpenRouter)
│   ├── B4_hardened_agent.py       # durable checkpointer + human-in-the-loop + lightweight observability
│   ├── B5_repo_ops_agent.py       # agent reads/edits REAL files + sub-agent runs REAL pytest (on the hardened base)
│   ├── B6_cross_thread_memory.py  # memory=[...]: remembers ACROSS threads (unlike B4's checkpointer, which only remembers within 1 thread)
│   ├── B9_pr_review_agent.py      # real-world: reviews a real git diff, posts a HITL-gated review comment
│   ├── B10_support_triage_agent.py # real-world: answers support tickets grounded in a local knowledge base, escalates what it can't answer
│   └── B11_ops_infra_triage_agent.py # real-world: triages real log files, files a HITL-gated incident; enforces read-only via FilesystemPermission
├── evals/
│   └── test_repo_ops_eval.py      # pytest: does the B5 agent still fix the bug correctly when you swap models?
├── vendor-deepagents/      # clone of langchain-ai/deepagents (e.g. nvidia_deep_agent example)
└── vendor-nemoclaw/        # clone of NVIDIA/nemoclaw-community (secure blueprint)
```

`vendor-deepagents/` and `vendor-nemoclaw/` are **not** committed to this repo (see
`.gitignore`) — they're clones of other people's repos, re-clone them when needed:
```bash
git clone https://github.com/langchain-ai/deepagents.git vendor-deepagents
git clone https://github.com/NVIDIA/nemoclaw-community.git vendor-nemoclaw
```

## Quickstart (B1 + B2)
```bash
uv sync                                   # B1: install deps (already synced on this machine)
cp .env.example .env                      # create the env file
# OPEN .env and fill in: OPENROUTER_API_KEY=sk-or-... (required to run B2/B3)
# NOTE: if your shell has PYTHONPATH pointing at another venv (e.g. a "Hermes" environment), use env -u PYTHONPATH
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B2_quickstart.py
```

## Models via OpenRouter (1 key, many models)
Deep Agents supports the `openrouter:` prefix. Format: `openrouter:<provider>/<model>`.
Example in `.env`:
```
OPENROUTER_API_KEY=sk-or-...
DEEPAGENTS_MODEL=openrouter:deepseek/deepseek-chat-v3
# or: openrouter:x-ai/grok-4.5
# or: openrouter:openai/gpt-4o-mini
```
Register a key at: https://openrouter.ai/keys

## Step 3 — Multi-model Deep Agent (Nemotron style, via OpenRouter)
A trimmed-down version of the official `vendor-deepagents/examples/nvidia_deep_agent` example, but
calling both models through OpenRouter (1 key). Orchestrator = strong model (grok-4.5), researcher
sub-agent = cheap/fast model (deepseek).
```bash
# .env: ORCHESTRATOR_MODEL + RESEARCHER_MODEL (already present in .env.example)
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B3_nemotron_deep_agent.py
```
`.env` variables for B3:
```
ORCHESTRATOR_MODEL=openrouter:x-ai/grok-4.5
RESEARCHER_MODEL=openrouter:deepseek/deepseek-chat-v3
```

## Step 4 — Harden the harness (checkpointer + human-in-the-loop + observability)
`B4_hardened_agent.py` reuses the simple orchestrator but adds 3 things a real environment needs:
- **Durable checkpointer** (`SqliteSaver`): state is saved to `workspace/b4_checkpoints.db`, so the
  agent resumes the correct thread after the process is stopped and restarted.
- **Human-in-the-loop** (`interrupt_on`): tools with real side effects (`send_email`, `delete_file`)
  pause for a human decision (`approve`/`reject`) before running; safe tools (`get_weather`) run straight through.
- **Lightweight observability**: a local callback handler logs tool calls + token usage, no LangSmith
  account needed. If you do have LangSmith, just set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`
  in `.env` — no code changes.

```bash
# Auto-approve demo (no manual input needed):
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B4_hardened_agent.py

# Manually approve each sensitive tool call (approve/reject via stdin):
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B4_hardened_agent.py --interactive

# Resume an old thread (thread_id printed on a previous run):
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B4_hardened_agent.py --thread-id <id>
```

## Step 5 — Repo-ops agent (real filesystem + real shell, on the hardened base)
`B5_repo_ops_agent.py` uses `LocalShellBackend` (real filesystem + real shell on your machine,
confined to `workspace/sample-repo/` via `virtual_mode=True` + `root_dir`) instead of B2's simulated
virtual filesystem. The orchestrator delegates to a `runner-agent` sub-agent that runs real `pytest`;
if a test fails, the orchestrator reads the file itself, fixes the bug with `edit_file`, then reruns
the tests to confirm. Every `write_file`/`edit_file`/`execute` call is blocked pending approval
(reusing the exact HITL mechanism from Step 4).

`workspace/sample-repo/` contains 1 intentional bug (`calc.py` subtracts an extra `1` in
`average()`) so the agent has something real to diagnose — if you find it already "clean" (no bug),
that's because a previous run's agent already fixed it; the file gets reset so the demo means
something again.

⚠️ `LocalShellBackend` has **no sandbox** — `execute` runs real shell commands with your
permissions and is NOT constrained by `virtual_mode` (only the filesystem tools are constrained).
The only safety layer here is HITL. For real isolation (Docker/VM), see
`vendor-deepagents/libs/partners/` (modal, runloop, daytona) and extend `BaseSandbox`.

```bash
# Auto-approve demo:
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B5_repo_ops_agent.py

# Manually approve each write_file/edit_file/execute call:
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B5_repo_ops_agent.py --interactive
```

## Step 6 — Cross-thread memory (different from the checkpointer)
`B6_cross_thread_memory.py` demonstrates `memory=[...]`, easily confused with the checkpointer (B4)
but a different mechanism:
- **Checkpointer** (B4): remembers *within the same thread* — message history, resumed via `thread_id`.
- **`memory=[...]`** (this one): remembers *across threads* — a REAL `AGENTS.md` file on disk
  (`workspace/agent_memory/AGENTS.md`), loaded into the system prompt at the start of EVERY
  conversation, which the agent updates itself via `edit_file` when it learns something new (gated
  by HITL approval).

The script runs 2 "conversations" with completely different `thread_id`s: conversation A teaches
the agent a preference (answer in bullet points), conversation B (a new, unrelated thread) still
answers according to that preference — proving the memory lives outside the scope of a single thread.

```bash
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B6_cross_thread_memory.py
```

## Step 7 (optional) — LangSmith tracing
Applies to **every** script above with zero code changes — LangChain reads this env var
automatically. Set it in `.env` (see `.env.example`):
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=deepagents-guide
```
Then run any script (B2-B5) as usual — the full trajectory (tool calls, sub-agent delegation,
per-step token usage) will show up on smith.langchain.com under that project name, which is much
easier to debug than reading console logs. Most useful for B3/B5 since they have several
orchestrator ↔ sub-agent steps that are hard to follow via print statements.

Note: this has only been verified as *mechanically correct* (a standard LangChain env var, no code
required) — it hasn't been tested end-to-end with a real LangSmith key since this machine doesn't have one.

## Step 8 — Eval: does the B5 agent still fix the bug correctly after swapping models?
`evals/test_repo_ops_eval.py` is a plain pytest file (no LangSmith required, unlike the full eval
suite in `vendor-deepagents/libs/evals`, which *requires* LangSmith tracing). Each test case: copies
`workspace/sample-repo/` (intentional bug) into its own tmp directory, runs the B5 agent (reusing
`build_repo_ops_agent`/`run_to_completion`, auto-approve), then runs real `pytest` in that tmp
directory and asserts it passes.

```bash
env -u PYTHONPATH uv run --no-sync .venv/bin/python -m pytest evals/ -v

# Change which models get evaluated:
EVAL_MODELS="openrouter:openai/gpt-4o-mini,openrouter:x-ai/grok-4.5" \
  env -u PYTHONPATH uv run --no-sync .venv/bin/python -m pytest evals/ -v
```

Actually verified: running with the 2 default models (`gpt-4o-mini` and `deepseek-chat-v3`) —
**`gpt-4o-mini` passed but `deepseek-chat-v3` failed** — the deepseek-driven agent didn't finish
fixing the bug in that run. This is exactly the value of this eval: you find out immediately when
swapping models breaks something, instead of manually running B5 and reading the log step by step.

## Part 2 — Real-world applications

B1–B8 build the harness. These 3 apply it to actual problems, each reusing the same HITL/
checkpointer patterns from Part 1 but pointed at a different domain.

## Step 9 — PR/code-review agent
`B9_pr_review_agent.py` reviews a real git commit instead of fixing a repo directly (unlike B5).
The orchestrator reads a real `git diff` in a scratch git repo (auto-bootstrapped on first run,
not committed — see `.gitignore`), delegates to a `ci-runner` sub-agent to run the tests, then
drafts a review comment — `post_review_comment` is a mocked, HITL-gated tool, same pattern as
B4's `send_email`. It's instructed to comment, not silently fix the file itself.

```bash
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B9_pr_review_agent.py
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B9_pr_review_agent.py --interactive
```

Verified live: the demo repo's second commit flips `is_even()`'s condition, breaking its test.
The agent read the diff, correctly diagnosed the bug, and posted a comment naming the exact fix —
without being told to run the test suite first (it inferred the bug straight from the diff, which
is a legitimate reviewer move, not a script bug).

## Step 10 — Support-ticket triage agent (retrieval)
`B10_support_triage_agent.py` fills the one gap B1–B8 never touched: retrieval. It answers
tickets grounded in a local knowledge base (`workspace/support-docs/`, the same entries as this
README's "Common errors & fixes" section below), and escalates to a human — `escalate_to_human`
is HITL-gated — when nothing relevant is found instead of guessing.

`search_docs` is a plain word-overlap ranking over local markdown files, **not** a real
embedding-based vector store — deliberate, so the demo needs no embeddings API key or new
dependency while still teaching the retrieval-augmented pattern. Swap in a real vector store
(e.g. Chroma + embeddings) for production use.

```bash
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B10_support_triage_agent.py
```

Verified live with 2 tickets: one covered by the docs (PYTHONPATH error) got a grounded answer;
one not covered (a Kubernetes deployment request) correctly triggered `escalate_to_human` instead
of a made-up answer.

## Step 11 — Ops/infra triage agent (enforced read-only)
`B11_ops_infra_triage_agent.py` explores real log files (`workspace/ops-logs/`) with the built-in
read-only tools (`ls`/`grep`/`glob`/`read_file`), looks for anomalies across services, and files
an incident — `create_incident` is HITL-gated, same pattern throughout Part 2.

The new mechanism here: read-only access is enforced by `permissions=[FilesystemPermission(
operations=["write"], paths=["/**"], mode="deny")]`, not just a system-prompt instruction. This
denies `write_file`/`edit_file` for every path regardless of what the model attempts.

```bash
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B11_ops_infra_triage_agent.py
```

Verified live, twice: (1) the agent correctly found the real anomaly — a burst of `db-primary`
connection timeouts in `api-service.log` — while ignoring the unrelated, normal
`worker-service.log`, and filed a `high`-severity incident. (2) A direct adversarial test —
instructing the agent to call `write_file` on `/hacked.txt`, then retrying `/tmp/hacked.txt` —
got `Error: permission denied for write` both times, and no file was created on disk.

## Common errors & fixes
- `No module named 'pydantic_core._pydantic_core'`: caused by an inherited PYTHONPATH from another venv.
  Fix: run with `env -u PYTHONPATH` or unset PYTHONPATH first.
- `ModuleNotFoundError: langchain_openai`: the provider package wasn't installed → already added to pyproject.
- `TypeError: ... unexpected keyword argument 'sub_agents'`: the real API uses `subagents` (no underscore).
- `KeyError: 'system_prompt'` when passing a sub-agent: use a **dict spec**
  `{"name":..., "system_prompt":..., "model":..., "tools":[...]}` (not a `SubAgent(agent=...)` object).

## Advanced examples (from vendor)
- Multi-model + GPU skills: `vendor-deepagents/examples/nvidia_deep_agent/`
- Secure blueprint (Deep Agents + OpenShell): `vendor-nemoclaw/examples/harness-engineering-playground/`
