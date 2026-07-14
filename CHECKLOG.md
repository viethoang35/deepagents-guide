# CHECKLOG

A running log of changes made to the `deepagents-guide` project, in chronological order.
Goal: let anyone (including future-you) understand where the repo has been without having
to re-read the entire chat history.

## 2026-07-13 (branch `tracing-labs`)

5 scripts extending LangSmith/Phoenix tracing were authored with the help of an external tool
("hermes-agent") and reviewed/fixed here to match this repo's standards.

### Fixed — numbering collision
- `scripts/B9_cost_tracking.py` renamed to **`scripts/B12_cost_tracking.py`**. `B9` was already
  taken by `B9_pr_review_agent.py`/`B9_pr_review_agent_live.py` on `main` — the new scripts were
  authored on a branch cut after that work landed, without checking for the collision.
- `B7_langsmith_trace.py`, `B7b_multi_agent_trace.py`, `B8_langsmith_eval.py`,
  `B8b_phoenix_eval.py` did **not** collide (Step 7 had no script before, Step 8's existing script
  is `evals/test_repo_ops_eval.py`, not `B8_*.py`) — kept as-is, folded into the existing Step
  7/8 README sections as additional variants instead of getting new step numbers.

### Fixed — real bugs found by live-running each script
- **`B7b_multi_agent_trace.py`**: crashed (`UnauthorizedResponseError`) on any real-looking
  `OPENROUTER_API_KEY`, including an invalid one — the "is this a fake key" check only matched
  specific placeholder strings, with no exception handling around the actual `model.invoke()`
  call. Also: `researcher()` ignored the graph state entirely and asked a hardcoded question,
  so the "handoff" the script exists to demonstrate wasn't actually wired up. Fixed both.
- **`B12_cost_tracking.py`** (formerly B9): identical crash, identical root cause. Added a
  try/except around the real invoke call.
- **`B8_langsmith_eval.py`**: the feedback-attachment step searched for the run via
  `client.list_runs(filter=f'{{"thread_id": "..."}}')` — confirmed against the installed
  `langsmith` SDK's own docstring that `filter` expects a query DSL string (e.g.
  `'eq(run_type, "chain")'`), not JSON, so this would never actually match a run. Replaced with
  an explicit `run_id` passed through the invoke `config` and reused directly with
  `create_feedback(run_id, ...)` — no search needed.
- **`B8b_phoenix_eval.py`**: an infrastructure error (e.g. a bad API key) recorded the same
  `pytest_pass=0.0` as a genuine model failure, which would misleadingly read as "both models
  failed the task." Now recorded as `eval.status="error"` with the exception message, excluded
  from the pass/fail comparison.
- (Introduced and immediately caught during my own fix) `key.startswith(FAKE_KEY_PREFIXES)`
  where the tuple included `""` — `str.startswith` treats every string as starting with `""`,
  so this always evaluated to "fake." Removed the empty-string entry.

### Verified live (not just "offline-verified" as originally committed)
- Root cause of the original crashes: `OPENROUTER_API_KEY` in `.env` had expired
  ("User not found" from OpenRouter) — replaced with a valid key first.
- `B7_langsmith_trace.py`, `B7b_multi_agent_trace.py`, `B12_cost_tracking.py`: real OpenRouter
  calls, real output, no crashes.
- `B8b_phoenix_eval.py`, `B12_cost_tracking.py`: started a real local Phoenix server and
  confirmed the `eval_run`/`cost_run` spans actually landed by querying Phoenix's own REST API
  directly — not just trusting each script's "wrote a span" print statement.
- Still unverified (no LangSmith key on this machine, same caveat as the original Step 7):
  whether traces/feedback/datasets actually reach smith.langchain.com for
  `B7_langsmith_trace.py`, `B7b_multi_agent_trace.py`, `B8_langsmith_eval.py`.

### Updated
- All 5 scripts translated from Vietnamese to English (docstrings, comments, print/log
  strings) to match the project's English-only convention for repo content.
- `README.md`: Step 7 now documents `B7_langsmith_trace.py`/`B7b_multi_agent_trace.py`; Step 8
  now documents `B8_langsmith_eval.py`/`B8b_phoenix_eval.py` alongside the original pytest eval;
  added a new Step 12 for `B12_cost_tracking.py`; updated the structure tree.
- `.env.example`: removed the duplicated/redundant `LANGSMITH_TRACING` vs `LANGCHAIN_TRACING_V2`
  block (kept `LANGCHAIN_TRACING_V2` as the one documented variable), added
  `LANGSMITH_EVAL_DATASET`, translated the remaining Vietnamese comment to English.
- `pyproject.toml`/`uv.lock`: `arize-phoenix` + `arize-phoenix-otel` were added to
  `pyproject.toml` in an earlier commit but `uv.lock` was only updated locally — committed the
  matching lock file.

## 2026-07-12

### Added — scripts (agent)

- **`scripts/B4_hardened_agent.py`** — 3 hardening layers for the harness:
  - Durable checkpointer (`SqliteSaver` → `workspace/b4_checkpoints.db`), correctly resumes
    a thread after the process is stopped and restarted.
  - Human-in-the-loop (`interrupt_on`) for tools with side effects (`send_email`,
    `delete_file`); safe tools (`get_weather`) run straight through without approval.
  - Lightweight observability: a local callback handler logs tool calls + token usage,
    no LangSmith account needed.
  - Verified live: ran both auto-approve and interactive modes, confirmed resuming an old
    `thread_id` reads back the correct history, and that rejecting an action actually blocks it.

- **`scripts/B5_repo_ops_agent.py`** — Repo-ops agent on the hardened base:
  - Uses `LocalShellBackend` (real filesystem + real shell, confined to
    `workspace/sample-repo/` via `virtual_mode=True`).
  - `runner-agent` sub-agent runs real `pytest`; the orchestrator reads/fixes the file
    itself when a test fails, then reruns to confirm.
  - Refactored into `build_repo_ops_agent()` + `run_to_completion()` so
    `evals/test_repo_ops_eval.py` can reuse them instead of duplicating the logic.
  - Verified live: ran with the default model, watched the agent self-recover when `python`
    wasn't on PATH, install `pytest` via `ensurepip`, correctly diagnose and fix the bug, and
    confirm the test passes. Also tested the reject path via `--interactive`.
  - ⚠️ Safety note: `LocalShellBackend` has no sandbox — `execute` is not constrained by
    `virtual_mode`. The only safety layer is HITL.

- **`scripts/B6_cross_thread_memory.py`** — Cross-thread memory (`memory=[...]`), distinct
  from the checkpointer (B4):
  - The checkpointer remembers *within one thread*; `memory=[...]` remembers *across
    threads* via a real `AGENTS.md` file on disk (`workspace/agent_memory/AGENTS.md`).
  - Verified live: thread A teaches the agent a preference, thread B (a new, unrelated
    `thread_id`) correctly applies that preference.

### Added — eval

- **`evals/test_repo_ops_eval.py`** — a plain pytest file (no LangSmith needed, unlike the
  full eval suite in `vendor-deepagents/libs/evals`), copies `sample-repo` into its own tmp
  directory, runs the B5 agent with a parametrized model, asserts real `pytest` passes afterward.
  - Verified with the 2 default models: **`gpt-4o-mini` passed, `deepseek-chat-v3` failed** —
    a real, observed finding, not a hypothetical one — exactly the point of this eval.

### Added — demo data

- `workspace/sample-repo/{calc.py,test_calc.py}` — an intentional bug (`average()` subtracts
  an extra 1) so B5/the eval have something real to diagnose. Left in its "buggy" state after
  each demo so the next run still has something meaningful to fix.
- `workspace/agent_memory/AGENTS.md` — generated by a live B6 demo run (real memory content).

### Added — Part 2: real-world applications (B9–B11)

- **`scripts/B9_pr_review_agent.py`** — PR/code-review agent:
  - Reviews a real `git diff` in a scratch git repo (`workspace/pr-review-repo/`, auto-bootstrapped
    by `ensure_demo_repo()`, gitignored — not committed, same reasoning as the vendor dirs).
  - `ci-runner` sub-agent runs real `pytest`; `post_review_comment` (mock) is HITL-gated.
  - Verified live: correctly diagnosed a flipped condition in `is_even()` from the diff alone and
    posted an accurate review comment — it skipped delegating to `ci-runner` in that run, which is
    a legitimate model choice (the bug was visible directly in the diff), not a script defect.

- **`scripts/B10_support_triage_agent.py`** — Support-ticket triage agent, the first use of
  retrieval anywhere in this repo:
  - `search_docs` does plain word-overlap ranking over `workspace/support-docs/` (markdown docs
    mirroring this README's own troubleshooting section) — deliberately not a real embedding
    vector store, to avoid needing a new API key/dependency for the demo.
  - `escalate_to_human` (mock) is HITL-gated.
  - Verified live with 2 tickets: a documented error got a grounded answer; an out-of-scope
    request (Kubernetes deployment) correctly triggered escalation instead of a made-up answer.

- **`scripts/B11_ops_infra_triage_agent.py`** — Ops/infra triage agent:
  - Explores real log files (`workspace/ops-logs/`) with built-in read-only tools; `create_incident`
    (mock) is HITL-gated.
  - First real use of the `permissions` param: `FilesystemPermission(operations=["write"],
    paths=["/**"], mode="deny")` enforces read-only regardless of what the model attempts.
  - Verified live: correctly found a real anomaly (a `db-primary` timeout burst in
    `api-service.log`) while ignoring the unrelated, normal `worker-service.log`, and filed a
    `high`-severity incident. Separately ran an adversarial test — explicitly instructing the
    agent to `write_file` to `/hacked.txt`, then `/tmp/hacked.txt` — both attempts returned
    `permission denied` and no file was created on disk.

### Updated

- `pyproject.toml`: added `langgraph-checkpoint-sqlite` (durable checkpointer for B4/B5) and
  `pytest` (to run `evals/`).
- `.env.example`: added optional LangSmith variables (`LANGSMITH_TRACING`,
  `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`).
- `.gitignore`: added `workspace/pr-review-repo/` (scratch git repo bootstrapped by B9, not meant
  to be committed — same reasoning as the vendor dirs).
- `README.md`: added Steps 4–8 (hardened harness, repo-ops agent, cross-thread memory, LangSmith
  tracing, eval script) plus a new "Part 2 — Real-world applications" section for Steps 9–11,
  updated the directory structure tree.
- `LICENSE`: added MIT license (copyright viethoang35) ahead of publishing publicly.
- All Vietnamese comments/docstrings/prose across the repo (README, this file, `.env.example`,
  every script, the eval) translated to English so the project is readable by a wider audience.

### Published

- Pushed to `https://github.com/viethoang35/deepagents-guide` (private, with a plan to switch to
  public later). Description + topics set via the repo's About panel.

### Not yet verified / known limitations

- LangSmith tracing: confirmed *mechanically correct* (a standard LangChain env var, no code
  required), but **not** tested end-to-end with a real key since this machine doesn't have one.
- `LocalShellBackend` in B5/B9 has no real sandbox — real isolation (Docker/VM) requires
  extending `BaseSandbox` (see `vendor-deepagents/libs/partners/`: modal, runloop, daytona).
- `search_docs` in B10 is lexical (word-overlap), not semantic — won't handle a question phrased
  very differently from the docs' wording as well as real embeddings would.
