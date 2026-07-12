"""
test_repo_ops_eval.py — Lightweight eval: does the B5 agent still fix the bug after swapping models?

This is NOT a unit test for your code — it calls a real model API (costs
tokens) to check the AGENT's behavior. Purpose: when you change
`DEEPAGENTS_MODEL` (e.g. from gpt-4o-mini to a cheaper model), this test
tells you immediately whether the agent can still diagnose + fix the bug,
instead of having to manually run B5 and read the log.

Unlike the full eval suite in `vendor-deepagents/libs/evals` (which requires
LangSmith tracing, its own registry, radar charts, etc.), this file only uses
plain pytest — it runs right away, no extra setup needed. If you have
LANGSMITH_TRACING=true (Step 7 in the README), traces from these eval runs
also show up on LangSmith automatically since that's just an env var.

Each test case:
  1. Copies `workspace/sample-repo/` (intentional bug) into its own tmp
     directory (doesn't touch the demo copy).
  2. Builds the B5 agent (reusing `build_repo_ops_agent`/`run_to_completion`
     from `scripts/B5_repo_ops_agent.py` — no duplicated logic).
  3. Runs the agent with auto-approve (same as demo mode, no human needed).
  4. Runs real pytest in the tmp directory, asserts exit code == 0.

Run:
  env -u PYTHONPATH uv run --no-sync .venv/bin/python -m pytest evals/ -v

Change the list of models to evaluate via an env var (default: B2's model + B3's researcher model):
  EVAL_MODELS="openrouter:openai/gpt-4o-mini,openrouter:deepseek/deepseek-chat-v3"
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from scripts.B5_repo_ops_agent import build_repo_ops_agent, run_to_completion

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent
SAMPLE_REPO = REPO_ROOT / "workspace" / "sample-repo"

DEFAULT_MODELS = (
    os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini"),
    os.getenv("RESEARCHER_MODEL", "openrouter:deepseek/deepseek-chat-v3"),
)
MODELS = tuple(
    dict.fromkeys(  # de-duplicate, preserve order, in case the 2 defaults are the same
        m.strip()
        for m in os.getenv("EVAL_MODELS", ",".join(DEFAULT_MODELS)).split(",")
        if m.strip()
    )
)

requires_openrouter_key = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="This eval calls real models via OpenRouter — needs OPENROUTER_API_KEY in .env",
)

USER_MSG = (
    "Run the test suite. If anything fails, diagnose and fix the bug "
    "with a minimal edit, then rerun the tests to confirm it passes."
)


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Copy sample-repo (intentional bug) into its own tmp directory for each test case."""
    dest = tmp_path / "sample-repo"
    shutil.copytree(SAMPLE_REPO, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return dest


@requires_openrouter_key
@pytest.mark.parametrize("model", MODELS)
def test_agent_fixes_bug(model: str, tmp_repo: Path) -> None:
    agent = build_repo_ops_agent(model, tmp_repo, MemorySaver())
    config = {"configurable": {"thread_id": f"eval-{model}"}}

    run_to_completion(agent, config, USER_MSG, interactive=False)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=tmp_repo,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"[{model}] agent did not leave the repo in a passing state:\n"
        f"{result.stdout}\n{result.stderr}"
    )
