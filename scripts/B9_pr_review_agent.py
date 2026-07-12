"""
B9_pr_review_agent.py — PR/code-review agent (real-world use case, Part 2).

Builds on the hardened base (B4) and the real filesystem/shell pattern (B5),
pointed at a different real problem: reviewing a git commit instead of fixing
a repo directly.

The orchestrator inspects a real `git diff` in a real (scratch) git repo,
delegates to a `ci-runner` sub-agent to run the test suite, then drafts a
review comment explaining what broke and why — posting that comment is a
mocked, HITL-gated tool (`post_review_comment`), same pattern as B4's
`send_email`. The orchestrator is instructed to comment, not silently fix the
file itself (unlike B5), which is closer to how a review bot should behave.

The demo repo (`workspace/pr-review-repo/`) is a real git repository with 2
commits: a working `is_even.py`, then a second commit that flips the
condition and breaks its test. It's bootstrapped automatically on first run
(see `ensure_demo_repo`) and gitignored — it's scratch state, not content to
commit, exactly like `vendor-deepagents/`.

Run (auto-approve demo):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B9_pr_review_agent.py

Run with manual approval (approve/reject via stdin):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B9_pr_review_agent.py --interactive
"""
import argparse
import os
import subprocess
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from scripts.B5_repo_ops_agent import resolve_decisions

load_dotenv()

MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")
REPO_ROOT = Path(__file__).parent.parent
DEMO_REPO = REPO_ROOT / "workspace" / "pr-review-repo"

IS_EVEN_OK = "def is_even(n):\n    return n % 2 == 0\n"
IS_EVEN_BROKEN = "def is_even(n):\n    return n % 2 == 1\n"
TEST_IS_EVEN = "from is_even import is_even\n\n\ndef test_is_even():\n    assert is_even(4) is True\n    assert is_even(3) is False\n"


def _git(*args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=demo", "-c", "user.email=demo@example.com", *args],
        cwd=DEMO_REPO,
        check=True,
        capture_output=True,
    )


def ensure_demo_repo() -> None:
    """Bootstrap a scratch git repo with a working commit + a breaking commit.

    Idempotent — does nothing if the repo already exists. Not committed to
    the outer deepagents-guide repo (see .gitignore): this is a real, throwaway
    git repository, not something you'd want nested inside another one.
    """
    if (DEMO_REPO / ".git").exists():
        return
    DEMO_REPO.mkdir(parents=True, exist_ok=True)
    _git("init", "-q")
    (DEMO_REPO / "is_even.py").write_text(IS_EVEN_OK)
    (DEMO_REPO / "test_is_even.py").write_text(TEST_IS_EVEN)
    _git("add", ".")
    _git("commit", "-q", "-m", "Add is_even with a passing test")
    (DEMO_REPO / "is_even.py").write_text(IS_EVEN_BROKEN)
    _git("commit", "-q", "-am", "Flip the parity check (introduces a bug)")


def post_review_comment(comment: str) -> str:
    """Post a review comment on the pull request (mock)."""
    return f"[MOCK] Posted review comment:\n{comment}"


CI_RUNNER_SUBAGENT = {
    "name": "ci-runner",
    "description": "Delegate to this agent to run the test suite and report raw output.",
    "system_prompt": (
        "You are a CI runner. You ONLY run test commands via the execute tool "
        "(e.g. `python3 -m pytest -q`). You never edit files. Report the raw "
        "output back to the orchestrator."
    ),
}

SYSTEM_PROMPT = (
    "You are a PR review assistant. Look at the diff introduced by the latest "
    "commit (`git diff HEAD~1 HEAD`), delegate to the ci-runner sub-agent to run "
    "the test suite, and diagnose what broke. Then call post_review_comment "
    "with a concise comment explaining the bug and suggesting a fix. Do NOT "
    "edit the file yourself — your job is to review, not to fix."
)


def build_pr_review_agent(model: str, root_dir: Path, checkpointer):
    backend = LocalShellBackend(root_dir=root_dir, virtual_mode=True)
    return create_deep_agent(
        model=model,
        backend=backend,
        subagents=[CI_RUNNER_SUBAGENT],
        tools=[post_review_comment],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        interrupt_on={"post_review_comment": True, "write_file": True, "edit_file": True, "execute": True},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Approve manually via stdin")
    args = parser.parse_args()

    ensure_demo_repo()
    agent = build_pr_review_agent(MODEL, DEMO_REPO, MemorySaver())

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    user_msg = "Review the latest commit and post a review comment on what's wrong."
    print(f">>> repo root = {DEMO_REPO}")
    print(f">>> User: {user_msg}\n")

    agent.invoke({"messages": [{"role": "user", "content": user_msg}]}, config=config)

    state = agent.get_state(config)
    while state.interrupts:
        interrupt_value = state.interrupts[0].value
        decisions = resolve_decisions(interrupt_value["action_requests"], args.interactive)
        agent.invoke(Command(resume={"decisions": decisions}), config=config)
        state = agent.get_state(config)

    messages = state.values.get("messages", [])
    final = messages[-1] if messages else None
    print(f"\n>>> Agent: {getattr(final, 'content', final)}")


if __name__ == "__main__":
    main()
