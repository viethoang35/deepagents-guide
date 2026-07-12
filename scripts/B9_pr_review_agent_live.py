"""
B9_pr_review_agent_live.py — PR-review agent against a REAL GitHub pull request.

This is B9 productionized one step: instead of a scratch git repo and a mock
`post_review_comment`, this script reads a real PR's diff via the GitHub API
and posts a REAL comment on a REAL pull request. Test-running still happens
locally against your checked-out working tree (LocalShellBackend), same
pattern as B9/B5 — this script does not clone or check out the PR branch
for you.

Requirements:
  - GITHUB_TOKEN in .env: a fine-grained personal access token scoped to the
    target repo, with "Pull requests: Read and write" AND "Issues: Read and
    write" permissions (posting a PR comment uses the issue-comments endpoint).
  - The PR you pass in must already exist on GitHub.
  - Run from a working tree that already has the PR's changes checked out
    (e.g. you're on the PR branch, or the files already exist on disk) if you
    want the ci-runner sub-agent to be able to run tests against them.

Run:
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B9_pr_review_agent_live.py \\
    --repo owner/repo --pr 1 --test-dir workspace/live-pr-demo

Run with manual approval before the comment is actually posted:
  ... --interactive
"""
import argparse
import os
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from scripts.B5_repo_ops_agent import resolve_decisions

load_dotenv()

MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API = "https://api.github.com"


def _headers() -> dict:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is not set in .env — required to call the GitHub API.")
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def make_github_tools(repo: str, pr_number: int):
    """Build the get_pr_diff/post_pr_comment tool pair bound to 1 repo + PR.

    Closures instead of globals so the tools stay simple functions (the shape
    `create_deep_agent` expects) while still knowing which repo/PR to hit.
    """

    def get_pr_diff() -> str:
        """Fetch the real diff for this pull request from the GitHub API."""
        headers = {**_headers(), "Accept": "application/vnd.github.v3.diff"}
        resp = requests.get(f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}", headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text

    def post_pr_comment(body: str) -> str:
        """Post a REAL comment on this pull request via the GitHub API."""
        resp = requests.post(
            f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments",
            headers=_headers(),
            json={"body": body},
            timeout=30,
        )
        resp.raise_for_status()
        return f"Posted comment: {resp.json().get('html_url')}"

    return get_pr_diff, post_pr_comment


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
    "You are a PR review assistant. Call get_pr_diff to fetch the real diff for "
    "this pull request, delegate to the ci-runner sub-agent to run the test "
    "suite against the checked-out files, and diagnose what's wrong. Then call "
    "post_pr_comment with a concise, specific comment — reference exact lines/"
    "function names from the diff. Do NOT edit any files yourself."
)


def build_live_pr_review_agent(model: str, test_dir: Path, repo: str, pr_number: int, checkpointer):
    backend = LocalShellBackend(root_dir=test_dir, virtual_mode=True)
    get_pr_diff, post_pr_comment = make_github_tools(repo, pr_number)
    return create_deep_agent(
        model=model,
        backend=backend,
        subagents=[CI_RUNNER_SUBAGENT],
        tools=[get_pr_diff, post_pr_comment],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        interrupt_on={"post_pr_comment": True, "write_file": True, "edit_file": True, "execute": True},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/repo, e.g. viethoang35/deepagents-guide")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument("--test-dir", required=True, help="Local directory with the PR's files checked out")
    parser.add_argument("--interactive", action="store_true", help="Approve the comment manually via stdin")
    args = parser.parse_args()

    test_dir = Path(args.test_dir).resolve()
    agent = build_live_pr_review_agent(MODEL, test_dir, args.repo, args.pr, MemorySaver())

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    user_msg = f"Review pull request #{args.pr} in {args.repo} and post a comment on what's wrong."
    print(f">>> repo = {args.repo}  PR #{args.pr}")
    print(f">>> local test dir = {test_dir}")
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
