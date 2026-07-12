"""
B5_repo_ops_agent.py — Repo-ops Deep Agent trên hardened base (Bước 5).

Orchestrator đọc file THẬT trong `workspace/sample-repo/` (không phải virtual
filesystem giả lập như B2), ủy quyền cho sub-agent "runner-agent" chạy pytest
thật qua shell thật, rồi tự sửa file khi test fail. Mọi hành động có side
effect (sửa file, chạy lệnh shell) đều bị chặn lại chờ người duyệt — dùng lại
đúng 3 lớp hardening từ B4 (checkpointer, human-in-the-loop, observability).

Backend dùng ở đây là `LocalShellBackend`: filesystem + shell THẬT, KHÔNG có
sandbox/cách ly nào (đọc kỹ docstring của nó — "unrestricted local shell
execution"). Layer an toàn duy nhất là:
  1. `virtual_mode=True` + `root_dir=workspace/sample-repo` -> các tool
     filesystem (ls/read_file/write_file/edit_file/glob/grep) không thể đi ra
     ngoài thư mục này (nhưng `execute` KHÔNG bị giới hạn bởi virtual_mode).
  2. `interrupt_on` bắt buộc duyệt tay cho write_file/edit_file/execute.
  3. `runner-agent` sub-agent chỉ được hướng dẫn (qua system_prompt) chạy
     pytest/lint — đây là ràng buộc "soft" (prompt), không phải sandbox
     thật. Muốn cách ly cứng (Docker/VM), xem `vendor-deepagents/libs/partners/`
     (modal, runloop, daytona) và extend `BaseSandbox`.

Chạy (demo tự động duyệt):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B5_repo_ops_agent.py

Chạy có duyệt thủ công từng bước (approve/reject qua stdin):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B5_repo_ops_agent.py --interactive
"""
import argparse
import os
import time
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

load_dotenv()

MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")
REPO_ROOT = Path(__file__).parent.parent
SAMPLE_REPO = REPO_ROOT / "workspace" / "sample-repo"
DB_PATH = REPO_ROOT / "workspace" / "b5_checkpoints.db"


class LocalObservabilityHandler(BaseCallbackHandler):
    """Log tool calls (tên, args, thời gian) và token usage — giống B4."""

    def __init__(self) -> None:
        self._tool_started_at: dict[UUID, float] = {}

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **kwargs: Any) -> None:
        self._tool_started_at[run_id] = time.monotonic()
        print(f"  [tool-call] {serialized.get('name', '?')}({input_str})")

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        started = self._tool_started_at.pop(run_id, None)
        elapsed = f"{(time.monotonic() - started) * 1000:.0f}ms" if started else "?"
        preview = str(output)[:160].replace("\n", " ")
        print(f"  [tool-done] ({elapsed}) -> {preview}")

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        for gen_list in getattr(response, "generations", []):
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None) if msg else None
                if usage:
                    print(f"  [tokens] total={usage.get('total_tokens')}")


def resolve_decisions(action_requests: list[dict], interactive: bool) -> list[dict]:
    decisions = []
    for req in action_requests:
        name, args = req["name"], req.get("args", {})
        if not interactive:
            print(f"  [auto-approve demo] {name}({args})")
            decisions.append({"type": "approve"})
            continue
        print(f"\n  Tool cần duyệt: {name}({args})")
        choice = input("  approve / reject? [a/r]: ").strip().lower()
        decisions.append({"type": "approve" if choice == "a" else "reject"})
    return decisions


RUNNER_SUBAGENT = {
    "name": "runner-agent",
    "description": (
        "Delegate to this agent to run tests or lint in the repo and "
        "report raw output. Give it one command to run at a time."
    ),
    "system_prompt": (
        "You are a test runner. You ONLY run test/lint commands via the "
        "execute tool (e.g. `python -m pytest -q`). You never edit files. "
        "Report the raw output back to the orchestrator."
    ),
}

REPO_OPS_SYSTEM_PROMPT = (
    "You are a repo-ops assistant working inside a small Python repo. "
    "Delegate running tests to the runner-agent sub-agent. If a test "
    "fails, read the relevant source file, diagnose the bug, apply a "
    "minimal fix with edit_file, then delegate to runner-agent again "
    "to confirm the fix. Keep changes minimal and explain your diagnosis."
)


def build_repo_ops_agent(model: str, root_dir: Path, checkpointer):
    """Build the repo-ops agent. Shared with `evals/test_repo_ops_eval.py`.

    Filesystem + shell THẬT, giới hạn root_dir. `execute()` chạy với
    cwd = root_dir (xem docstring `LocalShellBackend.execute`).
    """
    backend = LocalShellBackend(root_dir=root_dir, virtual_mode=True)
    return create_deep_agent(
        model=model,
        backend=backend,
        subagents=[RUNNER_SUBAGENT],
        system_prompt=REPO_OPS_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        interrupt_on={"write_file": True, "edit_file": True, "execute": True},
    )


def run_to_completion(agent, config: dict, user_msg: str, *, interactive: bool = False) -> str:
    """Invoke `agent` and auto-resume through every HITL interrupt until done.

    Shared with `evals/test_repo_ops_eval.py`. Returns the final message content.
    """
    agent.invoke({"messages": [{"role": "user", "content": user_msg}]}, config=config)

    state = agent.get_state(config)
    while state.interrupts:
        interrupt_value = state.interrupts[0].value
        decisions = resolve_decisions(interrupt_value["action_requests"], interactive)
        agent.invoke(Command(resume={"decisions": decisions}), config=config)
        state = agent.get_state(config)

    messages = state.values.get("messages", [])
    final = messages[-1] if messages else None
    return getattr(final, "content", str(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Duyệt thủ công qua stdin")
    args = parser.parse_args()

    DB_PATH.parent.mkdir(exist_ok=True)

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        agent = build_repo_ops_agent(MODEL, SAMPLE_REPO, checkpointer)

        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [LocalObservabilityHandler()],
        }
        print(f">>> thread_id = {thread_id}")
        print(f">>> repo root  = {SAMPLE_REPO}")

        user_msg = (
            "Run the test suite. If anything fails, diagnose and fix the bug "
            "with a minimal edit, then rerun the tests to confirm it passes."
        )
        print(f">>> User: {user_msg}\n")
        final = run_to_completion(agent, config, user_msg, interactive=args.interactive)
        print(f"\n>>> Agent: {final}")


if __name__ == "__main__":
    main()
