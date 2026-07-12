"""
B4_hardened_agent.py — A "production-safer" Deep Agents harness (Step 4).

Reuses the simple orchestrator (like B2) but adds 3 layers that are commonly
skipped when first learning Deep Agents:

  1. Durable checkpointer (SqliteSaver): conversation state is saved to a
     .db file, so the agent can stop mid-task and resume the correct thread
     after the process is restarted (no lost context).
  2. Human-in-the-loop (interrupt_on): "dangerous" tools (send_email,
     delete_file) pause for a human decision before actually running,
     instead of letting the model call them freely.
  3. Lightweight observability (a local callback handler): logs every tool
     call (name, args, elapsed time) + token usage on every model call — no
     external account needed. If you do have LangSmith, just set these 2 env
     vars, no code changes:
       LANGSMITH_TRACING=true
       LANGSMITH_API_KEY=ls__...

Run (auto-approve demo, no manual input needed):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B4_hardened_agent.py

Run with manual approval (approve/reject each step via stdin):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B4_hardened_agent.py --interactive

Resume an old thread (reuse the thread_id printed on a previous run):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B4_hardened_agent.py --thread-id <id>
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

load_dotenv()

MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")
DB_PATH = Path(__file__).parent.parent / "workspace" / "b4_checkpoints.db"


# --- Tools -----------------------------------------------------------------
# send_email and delete_file are "sensitive" tools: they have real side
# effects in production, so they're placed in interrupt_on below.

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (mock)."""
    return f"[MOCK] Email sent to {to} — subject: '{subject}'"


def delete_file(path: str) -> str:
    """Delete a file (mock)."""
    return f"[MOCK] Deleted file at {path}"


def get_weather(city: str) -> str:
    """Get the current weather for a given city (mock, safe — no approval needed)."""
    return f"It's always sunny in {city}!"


# --- Observability: local callback handler, no LangSmith needed ------------

class LocalObservabilityHandler(BaseCallbackHandler):
    """Log tool calls (name, args, elapsed time) and token usage on every model call."""

    def __init__(self) -> None:
        self._tool_started_at: dict[UUID, float] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_started_at[run_id] = time.monotonic()
        name = serialized.get("name", "?")
        print(f"  [tool-call] {name}({input_str})")

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        started = self._tool_started_at.pop(run_id, None)
        elapsed = f"{(time.monotonic() - started) * 1000:.0f}ms" if started else "?"
        preview = str(output)[:100].replace("\n", " ")
        print(f"  [tool-done] ({elapsed}) -> {preview}")

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        for gen_list in getattr(response, "generations", []):
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None) if msg else None
                if usage:
                    print(
                        f"  [tokens] input={usage.get('input_tokens')} "
                        f"output={usage.get('output_tokens')} "
                        f"total={usage.get('total_tokens')}"
                    )


# --- HITL: decide approve/reject for each action_request -------------------

def resolve_decisions(action_requests: list[dict], interactive: bool) -> list[dict]:
    decisions = []
    for req in action_requests:
        name = req["name"]
        args = req.get("args", {})
        if not interactive:
            print(f"  [auto-approve demo] {name}({args})")
            decisions.append({"type": "approve"})
            continue
        print(f"\n  Tool needs approval: {name}({args})")
        choice = input("  approve / reject? [a/r]: ").strip().lower()
        decisions.append({"type": "approve" if choice == "a" else "reject"})
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Approve manually via stdin")
    parser.add_argument("--thread-id", default=None, help="Resume an old thread (saved checkpoint)")
    args = parser.parse_args()

    DB_PATH.parent.mkdir(exist_ok=True)

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        agent = create_deep_agent(
            model=MODEL,
            tools=[get_weather, send_email, delete_file],
            system_prompt=(
                "You are an assistant that can check weather, send emails, and "
                "delete files. Use tools directly to accomplish the user's request."
            ),
            checkpointer=checkpointer,
            interrupt_on={
                "send_email": True,
                "delete_file": {"allowed_decisions": ["approve", "reject"]},
            },
        )

        thread_id = args.thread_id or str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [LocalObservabilityHandler()],
        }
        print(f">>> thread_id = {thread_id}  (save this to resume later with --thread-id)")
        print(f">>> checkpoint db = {DB_PATH}")

        if args.thread_id:
            # Resume an old thread: don't send a new message, just continue the saved state.
            state = agent.get_state(config)
            print(f">>> Resumed. {len(state.values.get('messages', []))} message(s) in history.")
            if not state.interrupts:
                print(">>> This thread has no pending interrupt — nothing to resume.")
                return
        else:
            user_msg = (
                "Check the weather in Hanoi, then email a 1-line summary to "
                "boss@example.com with subject 'Weather update', then delete "
                "the file draft.txt."
            )
            print(f">>> User: {user_msg}\n")
            agent.invoke({"messages": [{"role": "user", "content": user_msg}]}, config=config)

        # Loop: every time the graph pauses on an interrupt, ask for a decision then resume.
        state = agent.get_state(config)
        while state.interrupts:
            interrupt_value = state.interrupts[0].value
            action_requests = interrupt_value["action_requests"]
            decisions = resolve_decisions(action_requests, args.interactive)
            result = agent.invoke(Command(resume={"decisions": decisions}), config=config)
            state = agent.get_state(config)

        messages = state.values.get("messages", [])
        final = messages[-1] if messages else None
        print(f"\n>>> Agent: {getattr(final, 'content', final)}")


if __name__ == "__main__":
    main()
