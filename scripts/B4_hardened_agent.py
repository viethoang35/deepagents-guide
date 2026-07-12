"""
B4_hardened_agent.py — Deep Agents "production-safer" harness (Bước 4).

Lấy lại orchestrator đơn giản (giống B2) nhưng thêm 3 lớp thường bị bỏ qua
khi mới học Deep Agents:

  1. Checkpointer bền (SqliteSaver): state của conversation được lưu vào file
     .db, agent có thể dừng giữa chừng và resume lại đúng thread sau khi
     process khác được khởi động lại (không mất context).
  2. Human-in-the-loop (interrupt_on): các tool "nguy hiểm" (send_email,
     delete_file) bị chặn lại để chờ người duyệt trước khi thực thi thật,
     thay vì để model tự do gọi.
  3. Observability nhẹ (callback handler cục bộ): log mọi tool call (tên,
     args, thời gian chạy) + token usage mỗi lần gọi model — không cần tài
     khoản ngoài. Nếu bạn có LangSmith, set 2 biến env dưới là đủ, không cần
     sửa code:
       LANGSMITH_TRACING=true
       LANGSMITH_API_KEY=ls__...

Chạy (demo tự động duyệt, không cần nhập tay):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B4_hardened_agent.py

Chạy có duyệt thủ công (approve/reject từng bước qua stdin):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B4_hardened_agent.py --interactive

Resume 1 thread cũ (dùng lại thread_id đã in ra lần trước):
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
# send_email và delete_file là tool "nhạy cảm": có tác dụng phụ thật (side
# effect) nếu chạy production, nên được đưa vào interrupt_on ở dưới.

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (mock)."""
    return f"[MOCK] Email sent to {to} — subject: '{subject}'"


def delete_file(path: str) -> str:
    """Delete a file (mock)."""
    return f"[MOCK] Deleted file at {path}"


def get_weather(city: str) -> str:
    """Get the current weather for a given city (mock, an toàn — không cần duyệt)."""
    return f"It's always sunny in {city}!"


# --- Observability: callback handler cục bộ, không cần LangSmith -----------

class LocalObservabilityHandler(BaseCallbackHandler):
    """Log tool calls (tên, args, thời gian) và token usage mỗi lần gọi model."""

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


# --- HITL: quyết định approve/reject cho từng action_request ---------------

def resolve_decisions(action_requests: list[dict], interactive: bool) -> list[dict]:
    decisions = []
    for req in action_requests:
        name = req["name"]
        args = req.get("args", {})
        if not interactive:
            print(f"  [auto-approve demo] {name}({args})")
            decisions.append({"type": "approve"})
            continue
        print(f"\n  Tool cần duyệt: {name}({args})")
        choice = input("  approve / reject? [a/r]: ").strip().lower()
        decisions.append({"type": "approve" if choice == "a" else "reject"})
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Duyệt thủ công qua stdin")
    parser.add_argument("--thread-id", default=None, help="Resume 1 thread cũ (checkpoint đã lưu)")
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
        print(f">>> thread_id = {thread_id}  (lưu lại để --thread-id resume sau)")
        print(f">>> checkpoint db = {DB_PATH}")

        if args.thread_id:
            # Resume thread cũ: không gửi message mới, chỉ tiếp tục state đã lưu.
            state = agent.get_state(config)
            print(f">>> Resumed. {len(state.values.get('messages', []))} message(s) trong lịch sử.")
            if not state.interrupts:
                print(">>> Thread này không có interrupt đang chờ — không có gì để resume.")
                return
        else:
            user_msg = (
                "Check the weather in Hanoi, then email a 1-line summary to "
                "boss@example.com with subject 'Weather update', then delete "
                "the file draft.txt."
            )
            print(f">>> User: {user_msg}\n")
            agent.invoke({"messages": [{"role": "user", "content": user_msg}]}, config=config)

        # Vòng lặp: mỗi lần graph dừng lại vì interrupt, hỏi quyết định rồi resume.
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
