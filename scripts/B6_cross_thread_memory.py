"""
B6_cross_thread_memory.py — Cross-thread memory (Bước 6), khác với checkpointer.

Phân biệt 2 loại "nhớ" hay bị nhầm với nhau:
  - Checkpointer (B4): nhớ trong CÙNG 1 thread — lịch sử message của 1
    conversation, resume qua thread_id.
  - `memory=[...]` (bài này): nhớ XUYÊN thread — 1 file AGENTS.md THẬT trên
    đĩa, được nạp vào system prompt ở đầu MỌI conversation (mọi thread_id),
    và agent có thể tự cập nhật file này qua edit_file khi học được điều gì
    mới. Đây là cơ chế "long-term memory" của Deep Agents.

Demo: chạy 2 "conversation" hoàn toàn tách biệt (2 thread_id khác nhau, coi
như 2 lần mở app ở 2 ngày khác nhau):
  1. Conversation A: user báo 1 sở thích (trả lời dạng bullet point). Agent
     ghi sở thích này vào workspace/agent_memory/AGENTS.md qua edit_file
     (cần duyệt — vẫn dùng HITL như B4/B5).
  2. Conversation B: thread_id MỚI, không liên quan gì tới A. User hỏi 1 câu
     khác. Agent vẫn trả lời theo đúng sở thích đã lưu ở bước 1 — chứng minh
     memory này sống ngoài phạm vi 1 thread.

Chạy:
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B6_cross_thread_memory.py

Chạy lại (không xoá AGENTS.md): script sẽ append thêm — xoá
workspace/agent_memory/AGENTS.md nếu muốn xem lại từ đầu.
"""
import os
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

load_dotenv()

MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")
MEMORY_DIR = __import__("pathlib").Path(__file__).parent.parent / "workspace" / "agent_memory"
MEMORY_FILE = "/AGENTS.md"  # path ảo bên trong MEMORY_DIR (virtual_mode=True)


def run_turn(agent, user_msg: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    agent.invoke({"messages": [{"role": "user", "content": user_msg}]}, config=config)

    state = agent.get_state(config)
    while state.interrupts:
        action_requests = state.interrupts[0].value["action_requests"]
        for req in action_requests:
            print(f"  [auto-approve demo] {req['name']}({req.get('args', {})})")
        decisions = [{"type": "approve"} for _ in action_requests]
        agent.invoke(Command(resume={"decisions": decisions}), config=config)
        state = agent.get_state(config)

    messages = state.values.get("messages", [])
    final = messages[-1] if messages else None
    return getattr(final, "content", str(final))


def main() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    backend = FilesystemBackend(root_dir=MEMORY_DIR, virtual_mode=True)

    agent = create_deep_agent(
        model=MODEL,
        backend=backend,
        memory=[MEMORY_FILE],
        system_prompt=(
            "You are a helpful assistant. When the user states a lasting "
            "preference about how they want you to respond, save it to "
            f"{MEMORY_FILE} via edit_file (create it with write_file if it "
            "doesn't exist yet) so you remember it in future conversations."
        ),
        checkpointer=MemorySaver(),
        interrupt_on={"write_file": True, "edit_file": True},
    )

    print(">>> Conversation A (thread mới) — dạy agent 1 sở thích\n")
    thread_a = str(uuid.uuid4())
    msg_a = (
        "From now on, always answer in bullet points, no paragraphs. "
        "Please remember this for future conversations."
    )
    print(f"  User: {msg_a}")
    reply_a = run_turn(agent, msg_a, thread_a)
    print(f"  Agent: {reply_a}\n")

    print(">>> Conversation B (thread MỚI, không liên quan tới A)\n")
    thread_b = str(uuid.uuid4())
    msg_b = "What is a deep agent, in short?"
    print(f"  User: {msg_b}")
    reply_b = run_turn(agent, msg_b, thread_b)
    print(f"  Agent: {reply_b}\n")

    real_path = MEMORY_DIR / MEMORY_FILE.lstrip("/")
    print(f">>> Nội dung {MEMORY_FILE} hiện tại (file thật tại {real_path}):")
    print("  " + real_path.read_text().replace("\n", "\n  "))
    print(
        "\n>>> Nếu conversation B trả lời theo dạng bullet point (không phải "
        "đoạn văn) dù thread_id hoàn toàn mới -> memory đã hoạt động xuyên thread."
    )


if __name__ == "__main__":
    main()
