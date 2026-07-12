"""
B6_cross_thread_memory.py — Cross-thread memory (Step 6), different from the checkpointer.

Distinguishes 2 kinds of "memory" that are easy to confuse:
  - Checkpointer (B4): remembers within the SAME thread — the message
    history of 1 conversation, resumed via thread_id.
  - `memory=[...]` (this one): remembers ACROSS threads — a REAL AGENTS.md
    file on disk, loaded into the system prompt at the start of EVERY
    conversation (every thread_id), which the agent can update itself via
    edit_file when it learns something new. This is Deep Agents' "long-term
    memory" mechanism.

Demo: runs 2 completely separate "conversations" (2 different thread_ids, as
if opening the app on 2 different days):
  1. Conversation A: the user states a preference (answer in bullet points).
     The agent writes this preference to workspace/agent_memory/AGENTS.md
     via edit_file (needs approval — still using HITL like B4/B5).
  2. Conversation B: a NEW thread_id, unrelated to A. The user asks a
     different question. The agent still answers according to the
     preference saved in step 1 — proving this memory lives outside the
     scope of a single thread.

Run:
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B6_cross_thread_memory.py

Running it again (without deleting AGENTS.md): the script will append more —
delete workspace/agent_memory/AGENTS.md if you want to see it from scratch.
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
MEMORY_FILE = "/AGENTS.md"  # virtual path inside MEMORY_DIR (virtual_mode=True)


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

    print(">>> Conversation A (new thread) — teaching the agent a preference\n")
    thread_a = str(uuid.uuid4())
    msg_a = (
        "From now on, always answer in bullet points, no paragraphs. "
        "Please remember this for future conversations."
    )
    print(f"  User: {msg_a}")
    reply_a = run_turn(agent, msg_a, thread_a)
    print(f"  Agent: {reply_a}\n")

    print(">>> Conversation B (a NEW thread, unrelated to A)\n")
    thread_b = str(uuid.uuid4())
    msg_b = "What is a deep agent, in short?"
    print(f"  User: {msg_b}")
    reply_b = run_turn(agent, msg_b, thread_b)
    print(f"  Agent: {reply_b}\n")

    real_path = MEMORY_DIR / MEMORY_FILE.lstrip("/")
    print(f">>> Current content of {MEMORY_FILE} (a real file at {real_path}):")
    print("  " + real_path.read_text().replace("\n", "\n  "))
    print(
        "\n>>> If conversation B answered in bullet points (not a paragraph) despite "
        "using a brand-new thread_id -> memory worked across threads."
    )


if __name__ == "__main__":
    main()
