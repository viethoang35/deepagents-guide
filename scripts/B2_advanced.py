"""
B2_advanced.py — Deep Agents nâng cao: filesystem, sub-agent, streaming.

Minh họa các tính năng cốt lõi của Deep Agents:
  - Sub-agent (ủy quyền, context cô lập)
  - Virtual filesystem (đọc/ghi file)
  - Streaming (in từng bước)

Chạy:  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B2_advanced.py
(Model lấy từ DEEPAGENTS_MODEL trong .env — ví dụ openrouter:openai/gpt-4o-mini.
Cả agent chính và sub-agent researcher đều dùng chung biến MODEL này.)
"""
import os
from dotenv import load_dotenv
from deepagents import create_deep_agent

load_dotenv()

MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")


def web_search(query: str) -> str:
    """Search the web for a query (mock)."""
    return f"[mock search result] Top hits for '{query}': NVIDIA, LangChain, arXiv."


def main():
    # Sub-agent: researcher chạy context cô lập, chỉ làm web research.
    # Phiên bản deepagents hiện tại nhận subagents dưới dạng dict spec
    # (giống ví dụ nvidia_deep_agent), KHÔNG phải object SubAgent(agent=...).
    researcher_sub = {
        "name": "researcher-agent",
        "description": "Delegate web research to this agent for any topic.",
        "system_prompt": "You are a web researcher. Return concise bullet summaries.",
        "model": MODEL,
        "tools": [web_search],
    }

    # Agent chính: orchestrator, có filesystem + ủy quyền cho researcher
    # Lưu ý: API thực tế dùng 'subagents' (không gạch dưới)
    agent = create_deep_agent(
        model=MODEL,
        tools=[],
        subagents=[researcher_sub],
        system_prompt=(
            "You are a research assistant. Delegate web lookups to the "
            "researcher sub-agent, then write a short summary to a file."
        ),
        # Deep Agents tự cấp filesystem ảo; bạn có thể cấu hình quyền rõ ràng:
        # filesystem={"root": "./workspace", "writable_paths": ["./workspace/output"]}
    )

    user_msg = "Research 'agent harness security' and save a 3-line summary."
    print(f">>> User: {user_msg}\n")

    # Streaming: in từng chunk. Lưu ý: agent.stream() trả dict có key là tên
    # middleware (vd 'model', 'TodoListMiddleware.after_model'), nội dung nằm
    # trong chunk[<key>]['messages'] (list). Ta duyệt mọi value để tìm message.
    seen = set()
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_msg}]}
    ):
        if not isinstance(chunk, dict):
            continue
        for val in chunk.values():
            if isinstance(val, dict) and isinstance(val.get("messages"), list):
                for m in val["messages"]:
                    t = type(m).__name__
                    c = getattr(m, "content", "")
                    text = c if isinstance(c, str) else str(c)
                    # chỉ in các đoạn có nội dung mới, tránh lặp
                    key = (t, text[:60])
                    if text.strip() and key not in seen:
                        seen.add(key)
                        print(f"[step][{t}] {text[:120].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
