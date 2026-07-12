"""
B2_quickstart.py — Deep Agents cơ bản nhất + 1 tool tự viết.

Cách chạy (QUAN TRỌNG: dùng env -u PYTHONPATH do môi trường Hermes xuất PYTHONPATH):
  1. cp .env.example .env   # điền OPENROUTER_API_KEY (hoặc đổi DEEPAGENTS_MODEL)
  2. env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B2_quickstart.py

Model mặc định lấy từ DEEPAGENTS_MODEL trong .env (ví dụ openrouter:openai/gpt-4o-mini).
Đổi model bằng cách sửa .env hoặc biến MODEL ở dưới.

Script minh họa:
  - create_deep_agent với 1 tool tự viết (get_weather)
  - agent.invoke xử lý 1 tin nhắn user
  - in ra câu trả lời cuối cùng + trajectory
"""
import os
from dotenv import load_dotenv
from deepagents import create_deep_agent

load_dotenv()  # đọc .env nếu có

# Đổi model tại đây tuỳ key bạn có:
#   "openai:gpt-5.5"
#   "anthropic:claude-sonnet-4-6"
#   "google_genai:gemini-3.5-flash"
#   "ollama:llama3.1"   (local, cần Ollama chạy sẵn)
#   "openrouter:deepseek/deepseek-chat-v3"     (qua OpenRouter, 1 key gọi nhiều model)
#   "openrouter:x-ai/grok-4.5"
#   "openrouter:openai/gpt-4o-mini"
MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")


def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # Đây là tool giả lập — thay bằng API thật (open-meteo, v.v.) khi cần
    return f"It's always sunny in {city}!"


def main():
    agent = create_deep_agent(
        model=MODEL,
        tools=[get_weather],
        system_prompt="You are a helpful assistant that can check the weather.",
    )

    user_msg = "What is the weather in San Francisco?"
    print(f">>> User: {user_msg}\n")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]}
    )

    # result là dict chứa key "messages" (danh sách các bước)
    messages = result.get("messages", [])
    final = messages[-1]
    print(f">>> Agent: {getattr(final, 'content', final)}\n")

    # In tóm tắt các bước (để thấy agent gọi tool như thế nào)
    print("--- Trajectory (các bước agent đã làm) ---")
    for m in messages:
        t = type(m).__name__
        content = getattr(m, "content", "")
        if isinstance(content, str):
            preview = content[:120].replace("\n", " ")
        else:
            preview = str(content)[:120]
        print(f"  [{t}] {preview}")


if __name__ == "__main__":
    main()
