"""
B2_quickstart.py — The most basic Deep Agent + 1 custom tool.

How to run (IMPORTANT: use env -u PYTHONPATH because the Hermes environment exports PYTHONPATH):
  1. cp .env.example .env   # fill in OPENROUTER_API_KEY (or change DEEPAGENTS_MODEL)
  2. env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B2_quickstart.py

Default model comes from DEEPAGENTS_MODEL in .env (e.g. openrouter:openai/gpt-4o-mini).
Change the model by editing .env or the MODEL variable below.

This script demonstrates:
  - create_deep_agent with 1 custom tool (get_weather)
  - agent.invoke handling 1 user message
  - printing the final answer + trajectory
"""
import os
from dotenv import load_dotenv
from deepagents import create_deep_agent

load_dotenv()  # read .env if present

# Change the model here depending on which key you have:
#   "openai:gpt-5.5"
#   "anthropic:claude-sonnet-4-6"
#   "google_genai:gemini-3.5-flash"
#   "ollama:llama3.1"   (local, requires Ollama already running)
#   "openrouter:deepseek/deepseek-chat-v3"     (via OpenRouter, 1 key calls many models)
#   "openrouter:x-ai/grok-4.5"
#   "openrouter:openai/gpt-4o-mini"
MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")


def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # This is a mock tool — swap in a real API (open-meteo, etc.) when needed
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

    # result is a dict containing the "messages" key (a list of steps)
    messages = result.get("messages", [])
    final = messages[-1]
    print(f">>> Agent: {getattr(final, 'content', final)}\n")

    # Print a summary of each step (to see how the agent called tools)
    print("--- Trajectory (the steps the agent took) ---")
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
