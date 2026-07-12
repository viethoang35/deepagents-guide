"""
B3_nemotron_deep_agent.py — Multi-model Deep Agent qua OpenRouter (Bước 3).

Rút gọn từ ví dụ chính thức langchain-ai/deepagents/examples/nvidia_deep_agent,
nhưng thay vì dùng Anthropic (orchestrator) + NVIDIA NIM (researcher) riêng biệt,
cả hai model đều gọi qua OpenRouter -> bạn CHỈ CẦN 1 key (OPENROUTER_API_KEY).

Ý tưởng multi-model (giống NVIDIA):
  - Orchestrator: model mạnh, reasoning tốt (mặc định grok-4.5)
  - Researcher sub-agent: model rẻ/nhanh, làm volume work (mặc định deepseek)

Chạy:
  cp .env.example .env          # điền OPENROUTER_API_KEY + TAVILY_API_KEY
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B3_nemotron_deep_agent.py

Biến môi trường (trong .env):
  OPENROUTER_API_KEY=sk-or-...
  TAVILY_API_KEY=tvly-...                    # search thật (không có -> fallback mock)
  ORCHESTRATOR_MODEL=openrouter:x-ai/grok-4.5        # orchestrator
  RESEARCHER_MODEL=openrouter:deepseek/deepseek-chat-v3  # sub-agent researcher
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from deepagents import create_deep_agent

load_dotenv()

# Orchestrator: model mạnh, planning/synthesis (như frontier model bên NVIDIA)
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "openrouter:x-ai/grok-4.5")
# Researcher: model rẻ/nhanh, làm volume work (như Nemotron Super bên NVIDIA)
RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "openrouter:deepseek/deepseek-chat-v3")

current_date = datetime.now().strftime("%Y-%m-%d")


def web_search(query: str) -> str:
    """Search the web for a query using Tavily (real). Falls back to mock if no key."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return f"[mock search] results for '{query}': NVIDIA, LangChain, arXiv, HuggingFace."
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        res = client.search(query=query, max_results=3, topic="general")
        texts = []
        for r in res.get("results", []):
            texts.append(f"## {r.get('title')}\n{r.get('content')}\nURL: {r.get('url')}")
        return f"Found {len(texts)} result(s) for '{query}':\n\n" + "\n\n".join(texts)
    except Exception as e:
        return f"[search error] {e}"


def main():
    # Sub-agent researcher: chạy context cô lập, chỉ làm web research.
    # Dùng dict spec (đã verify đúng API deepagents hiện tại).
    researcher_sub = {
        "name": "researcher-agent",
        "description": (
            "Delegate research to this agent. Conducts web searches and gathers "
            "information on a topic. Give one focused research topic at a time."
        ),
        "system_prompt": (
            f"You are a web researcher (today is {current_date}). "
            "Return concise bullet-point summaries with sources."
        ),
        "tools": [web_search],
        "model": RESEARCHER_MODEL,
    }

    # Orchestrator agent: lập kế hoạch, ủy quyền cho researcher, tổng hợp kết quả.
    orchestrator = create_deep_agent(
        model=ORCHESTRATOR_MODEL,
        tools=[web_search],
        system_prompt=(
            f"You are a research orchestrator (today is {current_date}). "
            "Break the task into steps, delegate web lookups to the researcher-agent, "
            "then synthesize a concise answer."
        ),
        subagents=[researcher_sub],
    )

    user_msg = "Research 'agent harness security best practices' and summarize in 3 bullet points."
    print(f">>> Orchestrator ({ORCHESTRATOR_MODEL}) + Researcher ({RESEARCHER_MODEL})")
    print(f">>> User: {user_msg}\n")

    result = orchestrator.invoke(
        {"messages": [{"role": "user", "content": user_msg}]}
    )

    messages = result.get("messages", [])
    final = messages[-1]
    print(f">>> Agent: {getattr(final, 'content', final)}\n")

    print("--- Trajectory ---")
    for m in messages:
        t = type(m).__name__
        content = getattr(m, "content", "")
        preview = content[:100].replace("\n", " ") if isinstance(content, str) else str(content)[:100]
        print(f"  [{t}] {preview}")


if __name__ == "__main__":
    main()
