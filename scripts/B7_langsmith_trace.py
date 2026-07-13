"""
B7_langsmith_trace.py — Deep Agent + LangSmith tracing (Step 7, mở rộng trên branch tracing-labs).

Mở rộng từ B3 (multi-model Deep Agent) nhưng THÊM lớp observability production:
gửi TOÀN BỘ trajectory (orchestrator -> researcher sub-agent -> tool -> llm)
lên LangSmith cloud qua env vars (không sửa create_deep_agent).

So với B4 (chỉ log local qua callback handler), B7 gửi trace lên UI LangSmith để:
  - xem cây run phân nhánh (từng sub-agent = 1 node)
  - đo latency / token / cost từng bước
  - gắn feedback / chấm điểm (dùng cho B8 eval)

Cách bật (trong .env, KHÔNG sửa code):
  LANGCHAIN_TRACING_V2=true        # chuẩn LangChain hiện tại
  LANGSMITH_API_KEY=lsv2_...
  LANGSMITH_PROJECT=deepagents-guide

Chạy OFFLINE (fake key) để verify cấu trúc không cần API:
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B7_langsmith_trace.py
Chạy THẬT (cần OPENROUTER_API_KEY + LANGSMITH_API_KEY): trace lên smith.langchain.com.

Liên kết: tương đương Lab06/Lab07 trong dự án langsmith-tracing-labs, nhưng gắn
trực tiếp vào deepagents-guide với convention B-series.
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from deepagents import create_deep_agent

load_dotenv()

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "openrouter:x-ai/grok-4.5")
RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "openrouter:deepseek/deepseek-chat-v3")
current_date = datetime.now().strftime("%Y-%m-%d")


def web_search(query: str) -> str:
    """Search the web (mock khi không có Tavily key — đủ để verify trace)."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return f"[mock search] results for '{query}': NVIDIA, LangChain, arXiv, HuggingFace."
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        res = client.search(query=query, max_results=3, topic="general")
        texts = [f"## {r.get('title')}\n{r.get('content')}\nURL: {r.get('url')}" for r in res.get("results", [])]
        return f"Found {len(texts)} result(s) for '{query}':\n\n" + "\n\n".join(texts)
    except Exception as e:
        return f"[search error] {e}"


def build_agent():
    researcher_sub = {
        "name": "researcher-agent",
        "description": "Delegate research to this agent. Conducts web searches and gathers information on a topic.",
        "system_prompt": f"You are a web researcher (today is {current_date}). Return concise bullet-point summaries with sources.",
        "tools": [web_search],
        "model": RESEARCHER_MODEL,
    }
    return create_deep_agent(
        model=ORCHESTRATOR_MODEL,
        tools=[web_search],
        system_prompt=(
            f"You are a research orchestrator (today is {current_date}). "
            "Break the task into steps, delegate web lookups to the researcher-agent, "
            "then synthesize a concise answer."
        ),
        subagents=[researcher_sub],
    )


def main() -> None:
    print(f">>> B7 LangSmith tracing | Orchestrator={ORCHESTRATOR_MODEL} Researcher={RESEARCHER_MODEL}")
    tracing = os.getenv("LANGCHAIN_TRACING_V2", os.getenv("LANGSMITH_TRACING", "false"))
    key = os.getenv("LANGSMITH_API_KEY", "")
    print(f">>> LangSmith tracing = {tracing} | project = {os.getenv('LANGSMITH_PROJECT', 'deepagents-guide')}")

    agent = build_agent()
    print(f">>> Agent compiled: {type(agent).__name__} (kỳ vọng CompiledStateGraph)")

    is_fake = (not key) or key.startswith("lsv2_xxx") or key.startswith("ls__")
    if is_fake:
        print("\n[OFFLINE] Thiếu LANGSMITH_API_KEY thật -> verify cấu trúc, không gửi trace.")
        print("Để trace THẬT: .env có LANGCHAIN_TRACING_V2=true + LANGSMITH_API_KEY thật + OPENROUTER_API_KEY thật")
        return

    user_msg = "Research 'agent harness security best practices' and summarize in 3 bullet points."
    print(f">>> User: {user_msg}\n")
    result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
    final = result["messages"][-1]
    print(f">>> Agent: {getattr(final, 'content', final)}\n")
    print(">>> Mở https://smith.langchain.com/project/deepagents-guide để xem run tree.")


if __name__ == "__main__":
    main()
