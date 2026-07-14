"""
B7_langsmith_trace.py — Deep Agent + LangSmith tracing (Step 7 gets a real script).

Extends B3 (multi-model Deep Agent) with the production observability layer that
Step 7 previously only described in words: sending the FULL trajectory
(orchestrator -> researcher sub-agent -> tool -> llm) to LangSmith cloud via env
vars alone (no changes to create_deep_agent).

Unlike B4 (local-only logging via a callback handler), B7 ships the trace to the
LangSmith UI so you can:
  - see the branching run tree (each sub-agent = 1 node)
  - measure latency / tokens / cost per step
  - attach feedback / scores (used by B8's eval)

The agent always runs for real as long as OPENROUTER_API_KEY is valid — tracing
to LangSmith is a bonus layered on top via env vars, not a precondition to run
the script at all (this mirrors B3/B4: the agent's own behavior should always be
verifiable even without extra observability tooling turned on).

Enable tracing (in .env, no code changes):
  LANGCHAIN_TRACING_V2=true       # current standard LangChain env var
  LANGSMITH_API_KEY=lsv2_...
  LANGSMITH_PROJECT=deepagents-guide

Run (works with just OPENROUTER_API_KEY; tracing is skipped if no LangSmith key):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B7_langsmith_trace.py

Not yet verified end-to-end against a real LangSmith account (no key on this
machine) — the agent invocation itself is real and live-verified; only the
"trace actually shows up on smith.langchain.com" part is unverified.
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
    """Search the web (mock when no Tavily key is set — enough to exercise the trace)."""
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


def tracing_enabled() -> bool:
    on = os.getenv("LANGCHAIN_TRACING_V2", os.getenv("LANGSMITH_TRACING", "false")).lower() == "true"
    key = os.getenv("LANGSMITH_API_KEY", "").strip()
    return on and key.startswith("lsv2_") and len(key) > 20


def main() -> None:
    print(f">>> B7 LangSmith tracing | Orchestrator={ORCHESTRATOR_MODEL} Researcher={RESEARCHER_MODEL}")
    tracing = tracing_enabled()
    print(f">>> LangSmith tracing = {tracing} | project = {os.getenv('LANGSMITH_PROJECT', 'deepagents-guide')}")

    agent = build_agent()

    user_msg = "Research 'agent harness security best practices' and summarize in 3 bullet points."
    print(f">>> User: {user_msg}\n")
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
    except Exception as e:
        print(f">>> Agent call failed: {type(e).__name__}: {e}")
        print(">>> Check OPENROUTER_API_KEY in .env is valid.")
        return

    final = result["messages"][-1]
    print(f">>> Agent: {getattr(final, 'content', final)}\n")

    if tracing:
        print(f">>> Open https://smith.langchain.com/project/{os.getenv('LANGSMITH_PROJECT', 'deepagents-guide')} to see the run tree.")
    else:
        print(">>> Ran without LangSmith tracing (no LANGSMITH_API_KEY set) — agent output above is real, trace upload was skipped.")


if __name__ == "__main__":
    main()
