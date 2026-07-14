"""
B7b_multi_agent_trace.py — Hand-built multi-agent tracing with LangGraph (Step 7b).

Unlike B7 (Deep Agents' built-in packaging), this script builds a multi-agent
graph by hand with StateGraph to show the handoff mechanism explicitly:
orchestrator delegates to a researcher sub-agent by passing the graph state
forward. This is the core concept behind Deep Agents' "sub-agents get an
isolated context" behavior, just without the framework doing it for you.

Expected trace tree (LangSmith/Phoenix):
  Run(graph) -> Run(node: orchestrator) -> Run(llm)
             -> Run(node: researcher)    -> Run(llm) -> Run(tool: web_search)

Run (falls back to a scripted fake model if OPENROUTER_API_KEY isn't set or is
a known placeholder; a real-but-invalid key still hits the real API and reports
a clean error instead of crashing):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B7b_multi_agent_trace.py
Run for real (with OpenRouter + LANGCHAIN_TRACING_V2=true): traces to LangSmith/Phoenix.
"""
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()
MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")

FAKE_KEY_PREFIXES = ("sk-or-fake", "sk-or-xxx")


@tool
def web_search(query: str) -> str:
    """Search the web (mock)."""
    return f"[web] Results for '{query}': LangSmith traces every step of the agent."


def _using_placeholder_key() -> bool:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    return key.startswith(FAKE_KEY_PREFIXES) or key == ""


def get_model():
    if _using_placeholder_key():
        return GenericFakeChatModel(messages=iter([
            AIMessage(content="I'll delegate to the researcher to look this up."),
            AIMessage(content="Result: LangSmith helps trace every step of an agent."),
        ]))
    from langchain.chat_models import init_chat_model
    return init_chat_model(MODEL)


def orchestrator(state: MessagesState):
    model = get_model()
    resp = model.invoke([
        SystemMessage(content="You are the orchestrator. Delegate to the researcher if information is needed."),
        *state["messages"],
    ])
    return {"messages": [resp]}


def researcher(state: MessagesState):
    """Pick up where the orchestrator left off — the actual handoff, not a fresh question."""
    model = get_model()
    supports_tools = hasattr(model, "bind_tools") and not isinstance(model, GenericFakeChatModel)
    if supports_tools:
        decision = model.bind_tools([web_search]).invoke([
            SystemMessage(content="You are the researcher. Use web_search, then summarize."),
            *state["messages"],
        ])
    else:
        decision = model.invoke([
            SystemMessage(content="You are the researcher. Summarize what the orchestrator asked about."),
            *state["messages"],
        ])
    tool_out = web_search.invoke(state["messages"][-1].content if state["messages"] else "AI agent tracing")
    return {"messages": [decision, ToolMessage(content=tool_out, tool_call_id="demo")]}


def build_graph():
    g = StateGraph(MessagesState)
    g.add_node("orchestrator", orchestrator)
    g.add_node("researcher", researcher)
    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "researcher")   # handoff (delegation)
    g.add_edge("researcher", END)              # researcher returns the result -> done
    return g.compile()


def main() -> None:
    print(f">>> B7b Multi-Agent tracing | model={MODEL}")
    app = build_graph()
    print(f">>> Graph compiled: {type(app).__name__}")

    fake = _using_placeholder_key()
    if fake:
        print("\n[OFFLINE] No OPENROUTER_API_KEY (or a placeholder) -> using a scripted fake model, no API call made.")
    else:
        print(f"\n[LIVE] Tracing to project '{os.getenv('LANGSMITH_PROJECT', 'deepagents-guide')}' if LangSmith is enabled.")

    try:
        result = app.invoke({"messages": [HumanMessage(content="What is AI agent tracing?")]})
    except Exception as e:
        print(f"\n>>> Graph run failed: {type(e).__name__}: {e}")
        print(">>> Check OPENROUTER_API_KEY in .env is valid, or run without it to use the fake model.")
        return

    print("\n=== Result (messages) ===")
    for m in result["messages"]:
        t = type(m).__name__
        c = getattr(m, "content", "")
        c = c if isinstance(c, str) else str(c)
        print(f"[{t}] {c[:90]}")


if __name__ == "__main__":
    main()
