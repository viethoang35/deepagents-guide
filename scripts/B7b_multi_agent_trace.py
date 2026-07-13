"""
B7b_multi_agent_trace.py — Multi-Agent tracing tự xây (LangGraph) + observability (Step 7b).

Khác B7 (Deep Agent đóng gói sẵn), script này TỰ XÂY multi-agent bằng StateGraph để
minh họa RÕ cơ chế handoff: orchestrator ủy quyền cho researcher sub-agent.
Đây là khái niệm cốt lõi trong wiki Deep-Agents: "sub-agents cô lập context".

Trace tree kỳ vọng (LangSmith/Phoenix):
  Run(graph) -> Run(node: orchestrator) -> Run(llm)
             -> Run(node: researcher)    -> Run(llm) -> Run(tool: web_search)

Chạy OFFLINE (fake model) để verify cấu trúc không cần API:
  OPENROUTER_API_KEY=sk-or-fake env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B7b_multi_agent_trace.py
Chạy THẬT (OpenRouter + LANGCHAIN_TRACING_V2=true): trace lên LangSmith/Phoenix.
"""
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()
MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")


@tool
def web_search(query: str) -> str:
    """Search the web (mock)."""
    return f"[web] Kết quả về '{query}': LangSmith trace mỗi bước của agent."


def get_model():
    key = os.getenv("OPENROUTER_API_KEY", "")
    is_fake = (not key) or key.startswith("sk-or-fake") or key.startswith("sk-or-xxx")
    if is_fake:
        return GenericFakeChatModel(messages=iter([
            AIMessage(content="Tôi sẽ ủy quyền cho researcher tìm kiếm."),
            AIMessage(content="Kết quả: LangSmith giúp trace từng bước agent."),
        ]))
    from langchain.chat_models import init_chat_model
    return init_chat_model(MODEL)


def orchestrator(state: MessagesState):
    model = get_model()
    resp = model.invoke([
        SystemMessage(content="Bạn là orchestrator. Nếu cần thông tin, ủy quyền cho researcher."),
        *state["messages"],
    ])
    return {"messages": [resp]}


def researcher(state: MessagesState):
    model = get_model()
    supports_tools = hasattr(model, "bind_tools") and not isinstance(model, GenericFakeChatModel)
    if supports_tools:
        decision = model.bind_tools([web_search]).invoke([
            SystemMessage(content="Bạn là researcher. Dùng web_search rồi tóm tắt."),
            HumanMessage(content="Tracing AI agent là gì?"),
        ])
    else:
        decision = model.invoke([
            SystemMessage(content="Bạn là researcher. Tóm tắt về tracing AI agent."),
            HumanMessage(content="Tracing AI agent là gì?"),
        ])
    tool_out = web_search.invoke("AI agent tracing")
    return {"messages": [decision, ToolMessage(content=tool_out, tool_call_id="demo")]}


def build_graph():
    g = StateGraph(MessagesState)
    g.add_node("orchestrator", orchestrator)
    g.add_node("researcher", researcher)
    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "researcher")   # handoff (ủy quyền)
    g.add_edge("researcher", END)              # researcher trả kết quả -> kết thúc
    return g.compile()


def main() -> None:
    print(f">>> B7b Multi-Agent tracing | model={MODEL}")
    app = build_graph()
    print(f">>> Graph compiled: {type(app).__name__}")

    key = os.getenv("OPENROUTER_API_KEY", "")
    is_fake = (not key) or key.startswith("sk-or-fake") or key.startswith("sk-or-xxx")
    if is_fake:
        print("\n[OFFLINE] Fake model -> verify trace tree cấu trúc (không gọi API).")
    else:
        print(f"\n[THẬT] Trace gửi lên project '{os.getenv('LANGSMITH_PROJECT', 'deepagents-guide')}'.")

    result = app.invoke({"messages": [HumanMessage(content="Tracing AI agent là gì?")]})
    print("\n=== Kết quả (messages) ===")
    for m in result["messages"]:
        t = type(m).__name__
        c = getattr(m, "content", "")
        c = c if isinstance(c, str) else str(c)
        print(f"[{t}] {c[:90]}")


if __name__ == "__main__":
    main()
