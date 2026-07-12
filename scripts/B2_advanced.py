"""
B2_advanced.py — Advanced Deep Agents: filesystem, sub-agent, streaming.

Demonstrates core Deep Agents features:
  - Sub-agent (delegation, isolated context)
  - Virtual filesystem (read/write files)
  - Streaming (print each step)

Run:  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B2_advanced.py
(Model comes from DEEPAGENTS_MODEL in .env — e.g. openrouter:openai/gpt-4o-mini.
Both the main agent and the researcher sub-agent share this same MODEL variable.)
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
    # Sub-agent: researcher runs in an isolated context, only does web research.
    # The current deepagents version accepts subagents as a dict spec
    # (like the nvidia_deep_agent example), NOT a SubAgent(agent=...) object.
    researcher_sub = {
        "name": "researcher-agent",
        "description": "Delegate web research to this agent for any topic.",
        "system_prompt": "You are a web researcher. Return concise bullet summaries.",
        "model": MODEL,
        "tools": [web_search],
    }

    # Main agent: orchestrator, has filesystem access + delegates to researcher
    # Note: the real API uses 'subagents' (no underscore)
    agent = create_deep_agent(
        model=MODEL,
        tools=[],
        subagents=[researcher_sub],
        system_prompt=(
            "You are a research assistant. Delegate web lookups to the "
            "researcher sub-agent, then write a short summary to a file."
        ),
        # Deep Agents provides a virtual filesystem automatically; you can configure
        # explicit permissions: filesystem={"root": "./workspace", "writable_paths": ["./workspace/output"]}
    )

    user_msg = "Research 'agent harness security' and save a 3-line summary."
    print(f">>> User: {user_msg}\n")

    # Streaming: print each chunk. Note: agent.stream() returns a dict keyed by
    # middleware name (e.g. 'model', 'TodoListMiddleware.after_model'), with the
    # content living in chunk[<key>]['messages'] (a list). We scan every value to find messages.
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
                    # only print new content, avoid repeats
                    key = (t, text[:60])
                    if text.strip() and key not in seen:
                        seen.add(key)
                        print(f"[step][{t}] {text[:120].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
