"""
B10_support_triage_agent.py — Support-ticket triage agent (real-world use case, Part 2).

Fills the one gap the harness steps (B1-B8) never touched: retrieval. The
orchestrator answers support tickets grounded in a small local knowledge base
(`workspace/support-docs/`, the same troubleshooting entries from this
project's own README), and escalates to a human when the docs don't cover
the question — the escalation is a mocked, HITL-gated tool, same pattern as
B4/B9.

`search_docs` here is a plain word-overlap ranking over local markdown files
— NOT a real embedding-based vector store. That's a deliberate simplification:
it needs no embeddings API key and no new dependency, while still teaching the
actual retrieval-augmented pattern (search a knowledge base -> ground the
answer in what was retrieved -> escalate if nothing relevant was found). Swap
in a real vector store (e.g. Chroma + OpenAI/Voyage embeddings) for production use.

Run:
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B10_support_triage_agent.py

Run with manual approval on escalation:
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B10_support_triage_agent.py --interactive
"""
import argparse
import os
import re
import uuid
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deepagents import create_deep_agent
from scripts.B5_repo_ops_agent import resolve_decisions

load_dotenv()

MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")
DOCS_DIR = Path(__file__).parent.parent / "workspace" / "support-docs"


def _tokenize(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9']+", text.lower()))


def search_docs(query: str) -> str:
    """Search the local support docs and return the best-matching entries.

    Ranks each doc by word overlap with the query (simple lexical retrieval,
    not embeddings — see the module docstring for why).
    """
    query_words = _tokenize(query)
    scored = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        doc_words = _tokenize(path.read_text())
        overlap = sum((query_words & doc_words).values())
        if overlap > 0:
            scored.append((overlap, path))
    scored.sort(key=lambda x: -x[0])
    top = scored[:2]
    if not top:
        return "No matching documentation found."
    return "\n\n---\n\n".join(f"# {path.name}\n{path.read_text()}" for _, path in top)


def escalate_to_human(ticket_summary: str) -> str:
    """Escalate a support ticket to a human agent (mock)."""
    return f"[MOCK] Escalated to human support queue:\n{ticket_summary}"


SYSTEM_PROMPT = (
    "You are a support agent for the deepagents-guide project. For every ticket, "
    "call search_docs first to look for relevant documentation before answering. "
    "If the docs answer the question, reply grounded in what you found. If nothing "
    "relevant turns up, call escalate_to_human with a concise summary instead of guessing."
)


def build_support_agent(model: str, checkpointer):
    return create_deep_agent(
        model=model,
        tools=[search_docs, escalate_to_human],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        interrupt_on={"escalate_to_human": True},
    )


def run_ticket(agent, ticket: str, thread_id: str, interactive: bool) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    agent.invoke({"messages": [{"role": "user", "content": ticket}]}, config=config)

    state = agent.get_state(config)
    while state.interrupts:
        interrupt_value = state.interrupts[0].value
        decisions = resolve_decisions(interrupt_value["action_requests"], interactive)
        agent.invoke(Command(resume={"decisions": decisions}), config=config)
        state = agent.get_state(config)

    messages = state.values.get("messages", [])
    final = messages[-1] if messages else None
    return getattr(final, "content", str(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Approve escalation manually via stdin")
    args = parser.parse_args()

    agent = build_support_agent(MODEL, MemorySaver())

    print(">>> Ticket 1 (covered by docs)\n")
    ticket_1 = "I'm getting 'No module named pydantic_core._pydantic_core' when I run a script, help?"
    print(f"  User: {ticket_1}")
    reply_1 = run_ticket(agent, ticket_1, str(uuid.uuid4()), args.interactive)
    print(f"  Agent: {reply_1}\n")

    print(">>> Ticket 2 (NOT covered by docs — should escalate)\n")
    ticket_2 = "Can you deploy this project to a Kubernetes cluster with autoscaling for me?"
    print(f"  User: {ticket_2}")
    reply_2 = run_ticket(agent, ticket_2, str(uuid.uuid4()), args.interactive)
    print(f"  Agent: {reply_2}\n")


if __name__ == "__main__":
    main()
