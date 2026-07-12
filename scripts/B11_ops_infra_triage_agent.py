"""
B11_ops_infra_triage_agent.py — Ops/infra triage agent (real-world use case, Part 2).

The orchestrator explores real log files (`workspace/ops-logs/`) using the
built-in read-only filesystem tools (`ls`/`read_file`/`grep`/`glob`), looks
for anomalies across multiple services, and files an incident — a mocked,
HITL-gated tool, same pattern as B4/B9/B10.

Unlike B5/B9, this one enforces read-only access with a real mechanism
instead of just a system-prompt instruction: `permissions=[FilesystemPermission
(operations=["write"], paths=["/**"], mode="deny")]` makes `write_file`/
`edit_file` return a permission-denied error for every path, regardless of
what the model is told or tries to do. This is the `permissions` param
`create_deep_agent` exposes for exactly this kind of "the agent should never
be able to touch this data" guarantee — stronger than an instruction, weaker
than removing the tools outright (removing them entirely is even stronger,
but this demonstrates the permission-rule mechanism, since B5 already showed
tool-removal-by-convention via a narrowly scoped sub-agent).

Run:
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B11_ops_infra_triage_agent.py

Run with manual approval on incident creation:
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B11_ops_infra_triage_agent.py --interactive
"""
import argparse
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from scripts.B5_repo_ops_agent import resolve_decisions

load_dotenv()

MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")
LOGS_DIR = Path(__file__).parent.parent / "workspace" / "ops-logs"


def create_incident(summary: str, severity: str) -> str:
    """File an incident ticket (mock). severity should be one of: low, medium, high, critical."""
    return f"[MOCK] Incident created (severity={severity}):\n{summary}"


SYSTEM_PROMPT = (
    "You are an ops/infra triage assistant. Explore the log files under / "
    "(use ls/glob/grep/read_file) across all services, identify any real "
    "anomaly (error spikes, resource exhaustion, timeouts), and determine "
    "which service and time window it affects. If you find a real issue, call "
    "create_incident with a concise root-cause summary and a severity. If "
    "everything looks normal, say so — don't file an incident for routine logs."
)


def build_ops_triage_agent(model: str, logs_dir: Path, checkpointer):
    # Read-only intent enforced by a permission rule, not just a prompt: write_file/
    # edit_file are denied for every path under this backend, regardless of what the
    # model attempts.
    backend = FilesystemBackend(root_dir=logs_dir, virtual_mode=True)
    return create_deep_agent(
        model=model,
        backend=backend,
        tools=[create_incident],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
        interrupt_on={"create_incident": True},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Approve incident creation manually via stdin")
    args = parser.parse_args()

    agent = build_ops_triage_agent(MODEL, LOGS_DIR, MemorySaver())

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    user_msg = "Check the logs for issues in the last hour and file an incident if needed."
    print(f">>> logs dir = {LOGS_DIR}")
    print(f">>> User: {user_msg}\n")

    agent.invoke({"messages": [{"role": "user", "content": user_msg}]}, config=config)

    state = agent.get_state(config)
    while state.interrupts:
        interrupt_value = state.interrupts[0].value
        decisions = resolve_decisions(interrupt_value["action_requests"], args.interactive)
        agent.invoke(Command(resume={"decisions": decisions}), config=config)
        state = agent.get_state(config)

    messages = state.values.get("messages", [])
    final = messages[-1] if messages else None
    print(f"\n>>> Agent: {getattr(final, 'content', final)}")


if __name__ == "__main__":
    main()
