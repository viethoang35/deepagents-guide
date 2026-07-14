"""
B8_langsmith_eval.py — Eval results pushed to LangSmith (dataset + feedback) (Step 8, cloud variant).

Builds on B7 (tracing) and the existing evals/test_repo_ops_eval.py. Instead of
only reading pytest results on the console, this pushes the eval outcome to
LangSmith so you can:
  1. Create a `repo-ops-eval` dataset (each example = 1 model under evaluation).
  2. Run the B5 repo-ops agent per model (reusing build_repo_ops_agent/run_to_completion).
  3. Attach a `pytest_pass` (1.0 / 0.0) feedback score to the exact run for that
     model, so you can compare models on the LangSmith UI.

Feedback is attached via an explicit `run_id` passed through the invoke config
— not by searching for it afterward. `client.list_runs(filter=...)` takes a
query DSL string (e.g. `'eq(run_type, "chain")'`), not raw JSON; searching by
`{"thread_id": ...}` (the original approach here) would silently return no
matches. Generating the run_id ourselves and reusing it directly with
`create_feedback` sidesteps that entirely.

Enable (in .env):
  LANGCHAIN_TRACING_V2=true
  LANGSMITH_API_KEY=lsv2_...
  LANGSMITH_PROJECT=deepagents-guide
  OPENROUTER_API_KEY=sk-or-...        # B5 calls a real model

Run offline (missing LangSmith key) -> verifies wiring, makes no LangSmith API calls:
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B8_langsmith_eval.py
Run for real: fill in both keys; the eval will create the dataset + push feedback to LangSmith.

Note: B5 uses LocalShellBackend (NO sandbox) -> only ever runs against a tmp copy
of sample-repo (the fixture already does this). Never run with auto-approve on a
machine with data you care about.
"""
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent
SAMPLE_REPO = REPO_ROOT / "workspace" / "sample-repo"
USER_MSG = (
    "Run the test suite. If anything fails, diagnose and fix the bug "
    "with a minimal edit, then rerun the tests to confirm it passes."
)
DATASET_NAME = os.getenv("LANGSMITH_EVAL_DATASET", "repo-ops-eval")
MODELS = tuple(
    dict.fromkeys(
        m.strip()
        for m in os.getenv(
            "EVAL_MODELS",
            ",".join([
                os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini"),
                os.getenv("RESEARCHER_MODEL", "openrouter:deepseek/deepseek-chat-v3"),
            ]),
        ).split(",")
        if m.strip()
    )
)


def has_langsmith_key() -> bool:
    """Only real when the key has a valid shape: starts with 'lsv2_' and is longer than 20 chars.
    Any placeholder value (lsv2_xxx, ls__, empty) -> False -> run offline, no API calls."""
    k = os.getenv("LANGSMITH_API_KEY", "").strip()
    return k.startswith("lsv2_") and len(k) > 20


def has_openrouter_key() -> bool:
    """Only real when the OpenRouter key has a valid shape: starts with 'sk-or-' and is longer than 20 chars."""
    k = os.getenv("OPENROUTER_API_KEY", "").strip()
    return k.startswith("sk-or-") and len(k) > 20


def build_dataset(client) -> None:
    """Create the repo-ops-eval dataset (idempotent: skip if it already exists)."""
    try:
        client.create_dataset(
            DATASET_NAME,
            description="Repo-ops eval: does the B5 agent fix the intentional bug per model?",
        )
        print(f">>> Created dataset '{DATASET_NAME}'")
    except Exception as e:
        # dataset already exists, or some other error -> skip and continue
        print(f">>> Dataset '{DATASET_NAME}' already exists or was skipped: {e}")


def run_model_eval(model: str, client) -> float:
    """Run the B5 agent for 1 model (in a tmp repo), return 1.0 if pytest passes, 0.0 if it fails.
    Pushes feedback to LangSmith on the exact run, via an explicit run_id, if a client is set."""
    from langgraph.checkpoint.memory import MemorySaver
    from scripts.B5_repo_ops_agent import build_repo_ops_agent, run_to_completion

    import tempfile
    dest = Path(tempfile.mkdtemp()) / "sample-repo"
    shutil.copytree(SAMPLE_REPO, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))

    agent = build_repo_ops_agent(model, dest, MemorySaver())
    run_id = uuid.uuid4()
    config = {"configurable": {"thread_id": f"eval-{model}"}, "run_id": run_id}
    run_to_completion(agent, config, USER_MSG, interactive=False)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=dest, capture_output=True, text=True, timeout=60,
    )
    score = 1.0 if result.returncode == 0 else 0.0
    print(f">>> [{model}] pytest_returncode={result.returncode} -> pytest_pass={score}")

    if client is not None:
        try:
            client.create_feedback(run_id, "pytest_pass", score=score)
            print(f">>> Pushed pytest_pass={score} feedback to run {run_id}")
        except Exception as e:
            print(f">>> (warn) failed to push feedback: {e}")
    return score


def main() -> None:
    print(f">>> B8 LangSmith eval | dataset='{DATASET_NAME}' | models={MODELS}")
    client = None
    if has_langsmith_key() and has_openrouter_key():
        from langsmith import Client
        client = Client()
        build_dataset(client)
    else:
        print("\n[OFFLINE] Missing a valid LANGSMITH_API_KEY or OPENROUTER_API_KEY "
              "-> verifying wiring only, no API calls made.")
        print("For a real eval: .env needs LANGCHAIN_TRACING_V2=true + LANGSMITH_API_KEY (lsv2_...) + OPENROUTER_API_KEY (sk-or-...).")
        # Offline: still confirm B5 imports correctly to prove the wiring is right
        try:
            from scripts.B5_repo_ops_agent import build_repo_ops_agent, run_to_completion
            print(">>> Import of B5 OK (build_repo_ops_agent / run_to_completion available).")
        except Exception as e:
            print(f">>> (warn) failed to import B5: {e}")
        return

    for model in MODELS:
        try:
            run_model_eval(model, client)
        except Exception as e:
            print(f">>> [{model}] eval error: {e}")
    print(f"\n>>> Open https://smith.langchain.com/project/{os.getenv('LANGSMITH_PROJECT', 'deepagents-guide')} "
          f"-> dataset '{DATASET_NAME}' to compare models.")


if __name__ == "__main__":
    main()
