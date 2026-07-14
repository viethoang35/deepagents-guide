"""
B8b_phoenix_eval.py — Compare models on LOCAL PHOENIX, no API key needed (Step 8b).

Unlike B8 (pushes to LangSmith cloud, needs LANGSMITH_API_KEY), B8b writes the
full eval trace + pytest_pass score to **Phoenix running locally**
(http://localhost:6006) -> compare models COMPLETELY OFFLINE, no LangSmith
account, $0.

Per-model flow:
  1. Copy workspace/sample-repo (intentional bug) into a tmp directory.
  2. Build the repo-ops agent (reusing B5) and run it with auto-approve.
  3. Run real pytest in the tmp dir -> pytest_pass = 1.0 if it passes, 0.0 if it fails.
  4. Record 1 "eval_run" span in Phoenix with attributes: model, pytest_pass, returncode.
     If the run itself errored (e.g. an invalid API key) instead of the model
     genuinely failing the task, that's recorded as eval.status="error" with the
     error message, and excluded from the pass/fail comparison — an
     infrastructure failure should never look like "the model failed the bug fix."
  -> Open localhost:6006 -> see the comparison table for gpt-4o-mini vs deepseek
     (pass/fail, latency).

Start Phoenix first (Terminal 1):
  env -u PYTHONPATH uv run --no-sync python -m phoenix.server.main serve
Run the eval (Terminal 2):
  # OFFLINE (fake model, verifies wiring without calling the API):
  OPENROUTER_API_KEY=sk-or-fake env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B8b_phoenix_eval.py
  # LIVE (needs a valid OPENROUTER_API_KEY):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B8b_phoenix_eval.py

Note: B5 uses LocalShellBackend (NO sandbox) -> only ever runs against a tmp
copy of sample-repo. Never auto-approve on a machine with data you care about.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent
SAMPLE_REPO = REPO_ROOT / "workspace" / "sample-repo"
USER_MSG = (
    "Run the test suite. If anything fails, diagnose and fix the bug "
    "with a minimal edit, then rerun the tests to confirm it passes."
)
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
PROJECT = os.getenv("LANGSMITH_PROJECT", "deepagents-guide")


def setup_tracer():
    """Register a tracer that ships spans to Phoenix running locally (http://localhost:6006/v1/traces)."""
    provider = TracerProvider()
    try:
        from phoenix.otel import register
        provider = register(
            project_name=f"{PROJECT}-phoenix-eval",
            endpoint="http://localhost:6006/v1/traces",
        )
        print(f">>> Registered tracer with local Phoenix (project '{PROJECT}-phoenix-eval')")
    except Exception as e:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        print(f">>> (warn) Could not connect to Phoenix ({e}). Using ConsoleSpanExporter (prints to stdout).")
        print("         To see the UI: run `uv run --no-sync python -m phoenix.server.main serve`.")
    return trace.get_tracer(__name__)


def has_openrouter_key() -> bool:
    k = os.getenv("OPENROUTER_API_KEY", "").strip()
    return k.startswith("sk-or-") and len(k) > 20


def run_model_eval(model: str, tracer, *, fake: bool = False) -> float | None:
    """Run the eval for 1 model in a tmp repo, return pytest_pass (1.0/0.0), or
    None if the run itself errored (infrastructure failure, not a model failure).
    If fake=True (offline): SKIP the agent (Deep Agent would call the model when
    built), only run real pytest against sample-repo as-is to produce a genuine
    span for Phoenix."""
    dest = Path(tempfile.mkdtemp()) / "sample-repo"
    shutil.copytree(SAMPLE_REPO, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))

    with tracer.start_as_current_span("eval_run") as span:
        span.set_attribute("eval.model", model)
        span.set_attribute("eval.fake", fake)
        if fake:
            # Offline: don't build the agent (avoid calling OpenRouter). pytest's
            # result on the unmodified repo stands in for a real score.
            span.set_attribute("eval.mode", "offline-proxy")
        else:
            from langgraph.checkpoint.memory import MemorySaver
            from scripts.B5_repo_ops_agent import build_repo_ops_agent, run_to_completion
            try:
                agent = build_repo_ops_agent(model, dest, MemorySaver())
                config = {"configurable": {"thread_id": f"eval-{model}"}}
                run_to_completion(agent, config, USER_MSG, interactive=False)
            except Exception as e:
                span.set_attribute("eval.status", "error")
                span.set_attribute("eval.error", str(e))
                print(f">>> [{model}] agent run errored (not a model failure): {e}")
                return None

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=dest, capture_output=True, text=True, timeout=60,
        )
        score = 1.0 if result.returncode == 0 else 0.0
        span.set_attribute("eval.status", "ok")
        span.set_attribute("eval.pytest_pass", score)
        span.set_attribute("eval.pytest_returncode", result.returncode)
        print(f">>> [{model}] returncode={result.returncode} pytest_pass={score}")
    return score


def main() -> None:
    print(f">>> B8b Phoenix eval | models={MODELS}")
    tracer = setup_tracer()

    fake = not has_openrouter_key()
    if fake:
        print("\n[OFFLINE] No valid OPENROUTER_API_KEY -> running the eval with a FAKE model "
              "(no API calls), still recording spans in Phoenix to verify the pipeline.")
    else:
        print("\n[LIVE] Running the eval with a real model via OpenRouter, recording traces to Phoenix.")

    scores: dict[str, float | None] = {}
    for model in MODELS:
        scores[model] = run_model_eval(model, tracer, fake=fake)

    print("\n=== Eval summary (model comparison) ===")
    for m, s in scores.items():
        label = "ERROR (infra failure, not a model result)" if s is None else f"pytest_pass={s}"
        print(f"  {m}: {label}")
    print(f"\n>>> Open http://localhost:6006 (project '{PROJECT}-phoenix-eval') to see traces + comparison.")


if __name__ == "__main__":
    main()
