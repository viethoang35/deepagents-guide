"""
B8b_phoenix_eval.py — Eval so sánh model TRÊN PHOENIX LOCAL (Step 8b, không cần key).

Khác B8 (đẩy lên LangSmith cloud cần LANGSMITH_API_KEY), B8b ghi toàn bộ eval
trace + điểm pytest_pass vào **Phoenix chạy local** (http://localhost:6006) ->
so sánh model HOÀN TOÀN OFFLINE, KHÔNG cần LangSmith account, $0.

Flow mỗi model:
  1. Copy workspace/sample-repo (bug có chủ ý) vào tmp.
  2. Build repo-ops agent (tái dùng B5) + chạy auto-approve.
  3. Chạy pytest thật trong tmp -> pytest_pass = 1.0 nếu pass, 0.0 nếu fail.
  4. Ghi 1 span "eval_run" vào Phoenix với attributes: model, pytest_pass, returncode.
  -> Mở localhost:6006 -> xem bảng so sánh gpt-4o-mini vs deepseek (pass/fail, latency).

Chạy Phoenix trước (Terminal 1):
  env -u PYTHONPATH uv run --no-sync python -m phoenix.server.main serve
Chạy eval (Terminal 2):
  # OFFLINE (fake model, verify không gọi API):
  OPENROUTER_API_KEY=sk-or-fake env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B8b_phoenix_eval.py
  # THẬT (cần OPENROUTER_API_KEY hợp lệ):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B8b_phoenix_eval.py

Lưu ý: B5 dùng LocalShellBackend (KHÔNG sandbox) -> chỉ chạy trên tmp copy của
sample-repo. Không auto-approve trên máy có dữ liệu quan trọng.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime

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
    """Đăng ký tracer gửi vào Phoenix local (http://localhost:6006/v1/traces)."""
    provider = TracerProvider()
    try:
        from phoenix.otel import register
        provider = register(
            project_name=f"{PROJECT}-phoenix-eval",
            endpoint="http://localhost:6006/v1/traces",
        )
        print(f">>> Đã đăng ký tracer với Phoenix local (project '{PROJECT}-phoenix-eval')")
    except Exception as e:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        print(f">>> (warn) Không kết nối Phoenix ({e}). Dùng ConsoleSpanExporter (in ra stdout).")
        print("         Để xem UI: chạy `uv run --no-sync python -m phoenix.server.main serve`.")
    return trace.get_tracer(__name__)


def has_openrouter_key() -> bool:
    k = os.getenv("OPENROUTER_API_KEY", "").strip()
    return k.startswith("sk-or-") and len(k) > 20


def run_model_eval(model: str, tracer, *, fake: bool = False) -> float:
    """Chạy eval 1 model trong tmp repo, trả pytest_pass (1.0/0.0).
    Nếu fake=True (offline): BỎ QUA agent (Deep Agent gọi model khi build),
    chỉ chạy pytest thật trên sample-repo để sinh span có điểm thực lên Phoenix."""
    dest = Path(tempfile.mkdtemp()) / "sample-repo"
    shutil.copytree(SAMPLE_REPO, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))

    with tracer.start_as_current_span("eval_run") as span:
        span.set_attribute("eval.model", model)
        span.set_attribute("eval.fake", fake)
        if fake:
            # Offline: không build agent (tránh gọi OpenRouter). Dùng pytest làm proxy điểm.
            span.set_attribute("eval.mode", "offline-proxy")
        else:
            from langgraph.checkpoint.memory import MemorySaver
            from scripts.B5_repo_ops_agent import build_repo_ops_agent, run_to_completion
            agent = build_repo_ops_agent(model, dest, MemorySaver())
            config = {"configurable": {"thread_id": f"eval-{model}"}}
            run_to_completion(agent, config, USER_MSG, interactive=False)

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=dest, capture_output=True, text=True, timeout=60,
        )
        score = 1.0 if result.returncode == 0 else 0.0
        span.set_attribute("eval.pytest_pass", score)
        span.set_attribute("eval.pytest_returncode", result.returncode)
        print(f">>> [{model}] returncode={result.returncode} pytest_pass={score}")
    return score


def main() -> None:
    print(f">>> B8b Phoenix eval | models={MODELS}")
    tracer = setup_tracer()

    fake = not has_openrouter_key()
    if fake:
        print("\n[OFFLINE] Thiếu OPENROUTER_API_KEY thật -> chạy eval bằng FAKE model "
              "(không gọi API), vẫn ghi span vào Phoenix để verify pipeline.")
    else:
        print("\n[THẬT] Chạy eval với model thật qua OpenRouter, ghi trace vào Phoenix.")

    scores = {}
    for model in MODELS:
        try:
            scores[model] = run_model_eval(model, tracer, fake=fake)
        except Exception as e:
            print(f">>> [{model}] eval lỗi: {e}")
            scores[model] = 0.0

    print("\n=== Tóm tắt eval (so sánh model) ===")
    for m, s in scores.items():
        print(f"  {m}: pytest_pass={s}")
    print(f"\n>>> Mở http://localhost:6006 (project '{PROJECT}-phoenix-eval') để xem trace + so sánh.")


if __name__ == "__main__":
    main()
