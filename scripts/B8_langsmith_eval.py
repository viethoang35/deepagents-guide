"""
B8_langsmith_eval.py — Eval -> LangSmith dataset + feedback (Step 8, mở rộng tracing-labs).

Nối tiếp B7 (trace) và test_repo_ops_eval.py (đã có). Thay vì chỉ đọc kết quả pytest
trên console, B8 ĐẨY kết quả eval lên LangSmith để:
  1. Tạo dataset `repo-ops-eval` (mỗi example = 1 model cần đánh giá).
  2. Chạy B5 repo-ops agent trên từng model (tái dùng build_repo_ops_agent/run_to_completion).
  3. Gửi feedback `pytest_pass` (1.0 / 0.0) lên run tương ứng -> so sánh model có số
     trên LangSmith UI (giống wiki đã ghi: gpt-4o-mini PASS, deepseek FAIL).

Cách bật (trong .env):
  LANGCHAIN_TRACING_V2=true
  LANGSMITH_API_KEY=lsv2_...
  LANGSMITH_PROJECT=deepagents-guide
  OPENROUTER_API_KEY=sk-or-...        # B5 gọi model thật

Chạy OFFLINE (thiếu key) -> verify cấu trúc, không gọi API:
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B8_langsmith_eval.py
Chạy THẬT: điền đủ key, eval sẽ tạo dataset + gửi feedback lên LangSmith.

Lưu ý: B5 dùng LocalShellBackend (KHÔNG sandbox) -> chỉ chạy trên sample-repo
copy trong tmp (fixture đã làm vậy). Không chạy auto-approve trên máy có dữ liệu quan trọng.
"""
import os
import shutil
import subprocess
import sys
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
    """Chỉ THẬT khi key có định dạng hợp lệ: bắt đầu bằng 'lsv2_' và dài > 20 ký tự.
    Mọi giá trị giả (lsv2_xxx, ls__, rỗng) -> False -> chạy offline, KHÔNG gọi API."""
    k = os.getenv("LANGSMITH_API_KEY", "").strip()
    return k.startswith("lsv2_") and len(k) > 20


def has_openrouter_key() -> bool:
    """Chỉ THẬT khi key OpenRouter hợp lệ: bắt đầu bằng 'sk-or-' và dài > 20 ký tự."""
    k = os.getenv("OPENROUTER_API_KEY", "").strip()
    return k.startswith("sk-or-") and len(k) > 20


def build_dataset(client) -> None:
    """Tạo dataset repo-ops-eval (idempotent: bỏ qua nếu đã có)."""
    try:
        client.create_dataset(
            DATASET_NAME,
            description="Repo-ops eval: does the B5 agent fix the intentional bug per model?",
        )
        print(f">>> Đã tạo dataset '{DATASET_NAME}'")
    except Exception as e:
        # dataset đã tồn tại hoặc lỗi khác -> bỏ qua để tiếp tục
        print(f">>> Dataset '{DATASET_NAME}' đã có hoặc bỏ qua: {e}")


def run_model_eval(model: str, client) -> float:
    """Chạy B5 agent trên 1 model (trong tmp repo), trả về 1.0 nếu pytest pass, 0.0 nếu fail.
    Gửi feedback lên LangSmith nếu có key thật."""
    from langgraph.checkpoint.memory import MemorySaver
    from scripts.B5_repo_ops_agent import build_repo_ops_agent, run_to_completion

    import tempfile
    dest = Path(tempfile.mkdtemp()) / "sample-repo"
    shutil.copytree(SAMPLE_REPO, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))

    agent = build_repo_ops_agent(model, dest, MemorySaver())
    config = {"configurable": {"thread_id": f"eval-{model}"}}
    run_to_completion(agent, config, USER_MSG, interactive=False)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=dest, capture_output=True, text=True, timeout=60,
    )
    score = 1.0 if result.returncode == 0 else 0.0
    print(f">>> [{model}] pytest_returncode={result.returncode} -> pytest_pass={score}")

    if client is not None:
        try:
            # Gắn feedback lên run cuối của thread (best-effort)
            runs = list(client.list_runs(
                project_name=os.getenv("LANGSMITH_PROJECT", "deepagents-guide"),
                filter=f'{{"thread_id": "{config["configurable"]["thread_id"]}"}}',
                limit=1,
            ))
            if runs:
                client.create_feedback(runs[0].id, "pytest_pass", score=score)
                print(f">>> Đã gửi feedback pytest_pass={score} lên run {runs[0].id}")
        except Exception as e:
            print(f">>> (warn) gửi feedback thất bại: {e}")
    return score


def main() -> None:
    print(f">>> B8 LangSmith eval | dataset='{DATASET_NAME}' | models={MODELS}")
    client = None
    if has_langsmith_key() and has_openrouter_key():
        from langsmith import Client
        client = Client()
        build_dataset(client)
    else:
        print("\n[OFFLINE] Thiếu LANGSMITH_API_KEY hoặc OPENROUTER_API_KEY thật (định dạng hợp lệ) "
              "-> verify cấu trúc, KHÔNG gọi API.")
        print("Để eval THẬT: .env có LANGCHAIN_TRACING_V2=true + LANGSMITH_API_KEY (lsv2_...) + OPENROUTER_API_KEY (sk-or-...).")
        # Offline: vẫn import được B5 để chứng minh wiring đúng
        try:
            from scripts.B5_repo_ops_agent import build_repo_ops_agent, run_to_completion
            print(">>> Import B5 OK (build_repo_ops_agent / run_to_completion available).")
        except Exception as e:
            print(f">>> (warn) import B5: {e}")
        return

    for model in MODELS:
        try:
            run_model_eval(model, client)
        except Exception as e:
            print(f">>> [{model}] eval lỗi: {e}")
    print(f"\n>>> Mở https://smith.langchain.com/project/{os.getenv('LANGSMITH_PROJECT','deepagents-guide')} "
          f"-> dataset '{DATASET_NAME}' để so sánh model.")


if __name__ == "__main__":
    main()
