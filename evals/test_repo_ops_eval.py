"""
test_repo_ops_eval.py — Eval nhẹ: agent B5 có còn fix đúng bug khi đổi model?

Đây KHÔNG phải là unit test cho code của bạn — nó gọi API model thật (tốn
token) để kiểm tra hành vi của AGENT. Mục đích: khi bạn đổi `DEEPAGENTS_MODEL`
(ví dụ từ gpt-4o-mini sang 1 model rẻ hơn), test này cho biết ngay agent còn
đủ khả năng chẩn đoán + sửa bug hay không, thay vì phải chạy tay B5 và đọc log.

Khác với bộ eval đầy đủ ở `vendor-deepagents/libs/evals` (đòi hỏi LangSmith
tracing bắt buộc, registry riêng, radar chart...), file này chỉ dùng pytest
thường — chạy được ngay, không cần setup thêm. Nếu bạn có LANGSMITH_TRACING=true
(Bước 7 trong README), trace của các lần eval này cũng lên LangSmith luôn vì
đó chỉ là biến env.

Mỗi test case:
  1. Copy `workspace/sample-repo/` (bug cố ý) vào 1 thư mục tmp riêng (không
     đụng vào bản demo).
  2. Build agent B5 (dùng lại `build_repo_ops_agent`/`run_to_completion` từ
     `scripts/B5_repo_ops_agent.py` — không viết lại logic).
  3. Chạy agent với auto-approve (giống demo mode, không cần người duyệt).
  4. Chạy pytest THẬT trong thư mục tmp, assert exit code == 0.

Chạy:
  env -u PYTHONPATH uv run --no-sync .venv/bin/python -m pytest evals/ -v

Đổi danh sách model muốn eval qua biến env (mặc định: model của B2 + B3-researcher):
  EVAL_MODELS="openrouter:openai/gpt-4o-mini,openrouter:deepseek/deepseek-chat-v3"
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from scripts.B5_repo_ops_agent import build_repo_ops_agent, run_to_completion

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent
SAMPLE_REPO = REPO_ROOT / "workspace" / "sample-repo"

DEFAULT_MODELS = (
    os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini"),
    os.getenv("RESEARCHER_MODEL", "openrouter:deepseek/deepseek-chat-v3"),
)
MODELS = tuple(
    dict.fromkeys(  # khử trùng, giữ thứ tự, phòng khi 2 default trùng nhau
        m.strip()
        for m in os.getenv("EVAL_MODELS", ",".join(DEFAULT_MODELS)).split(",")
        if m.strip()
    )
)

requires_openrouter_key = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="Eval này gọi model thật qua OpenRouter — cần OPENROUTER_API_KEY trong .env",
)

USER_MSG = (
    "Run the test suite. If anything fails, diagnose and fix the bug "
    "with a minimal edit, then rerun the tests to confirm it passes."
)


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Copy sample-repo (bug cố ý) vào thư mục tmp riêng cho mỗi test case."""
    dest = tmp_path / "sample-repo"
    shutil.copytree(SAMPLE_REPO, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return dest


@requires_openrouter_key
@pytest.mark.parametrize("model", MODELS)
def test_agent_fixes_bug(model: str, tmp_repo: Path) -> None:
    agent = build_repo_ops_agent(model, tmp_repo, MemorySaver())
    config = {"configurable": {"thread_id": f"eval-{model}"}}

    run_to_completion(agent, config, USER_MSG, interactive=False)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=tmp_repo,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"[{model}] agent did not leave the repo in a passing state:\n"
        f"{result.stdout}\n{result.stderr}"
    )
