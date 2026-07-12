# CHECKLOG

Tổng hợp các thay đổi đã thực hiện trong dự án `deepagents-guide`, theo thời gian.
Mục tiêu: giúp bất kỳ ai (kể cả future-you) nắm được repo đã đi từ đâu tới đâu mà
không phải đọc lại toàn bộ lịch sử chat.

## 2026-07-12

### Thêm mới — scripts (agent)

- **`scripts/B4_hardened_agent.py`** — 3 lớp hardening cho harness:
  - Checkpointer bền (`SqliteSaver` → `workspace/b4_checkpoints.db`), resume đúng
    thread sau khi tắt/mở lại process.
  - Human-in-the-loop (`interrupt_on`) cho tool có side effect (`send_email`,
    `delete_file`); tool an toàn (`get_weather`) chạy thẳng không cần duyệt.
  - Observability nhẹ: callback handler cục bộ log tool call + token usage,
    không cần LangSmith.
  - Đã verify: chạy thật (auto-approve + interactive), xác nhận resume qua
    `thread_id` cũ đọc đúng lịch sử, và reject thực sự chặn hành động.

- **`scripts/B5_repo_ops_agent.py`** — Repo-ops agent trên hardened base:
  - Dùng `LocalShellBackend` (filesystem + shell THẬT, giới hạn
    `workspace/sample-repo/` qua `virtual_mode=True`).
  - Sub-agent `runner-agent` chạy `pytest` thật; orchestrator tự đọc/sửa file khi
    test fail rồi chạy lại để xác nhận.
  - Refactor thành `build_repo_ops_agent()` + `run_to_completion()` để dùng lại
    trong `evals/test_repo_ops_eval.py` (không copy-paste logic).
  - Đã verify: chạy thật với model mặc định, agent tự phục hồi khi `python` không
    có trên PATH, cài `pytest` qua `ensurepip`, chẩn đoán đúng bug, sửa, và xác
    nhận test pass. Test cả nhánh reject qua `--interactive`.
  - ⚠️ Ghi chú an toàn: `LocalShellBackend` không sandbox — `execute` không bị
    giới hạn bởi `virtual_mode`. Lớp an toàn duy nhất là HITL.

- **`scripts/B6_cross_thread_memory.py`** — Cross-thread memory (`memory=[...]`),
  phân biệt với checkpointer (B4):
  - Checkpointer nhớ *trong 1 thread*; `memory=[...]` nhớ *xuyên thread* qua 1 file
    `AGENTS.md` thật trên đĩa (`workspace/agent_memory/AGENTS.md`).
  - Đã verify: thread A dạy agent 1 sở thích, thread B (thread_id mới, không liên
    quan) vẫn áp dụng đúng sở thích đó.

### Thêm mới — eval

- **`evals/test_repo_ops_eval.py`** — pytest thường (không cần LangSmith, khác bộ
  eval đầy đủ ở `vendor-deepagents/libs/evals`), copy `sample-repo` vào thư mục tmp
  riêng, chạy agent B5 với model được parametrize, assert `pytest` thật pass sau đó.
  - Đã verify với 2 model mặc định: **`gpt-4o-mini` pass, `deepseek-chat-v3`
    fail** — phát hiện thật, không phải giả định, đúng mục đích của eval này.

### Thêm mới — dữ liệu demo

- `workspace/sample-repo/{calc.py,test_calc.py}` — bug cố ý (`average()` trừ dư 1)
  để B5/eval có cái thật để chẩn đoán. Giữ nguyên ở trạng thái "có bug" sau mỗi lần
  demo để lần chạy tiếp theo còn ý nghĩa.
- `workspace/agent_memory/AGENTS.md` — sinh ra từ lần chạy demo B6 (memory thật).

### Cập nhật

- `pyproject.toml`: thêm `langgraph-checkpoint-sqlite` (checkpointer bền cho B4/B5)
  và `pytest` (chạy `evals/`).
- `.env.example`: thêm biến LangSmith tuỳ chọn (`LANGSMITH_TRACING`,
  `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`).
- `README.md`: thêm Bước 4–8 (hardened harness, repo-ops agent, cross-thread
  memory, LangSmith tracing, eval script), cập nhật cây cấu trúc thư mục.

### Chưa verify / giới hạn đã biết

- LangSmith tracing: xác nhận đúng cơ chế (biến env chuẩn của LangChain, không
  cần sửa code), **chưa** test end-to-end với key thật vì máy này chưa có key.
- `LocalShellBackend` trong B5 không có sandbox thật — muốn cách ly cứng
  (Docker/VM) cần extend `BaseSandbox` (xem `vendor-deepagents/libs/partners/`:
  modal, runloop, daytona).
