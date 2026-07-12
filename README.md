# Deep Agents Hands-on Guide

Dự án mẫu thực hành LangChain Deep Agents (verify trên macOS aarch64, Python 3.12, uv 0.11).

## Cấu trúc
```
deepagents-guide/
├── pyproject.toml          # deps: deepagents, langchain, langgraph, provider packages
├── .env.example            # copy thành .env, điền 1 model key
├── scripts/
│   ├── B2_quickstart.py    # agent đơn + 1 tool tự viết (verify chạy tới API call)
│   ├── B2_advanced.py      # sub-agent (dict spec) + filesystem + streaming
│   ├── B3_nemotron_deep_agent.py  # multi-model: orchestrator + researcher sub-agent (OpenRouter)
│   ├── B4_hardened_agent.py       # checkpointer bền + human-in-the-loop + observability nhẹ
│   ├── B5_repo_ops_agent.py       # agent đọc/sửa file THẬT + sub-agent chạy pytest THẬT (trên hardened base)
│   └── B6_cross_thread_memory.py  # memory=[...]: nhớ XUYÊN thread (khác checkpointer B4 chỉ nhớ trong 1 thread)
├── evals/
│   └── test_repo_ops_eval.py      # pytest: agent B5 còn fix đúng bug không khi đổi model
├── vendor-deepagents/      # clone langchain-ai/deepagents (ví dụ nvidia_deep_agent)
└── vendor-nemoclaw/        # clone NVIDIA/nemoclaw-community (secure blueprint)
```

`vendor-deepagents/` và `vendor-nemoclaw/` **không** được commit vào repo này (xem
`.gitignore`) — chúng là clone của repo người khác, tự clone lại khi cần:
```bash
git clone https://github.com/langchain-ai/deepagents.git vendor-deepagents
git clone https://github.com/NVIDIA/nemoclaw-community.git vendor-nemoclaw
```

## Chạy nhanh (B1 + B2)
```bash
uv sync                                   # B1: cài deps (đã sync sẵn trên máy này)
cp .env.example .env                      # tạo file env
# MỞ .env, điền: OPENROUTER_API_KEY=sk-or-... (bắt buộc để chạy B2/B3)
# LƯU Ý: nếu shell có PYTHONPATH trỏ venv khác (như môi trường Hermes), dùng env -u PYTHONPATH
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B2_quickstart.py
```

## Model qua OpenRouter (1 key, nhiều model)
Deep Agents hỗ trợ prefix `openrouter:`. Định dạng: `openrouter:<provider>/<model>`.
Ví dụ trong `.env`:
```
OPENROUTER_API_KEY=sk-or-...
DEEPAGENTS_MODEL=openrouter:deepseek/deepseek-chat-v3
# hoặc: openrouter:x-ai/grok-4.5
# hoặc: openrouter:openai/gpt-4o-mini
```
Đăng ký key: https://openrouter.ai/keys

## Bước 3 — Multi-model Deep Agent (Nemotron style, qua OpenRouter)
Rút gọn từ ví dụ `vendor-deepagents/examples/nvidia_deep_agent`, nhưng gọi cả 2 model qua
OpenRouter (1 key). Orchestrator = model mạnh (grok-4.5), researcher sub-agent = model rẻ/nhanh (deepseek).
```bash
# .env: ORCHESTRATOR_MODEL + RESEARCHER_MODEL (đã có sẵn trong .env.example)
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B3_nemotron_deep_agent.py
```
Biến `.env` cho B3:
```
ORCHESTRATOR_MODEL=openrouter:x-ai/grok-4.5
RESEARCHER_MODEL=openrouter:deepseek/deepseek-chat-v3
```

## Bước 4 — Harden the harness (checkpointer + human-in-the-loop + observability)
`B4_hardened_agent.py` lấy lại orchestrator đơn giản nhưng thêm 3 thứ cần cho môi trường thật:
- **Checkpointer bền** (`SqliteSaver`): state lưu vào `workspace/b4_checkpoints.db`, resume lại đúng
  thread sau khi tắt/mở lại process.
- **Human-in-the-loop** (`interrupt_on`): tool có side effect thật (`send_email`, `delete_file`) bị
  dừng lại chờ người duyệt (`approve`/`reject`) trước khi chạy; tool an toàn (`get_weather`) chạy thẳng.
- **Observability nhẹ**: callback handler cục bộ log tool call + token usage, không cần LangSmith.
  Nếu có LangSmith, chỉ cần set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` trong `.env`, không sửa code.

```bash
# Demo tự động duyệt (không cần nhập tay):
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B4_hardened_agent.py

# Duyệt thủ công từng tool nhạy cảm (approve/reject qua stdin):
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B4_hardened_agent.py --interactive

# Resume 1 thread cũ (thread_id được in ra ở lần chạy trước):
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B4_hardened_agent.py --thread-id <id>
```

## Bước 5 — Repo-ops agent (filesystem + shell THẬT, trên hardened base)
`B5_repo_ops_agent.py` dùng `LocalShellBackend` (filesystem + shell thật trên máy bạn,
giới hạn trong `workspace/sample-repo/` qua `virtual_mode=True` + `root_dir`) thay cho
virtual filesystem giả lập của B2. Orchestrator ủy quyền cho sub-agent `runner-agent`
chạy `pytest` thật; nếu test fail, orchestrator tự đọc file, sửa lỗi bằng `edit_file`,
rồi chạy lại test để xác nhận. Mọi `write_file`/`edit_file`/`execute` đều bị chặn chờ
duyệt (tái dùng đúng cơ chế HITL từ Bước 4).

`workspace/sample-repo/` chứa 1 bug cố ý (`calc.py` trừ dư `1` trong `average()`) để
agent có cái thật để chẩn đoán — nếu bạn thấy nó đã "sạch" (không còn bug) là vì lần
chạy trước agent đã fix xong, sửa lại file để demo lại từ đầu.

⚠️ `LocalShellBackend` **không sandbox** — `execute` chạy lệnh shell thật với quyền
của bạn, không bị giới hạn bởi `virtual_mode` (chỉ các tool filesystem mới bị giới
hạn). An toàn duy nhất ở đây là lớp HITL. Muốn cách ly thật (Docker/VM), xem
`vendor-deepagents/libs/partners/` (modal, runloop, daytona) và extend `BaseSandbox`.

```bash
# Demo tự động duyệt:
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B5_repo_ops_agent.py

# Duyệt thủ công từng write_file/edit_file/execute:
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B5_repo_ops_agent.py --interactive
```

## Bước 6 — Cross-thread memory (khác checkpointer)
`B6_cross_thread_memory.py` minh họa `memory=[...]`, dễ nhầm với checkpointer (B4)
nhưng là 2 cơ chế khác nhau:
- **Checkpointer** (B4): nhớ *trong cùng 1 thread* — lịch sử message, resume qua `thread_id`.
- **`memory=[...]`** (bài này): nhớ *xuyên thread* — 1 file `AGENTS.md` THẬT trên đĩa
  (`workspace/agent_memory/AGENTS.md`), nạp vào system prompt ở đầu MỌI conversation,
  agent tự cập nhật qua `edit_file` khi học được điều gì mới (qua HITL duyệt).

Script chạy 2 "conversation" với `thread_id` hoàn toàn khác nhau: conversation A dạy
agent 1 sở thích (trả lời dạng bullet point), conversation B (thread mới, không liên
quan) vẫn trả lời đúng theo sở thích đó — chứng minh memory sống ngoài phạm vi 1 thread.

```bash
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B6_cross_thread_memory.py
```

## Bước 7 (tuỳ chọn) — LangSmith tracing
Áp dụng cho **mọi** script ở trên, không cần sửa 1 dòng code nào — LangChain tự đọc
biến env này. Set trong `.env` (xem `.env.example`):
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=deepagents-guide
```
Sau đó chạy bất kỳ script nào (B2-B5) như bình thường — trajectory đầy đủ (tool call,
sub-agent delegation, token usage per step) sẽ xuất hiện trên smith.langchain.com theo
project name, dễ debug hơn nhiều so với đọc log console. Hữu ích nhất cho B3/B5 vì có
nhiều bước orchestrator ↔ sub-agent khó theo dõi qua print.

Lưu ý: mục này mới verify được là *đúng cơ chế* (biến env chuẩn của LangChain, không
cần code) — chưa test end-to-end với key LangSmith thật vì máy này chưa có key đó.

## Bước 8 — Eval: agent B5 có còn fix đúng bug khi đổi model không?
`evals/test_repo_ops_eval.py` là 1 pytest thường (không cần LangSmith, khác bộ eval
đầy đủ ở `vendor-deepagents/libs/evals` vốn *bắt buộc* LangSmith tracing). Mỗi test
case: copy `workspace/sample-repo/` (bug cố ý) vào 1 thư mục tmp riêng, chạy agent B5
(tái dùng `build_repo_ops_agent`/`run_to_completion`, auto-approve), rồi chạy `pytest`
thật trong thư mục tmp và assert nó pass.

```bash
env -u PYTHONPATH uv run --no-sync .venv/bin/python -m pytest evals/ -v

# Đổi danh sách model muốn eval:
EVAL_MODELS="openrouter:openai/gpt-4o-mini,openrouter:x-ai/grok-4.5" \
  env -u PYTHONPATH uv run --no-sync .venv/bin/python -m pytest evals/ -v
```

Đã verify thật: chạy với 2 model mặc định (`gpt-4o-mini` và `deepseek-chat-v3`) thì
**`gpt-4o-mini` pass nhưng `deepseek-chat-v3` fail** — agent dùng deepseek không sửa
xong bug trong lần chạy đó. Đây chính xác là giá trị của eval này: bạn biết ngay khi
đổi model mà không cần tự chạy tay B5 và đọc log từng bước.

## Lỗi thường gặp & fix
- `No module named 'pydantic_core._pydantic_core'`: do PYTHONPATH kế thừa từ venv khác.
  Fix: chạy với `env -u PYTHONPATH` hoặc unset PYTHONPATH trước.
- `ModuleNotFoundError: langchain_openai`: chưa cài provider package → đã thêm vào pyproject.
- `TypeError: ... unexpected keyword argument 'sub_agents'`: API thật dùng `subagents` (không gạch dưới).
- `KeyError: 'system_prompt'` khi truyền sub-agent: dùng **dict spec** `{"name":..., "system_prompt":..., "model":..., "tools":[...]}` (không dùng object SubAgent(agent=...)).

## Ví dụ nâng cao (từ vendor)
- Multi-model + GPU skills: `vendor-deepagents/examples/nvidia_deep_agent/`
- Secure blueprint (Deep Agents + OpenShell): `vendor-nemoclaw/examples/harness-engineering-playground/`
