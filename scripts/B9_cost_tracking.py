"""
B9_cost_tracking.py — Cost & token tracking cho Deep Agent (Step 9, mở rộng tracing-labs).

Đo tokens (input/output/total) + ước tính CHI PHÍ ($) cho mỗi lần chạy agent,
gắn vào trace (LangSmith nếu bật, hoặc Phoenix local). Giúp trả lời câu hỏi
sản xuất: "1 lần agent chạy tốn bao nhiêu token / bao nhiêu $? model nào rẻ hơn?"

Cách ước tính cost: bảng giá từng model (token / 1M). Giá LLM thay đổi ->
chỉ là ước tính hướng dẫn, để trong COST_TABLE dưới, update khi cần.

Chạy OFFLINE (fake key) -> verify logic tính cost, KHÔNG gọi API:
  OPENROUTER_API_KEY=sk-or-fake env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B9_cost_tracking.py
Chạy THẬT (cần OPENROUTER_API_KEY):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B9_cost_tracking.py
  (nếu LANGCHAIN_TRACING_V2=true + LANGSMITH_API_KEY -> trace + cost lên LangSmith)
  (nếu Phoenix chạy local -> cost span lên localhost:6006)
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

load_dotenv()

MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")

# Bảng giá ước tính (USD per 1M tokens) — tham khảo, update khi giá thay đổi.
# Key = model string dùng trong OpenRouter.
COST_TABLE = {
    "openrouter:openai/gpt-4o-mini":       {"in": 0.15, "out": 0.60},
    "openrouter:deepseek/deepseek-chat-v3": {"in": 0.27, "out": 1.10},
    "openrouter:x-ai/grok-4.5":             {"in": 2.00, "out": 10.00},
}


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    """Ước tính $ từ token theo COST_TABLE."""
    rate = COST_TABLE.get(model, {"in": 1.0, "out": 2.0})  # default fallback
    return (in_tok / 1_000_000) * rate["in"] + (out_tok / 1_000_000) * rate["out"]


def has_openrouter_key() -> bool:
    k = os.getenv("OPENROUTER_API_KEY", "").strip()
    return k.startswith("sk-or-") and len(k) > 20


def build_agent(use_fake: bool):
    def get_weather(city: str) -> str:
        """Get current weather (mock)."""
        return f"It's sunny in {city}!"

    if use_fake:
        # GenericFakeChatModel không có bind_tools -> Deep Agent gọi khi build -> lỗi.
        # Subclass override bind_tools để offline verify không gọi API.
        class FakeModel(GenericFakeChatModel):
            def bind_tools(self, *a, **k):
                return self
        fake = FakeModel(messages=iter([AIMessage(content="Weather: sunny, 30C.", usage_metadata={
            "input_tokens": 120, "output_tokens": 25, "total_tokens": 145})]))
        return create_deep_agent(model=fake, tools=[get_weather],
                                 system_prompt="You are a weather assistant.")
    return create_deep_agent(model=MODEL, tools=[get_weather],
                             system_prompt="You are a weather assistant.")


def extract_tokens(result) -> tuple[int, int, int]:
    """Lấy token từ message cuối (usage_metadata)."""
    msgs = result.get("messages", [])
    last = msgs[-1] if msgs else None
    um = getattr(last, "usage_metadata", None)
    if um:
        if isinstance(um, dict):
            return (um.get("input_tokens", 0), um.get("output_tokens", 0), um.get("total_tokens", 0))
        return (getattr(um, "input_tokens", 0), getattr(um, "output_tokens", 0), getattr(um, "total_tokens", 0))
    return (0, 0, 0)


def main() -> None:
    fake = not has_openrouter_key()
    print(f">>> B9 cost tracking | model={MODEL} | mode={'OFFLINE(fake)' if fake else 'THẬT'}")
    agent = build_agent(fake)

    result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Hanoi?"}]})
    in_t, out_t, tot_t = extract_tokens(result)
    cost = estimate_cost(MODEL, in_t, out_t)
    print(f">>> tokens: in={in_t} out={out_t} total={tot_t}")
    print(f">>> estimated cost: ${cost:.6f}  (theo COST_TABLE)")

    # Ghi vào Phoenix nếu chạy local (best-effort, không fail nếu không có)
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
        from phoenix.otel import register
        provider = register(project_name="deepagents-guide-cost", endpoint="http://localhost:6006/v1/traces")
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("cost_run") as span:
            span.set_attribute("cost.model", MODEL)
            span.set_attribute("cost.input_tokens", in_t)
            span.set_attribute("cost.output_tokens", out_t)
            span.set_attribute("cost.total_tokens", tot_t)
            span.set_attribute("cost.est_usd", cost)
        print(">>> Đã ghi cost span vào Phoenix (project 'deepagents-guide-cost').")
    except Exception as e:
        print(f">>> (info) Phoenix không khả dụng ({e}) — bỏ qua ghi cost span.")


if __name__ == "__main__":
    main()
