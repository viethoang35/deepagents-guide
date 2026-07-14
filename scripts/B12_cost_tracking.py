"""
B12_cost_tracking.py — Cost & token tracking for a Deep Agent (Step 12).

Measures tokens (input/output/total) + estimates the COST ($) of a single agent
run, and attaches it to a trace (LangSmith if enabled, or local Phoenix). Helps
answer the production question: "how many tokens / how many $ does one agent
run cost? which model is cheaper?"

Cost estimation: a per-model price table (per 1M tokens). LLM pricing changes
often -> this is a guiding estimate, kept in COST_TABLE below, update as needed.

Run offline (no/placeholder key) -> verifies the cost-calculation logic, makes no API call:
  OPENROUTER_API_KEY=sk-or-fake env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B12_cost_tracking.py
Run for real (needs a valid OPENROUTER_API_KEY):
  env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B12_cost_tracking.py
  (with LANGCHAIN_TRACING_V2=true + LANGSMITH_API_KEY -> trace + cost go to LangSmith)
  (with Phoenix running locally -> the cost span goes to localhost:6006)
"""
import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

load_dotenv()

MODEL = os.getenv("DEEPAGENTS_MODEL", "openrouter:openai/gpt-4o-mini")

# Estimated pricing (USD per 1M tokens) — reference only, update when prices change.
# Key = the model string used with OpenRouter.
COST_TABLE = {
    "openrouter:openai/gpt-4o-mini":       {"in": 0.15, "out": 0.60},
    "openrouter:deepseek/deepseek-chat-v3": {"in": 0.27, "out": 1.10},
    "openrouter:x-ai/grok-4.5":             {"in": 2.00, "out": 10.00},
}


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    """Estimate $ from token counts using COST_TABLE."""
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
        # GenericFakeChatModel has no bind_tools -> Deep Agent calls it while
        # building -> error. Subclass to override bind_tools so offline
        # verification never makes an API call.
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
    """Pull token counts from the last message's usage_metadata."""
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
    print(f">>> B12 cost tracking | model={MODEL} | mode={'OFFLINE(fake)' if fake else 'LIVE'}")
    agent = build_agent(fake)

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Hanoi?"}]})
    except Exception as e:
        print(f">>> Agent call failed: {type(e).__name__}: {e}")
        print(">>> Check OPENROUTER_API_KEY in .env is valid.")
        return

    in_t, out_t, tot_t = extract_tokens(result)
    cost = estimate_cost(MODEL, in_t, out_t)
    print(f">>> tokens: in={in_t} out={out_t} total={tot_t}")
    print(f">>> estimated cost: ${cost:.6f}  (per COST_TABLE)")

    # Write to Phoenix if it's running locally (best-effort, doesn't fail the run if not)
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
        print(">>> Wrote a cost span to Phoenix (project 'deepagents-guide-cost').")
    except Exception as e:
        print(f">>> (info) Phoenix not available ({e}) — skipped writing the cost span.")


if __name__ == "__main__":
    main()
