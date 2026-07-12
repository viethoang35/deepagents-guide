# Error: TypeError unexpected keyword argument 'sub_agents' / KeyError 'system_prompt'

**Cause**: two common mistakes when wiring up sub-agents:
1. Using `sub_agents` (with an underscore) — the real parameter on `create_deep_agent`
   is `subagents` (no underscore).
2. Passing a `SubAgent(agent=...)` object instead of a plain dict.

**Fix**: sub-agents must be a **dict spec**:
```python
subagents=[{
    "name": "researcher-agent",
    "description": "Delegate web research to this agent for any topic.",
    "system_prompt": "You are a web researcher.",
    "model": "openrouter:deepseek/deepseek-chat-v3",
    "tools": [web_search],
}]
```
