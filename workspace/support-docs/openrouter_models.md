# How do I use a different model via OpenRouter?

Deep Agents supports the `openrouter:` prefix. Set `OPENROUTER_API_KEY` in `.env`, then
pick any model with the format `openrouter:<provider>/<model>`, for example:
```
DEEPAGENTS_MODEL=openrouter:deepseek/deepseek-chat-v3
DEEPAGENTS_MODEL=openrouter:x-ai/grok-4.5
DEEPAGENTS_MODEL=openrouter:openai/gpt-4o-mini
```
One key lets you call many different providers/models without separate accounts for each.
Register a key at https://openrouter.ai/keys.
