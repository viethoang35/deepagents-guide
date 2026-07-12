# Error: ModuleNotFoundError: langchain_openai (or langchain_anthropic, langchain_google_genai)

**Cause**: `create_deep_agent` needs the LangChain integration package for whichever model
provider you selected, and it wasn't installed.

**Fix**: the provider packages are already listed in `pyproject.toml`
(`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-openrouter`).
Run `uv sync` to install them. If you add a brand-new provider, add its package to
`pyproject.toml` first.
