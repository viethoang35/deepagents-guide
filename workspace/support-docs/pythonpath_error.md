# Error: No module named 'pydantic_core._pydantic_core'

**Cause**: your shell has a `PYTHONPATH` environment variable inherited from another
virtualenv (for example a "Hermes" environment), which shadows this project's own venv.

**Fix**: always run scripts with `env -u PYTHONPATH` to unset it first:
```
env -u PYTHONPATH uv run --no-sync .venv/bin/python scripts/B2_quickstart.py
```
