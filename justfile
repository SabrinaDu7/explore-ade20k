default: fmt lint test

# Format code
fmt:
    uv run ruff format .

# Lint code
lint:
    uv run ruff check .
    uv run basedpyright

# Run tests
test:
    uv run pytest

# Sync notebooks (.py -> .ipynb)
sync:
    uv run jupytext --sync notebooks/*.py

# Install dev dependencies
install:
    uv sync --group dev
