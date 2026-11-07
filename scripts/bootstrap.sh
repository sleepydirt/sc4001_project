#!/bin/bash
set -euo pipefail
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python constraints-linux-py312.txt
uv pip install --python .venv/bin/python -e . --no-deps
.venv/bin/python scripts/fetch_sources.py
