#!/usr/bin/env bash
# Install the repo's git hooks (works in worktrees too) and the tools they need.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
hooks="$(git rev-parse --git-path hooks)"
mkdir -p "$hooks"
ln -sf "$root/scripts/hooks/pre-commit" "$hooks/pre-commit"
chmod +x "$root/scripts/hooks/pre-commit"
if [ -f "$root/.venv/bin/activate" ]; then
  source "$root/.venv/bin/activate"
  command -v ruff >/dev/null || uv pip install -q ruff
  python -c "import xdist" 2>/dev/null || uv pip install -q pytest-xdist
fi
echo "installed pre-commit -> $hooks/pre-commit"
