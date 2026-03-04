#!/usr/bin/env bash
set -euo pipefail

# Clone (first run) or pull (next runs) inside a Kubeflow notebook environment.
# Usage:
#   bash scripts/kubeflow_sync.sh <repo_url> [branch] [target_dir]
#
# Example:
#   bash scripts/kubeflow_sync.sh https://github.com/rym173/euh-fyp.git clean-main /home/jovyan/work/EoH-modified

REPO_URL="${1:-}"
BRANCH="${2:-clean-main}"
TARGET_DIR="${3:-/home/jovyan/work/EoH-modified}"

if [[ -z "$REPO_URL" ]]; then
  echo "Error: repo URL is required."
  echo "Usage: bash scripts/kubeflow_sync.sh <repo_url> [branch] [target_dir]"
  exit 1
fi

if [[ -d "$TARGET_DIR/.git" ]]; then
  echo "Repository exists at $TARGET_DIR. Pulling latest '$BRANCH'..."
  git -C "$TARGET_DIR" fetch origin
  git -C "$TARGET_DIR" checkout "$BRANCH"
  git -C "$TARGET_DIR" pull --rebase origin "$BRANCH"
else
  echo "Cloning $REPO_URL into $TARGET_DIR..."
  git clone -b "$BRANCH" "$REPO_URL" "$TARGET_DIR"
fi

if [[ -f "$TARGET_DIR/eoh/setup.py" || -f "$TARGET_DIR/eoh/pyproject.toml" ]]; then
  echo "Installing project package in editable mode..."
  python -m pip install -e "$TARGET_DIR/eoh"
fi

echo "Kubeflow sync complete."
echo "Run from notebook:"
echo "  cd $TARGET_DIR"
echo "  python examples/bp_online.py"
