#!/usr/bin/env bash
set -euo pipefail

# One-command sync: pull latest remote changes, commit local updates, then push.
# Usage:
#   ./scripts/git_sync.sh "your commit message"
#   ./scripts/git_sync.sh
#   ./scripts/git_sync.sh --help

usage() {
  cat <<'EOF'
Usage: ./scripts/git_sync.sh [commit_message]

Pull latest remote changes (rebase), stage all local changes, commit, and push.
If commit_message is omitted, a timestamped message is used.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: run this script inside a Git repository."
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" == "HEAD" ]]; then
  echo "Error: detached HEAD. Checkout a branch first."
  exit 1
fi

if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  git pull --rebase --autostash
fi

git add -A
if git diff --cached --quiet; then
  echo "No local changes to commit."
  exit 0
fi

DEFAULT_MESSAGE="sync: $(date '+%Y-%m-%d %H:%M:%S %Z')"
COMMIT_MESSAGE="${1:-$DEFAULT_MESSAGE}"

git commit -m "$COMMIT_MESSAGE"

if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  git push
else
  git push -u origin "$BRANCH"
fi

echo "Sync complete on branch '$BRANCH'."
