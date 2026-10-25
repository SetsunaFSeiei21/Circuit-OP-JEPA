#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel)"
# Refuse accidental publication from an enclosing unrelated repository.
[[ "$PROJECT_ROOT" == "$REPO_ROOT" ]] || { echo 'Project must be its own Git checkout before publishing' >&2; exit 1; }
BRANCH="$(git -C "$REPO_ROOT" symbolic-ref --quiet --short HEAD)" || { echo 'Detached HEAD; switch to intended branch first' >&2; exit 1; }
REMOTE="$(git -C "$REPO_ROOT" remote get-url origin)"
case "$REMOTE" in
  https://github.com/SetsunaFSeiei21/Circuit-OP-JEPA|https://github.com/SetsunaFSeiei21/Circuit-OP-JEPA.git|git@github.com:SetsunaFSeiei21/Circuit-OP-JEPA.git) ;;
  *) echo 'Refusing unverified origin; inspect and adapt allowlist for an intentional fork' >&2; exit 1 ;;
esac
# Stage files explicitly beforehand. Never git add -A, reset, force-push, or pull.
printf 'Target: %s (%s)\n' "$REMOTE" "$BRANCH"
git -C "$REPO_ROOT" diff --cached --stat
if [[ "${PUSH_CONFIRM:-0}" != 1 ]]; then
    echo 'Dry run. Review staged changes, then set PUSH_CONFIRM=1 to commit/push.'
    exit 0
fi
if ! git -C "$REPO_ROOT" diff --cached --quiet; then
    if (( $# > 0 )); then git -C "$REPO_ROOT" commit -m "$*";
    else git -C "$REPO_ROOT" commit -F "$SCRIPT_DIR/commit_message.txt"; fi
fi
git -C "$REPO_ROOT" push -u origin "$BRANCH"
