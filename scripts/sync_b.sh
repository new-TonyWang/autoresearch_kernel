#!/usr/bin/env bash
# =============================================================================
# Machine B (Sync Node) — Pull/Push loop
#
# This script runs on Machine B which has internet access and shares storage
# with Machine A. It does two things:
#   1. Pulls latest code from GitHub (pushed by Machine C)
#   2. Pushes any new local commits back to GitHub (committed by Machine A)
#
# Usage:
#   ./sync_b.sh                  # one-shot: pull then push
#   ./sync_b.sh --loop [SEC]     # continuous loop (default 30s interval)
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/tongyu/workspace/ai4ai/autoresearch_kernel}"
POLL_INTERVAL="${2:-30}"   # seconds between polls in loop mode
BRANCH="main"

cd "$REPO_DIR"

pull_from_github() {
    echo "[B][$(date '+%H:%M:%S')] Pulling from GitHub..."
    local before=$(git rev-parse HEAD)
    git fetch origin "$BRANCH" 2>/dev/null
    local remote_head=$(git rev-parse "origin/$BRANCH")

    if [ "$before" != "$remote_head" ]; then
        git merge --ff-only "origin/$BRANCH"
        echo "[B][$(date '+%H:%M:%S')] Updated: $(git log --oneline ${before}..HEAD | head -5)"
    else
        echo "[B][$(date '+%H:%M:%S')] Already up to date ($(git rev-parse --short HEAD))"
    fi
}

push_to_github() {
    # Check if there are local commits not yet on remote
    local local_head=$(git rev-parse HEAD)
    local remote_head=$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo "none")

    if [ "$local_head" != "$remote_head" ]; then
        echo "[B][$(date '+%H:%M:%S')] Pushing results to GitHub..."
        git push origin "$BRANCH"
        echo "[B][$(date '+%H:%M:%S')] Pushed $(git rev-parse --short HEAD)"
    else
        echo "[B][$(date '+%H:%M:%S')] Nothing new to push"
    fi
}

# --- Main ---
if [ "${1:-}" = "--loop" ]; then
    echo "[B] Starting sync loop (interval: ${POLL_INTERVAL}s) in $REPO_DIR"
    echo "[B] Press Ctrl+C to stop"
    while true; do
        echo ""
        echo "========== Sync cycle at $(date) =========="
        pull_from_github
        push_to_github
        sleep "$POLL_INTERVAL"
    done
else
    pull_from_github
    push_to_github
fi
