#!/usr/bin/env bash
# =============================================================================
# Machine C (Local) — Push code and pull results
#
# This script runs on the local development machine. It:
#   1. Commits and pushes code changes to GitHub
#   2. Polls for results (commits pushed by Machine B from Machine A)
#
# Usage:
#   ./sync_c.sh push [msg]    # commit & push current changes
#   ./sync_c.sh pull          # pull latest results from GitHub
#   ./sync_c.sh wait [SEC]    # poll until new results arrive (default 30s)
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/tongyu/workspace/ai4ai/autoresearch_kernel}"
TASK_DIR="001_attention_softmax_dropout_value_matmul_backward"
BRANCH="main"

cd "$REPO_DIR"

do_push() {
    local msg="${1:-update solution}"
    echo "[C][$(date '+%H:%M:%S')] Staging and pushing..."

    cd "$REPO_DIR/$TASK_DIR"
    git add solution.json round_*/solution.json *.cpp *.cu *.h *.cuh 2>/dev/null || true
    cd "$REPO_DIR"

    if git diff --cached --quiet; then
        echo "[C][$(date '+%H:%M:%S')] Nothing to commit"
    else
        git commit -m "$msg"
        echo "[C][$(date '+%H:%M:%S')] Committed: $(git log --oneline -1)"
    fi

    git push origin "$BRANCH"
    echo "[C][$(date '+%H:%M:%S')] Pushed $(git rev-parse --short HEAD)"
}

do_pull() {
    echo "[C][$(date '+%H:%M:%S')] Pulling results from GitHub..."
    local before=$(git rev-parse HEAD)
    git pull --ff-only origin "$BRANCH" 2>/dev/null

    if [ "$before" != "$(git rev-parse HEAD)" ]; then
        echo "[C][$(date '+%H:%M:%S')] New results:"
        git log --oneline "${before}..HEAD"

        # Show latest log if available
        local latest_log=$(find "$REPO_DIR/$TASK_DIR" -name "eval.log" -newer "$REPO_DIR/.git/FETCH_HEAD" 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            echo ""
            echo "=== Latest eval log ==="
            cat "$latest_log"
        fi
    else
        echo "[C][$(date '+%H:%M:%S')] No new results yet"
    fi
}

do_wait() {
    local interval="${1:-30}"
    local start_commit=$(git rev-parse HEAD)

    echo "[C] Waiting for results (poll every ${interval}s)..."
    echo "[C] Current commit: $(git rev-parse --short HEAD)"
    echo "[C] Press Ctrl+C to stop"

    while true; do
        git fetch origin "$BRANCH" 2>/dev/null
        local remote_head=$(git rev-parse "origin/$BRANCH")

        if [ "$start_commit" != "$remote_head" ]; then
            echo ""
            echo "[C][$(date '+%H:%M:%S')] New results arrived!"
            git merge --ff-only "origin/$BRANCH"
            git log --oneline "${start_commit}..HEAD"

            # Show eval log
            local latest_log=$(find "$REPO_DIR/$TASK_DIR" -name "eval.log" -newer "$REPO_DIR/.git/FETCH_HEAD" 2>/dev/null | sort -r | head -1)
            if [ -n "$latest_log" ]; then
                echo ""
                echo "=== Eval log: $latest_log ==="
                cat "$latest_log"
            fi
            break
        fi

        echo -n "."
        sleep "$interval"
    done
}

# --- Main ---
case "${1:-help}" in
    push)
        shift
        do_push "${*:-update solution}"
        ;;
    pull)
        do_pull
        ;;
    wait)
        do_wait "${2:-30}"
        ;;
    *)
        echo "Usage:"
        echo "  $0 push [commit message]  — commit & push code to GitHub"
        echo "  $0 pull                   — pull latest results"
        echo "  $0 wait [interval_sec]    — poll until results arrive"
        ;;
esac
