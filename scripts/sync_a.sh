#!/usr/bin/env bash
# =============================================================================
# Machine A (Compute Node) — Watch, compile, run, commit
#
# This script runs on Machine A which has the GPU but no GitHub access.
# It shares storage with Machine B. It:
#   1. Watches for new commits (code changes pulled by Machine B)
#   2. Reads solution.json, compiles and evaluates the kernel on GPU
#   3. Writes logs to the round directory
#   4. Commits results (Machine B will push them to GitHub)
#
# Usage:
#   ./sync_a.sh                  # one-shot: run latest solution
#   ./sync_a.sh --loop [SEC]     # continuous watch loop (default 30s interval)
#
# Environment variables:
#   REPO_DIR     — path to the repo (default: /home/tongyu/workspace/ai4ai/autoresearch_kernel)
#   TASK_DIR     — task subdirectory name
#   EVAL_CMD     — evaluation command (receives solution.json path as $1)
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/tongyu/workspace/ai4ai/autoresearch_kernel}"
TASK_DIR="${TASK_DIR:-001_attention_softmax_dropout_value_matmul_backward}"
POLL_INTERVAL="${2:-30}"
BRANCH="main"

WORK_DIR="$REPO_DIR/$TASK_DIR"
LAST_COMMIT_FILE="$WORK_DIR/.last_processed_commit"

cd "$REPO_DIR"

get_latest_commit() {
    git rev-parse HEAD
}

get_last_processed() {
    if [ -f "$LAST_COMMIT_FILE" ]; then
        cat "$LAST_COMMIT_FILE"
    else
        echo "none"
    fi
}

run_evaluation() {
    local solution_file="$WORK_DIR/solution.json"

    if [ ! -f "$solution_file" ]; then
        echo "[A][$(date '+%H:%M:%S')] ERROR: solution.json not found at $solution_file"
        return 1
    fi

    # Extract round_dir from solution.json
    local round_dir
    round_dir=$(python3 -c "import json; print(json.load(open('$solution_file'))['round_dir'])" 2>/dev/null || echo "round_0")
    local round_path="$WORK_DIR/$round_dir"
    mkdir -p "$round_path"

    local log_file="$round_path/eval.log"
    local commit_hash=$(git rev-parse --short HEAD)

    echo "[A][$(date '+%H:%M:%S')] Running evaluation for commit $commit_hash, round: $round_dir"
    echo "[A][$(date '+%H:%M:%S')] Log: $log_file"

    # Copy solution.json to round dir for record
    cp "$solution_file" "$round_path/solution.json"

    # --- Run evaluation ---
    # Replace this with your actual evaluation command.
    # The EVAL_CMD should accept the solution.json path and the task definition.
    # Example: python evaluate.py --solution solution.json --definition definition.json
    if [ -n "${EVAL_CMD:-}" ]; then
        echo "[A][$(date '+%H:%M:%S')] Running: $EVAL_CMD $solution_file"
        eval "$EVAL_CMD \"$solution_file\"" 2>&1 | tee "$log_file"
        local exit_code=${PIPESTATUS[0]}
    else
        echo "[A][$(date '+%H:%M:%S')] WARNING: EVAL_CMD not set. Skipping actual evaluation."
        echo "[A][$(date '+%H:%M:%S')] Set EVAL_CMD to your evaluation script/command."
        echo "EVAL_CMD not set — skipped evaluation" > "$log_file"
        local exit_code=0
    fi

    echo "[A][$(date '+%H:%M:%S')] Evaluation finished (exit code: $exit_code)"

    # --- Commit results ---
    cd "$REPO_DIR"
    git add "$round_path/"
    if git diff --cached --quiet; then
        echo "[A][$(date '+%H:%M:%S')] No new results to commit"
    else
        git commit -m "$(cat <<EOF
[A] eval results for $round_dir (commit $commit_hash)

Exit code: $exit_code
EOF
        echo "[A][$(date '+%H:%M:%S')] Committed results for $round_dir"
    fi

    # Mark this commit as processed
    echo "$(get_latest_commit)" > "$LAST_COMMIT_FILE"

    return $exit_code
}

check_and_run() {
    local current=$(get_latest_commit)
    local last=$(get_last_processed)

    if [ "$current" = "$last" ]; then
        echo "[A][$(date '+%H:%M:%S')] No new commits (at $(git rev-parse --short HEAD))"
        return 0
    fi

    echo "[A][$(date '+%H:%M:%S')] New commit detected: $(git log --oneline -1)"
    run_evaluation
}

# --- Main ---
if [ "${1:-}" = "--loop" ]; then
    echo "[A] Starting watch loop (interval: ${POLL_INTERVAL}s) in $WORK_DIR"
    echo "[A] Press Ctrl+C to stop"
    while true; do
        echo ""
        echo "========== Check cycle at $(date) =========="
        check_and_run || true
        sleep "$POLL_INTERVAL"
    done
else
    run_evaluation
fi
