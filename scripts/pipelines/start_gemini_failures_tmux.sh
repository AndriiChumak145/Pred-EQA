#!/bin/bash
# Wrapper to launch Gemini failure cases evaluation in a background tmux session

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

TOKEN_FILE="/home/dani/concept-scenesplat/tokens/gemini_api_key"
if [ -z "$GEMINI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
    if [ -f "$TOKEN_FILE" ]; then
        export GEMINI_API_KEY=$(cat "$TOKEN_FILE" | tr -d '\n\r')
    else
        echo "ERROR: Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set, and $TOKEN_FILE was not found."
        exit 1
    fi
fi

API_KEY="${GEMINI_API_KEY:-$GOOGLE_API_KEY}"

tmux kill-session -t pred_eqa_gemini_failures 2>/dev/null || true
tmux new-session -d -s pred_eqa_gemini_failures "cd $PROJECT_DIR && export GEMINI_API_KEY='$API_KEY' && chmod +x scripts/pipelines/run_gemini_failures.sh && ./scripts/pipelines/run_gemini_failures.sh > logs/pred_eqa_gemini_failures.log 2>&1"

echo "=================================================================="
echo "Successfully launched Pred-EQA Gemini Failures Evaluation!"
echo "Tmux session: pred_eqa_gemini_failures"
echo "Log file: logs/pred_eqa_gemini_failures.log"
echo "=================================================================="
echo "Commands to monitor:"
echo "  tmux attach -t pred_eqa_gemini_failures"
echo "  tail -f $PROJECT_DIR/logs/pred_eqa_gemini_failures.log"
