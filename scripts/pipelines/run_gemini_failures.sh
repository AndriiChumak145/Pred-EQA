#!/bin/bash
# Script to evaluate Pred-EQA on the 3 failure cases (1005, 1006, 103) using Gemini (gemini-robotics-er-2-preview)

set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

TOKEN_FILE="${TOKEN_FILE:-$WORKSPACE_ROOT/tokens/gemini_api_key}"
if [ -z "$GEMINI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
    if [ -f "$TOKEN_FILE" ]; then
        export GEMINI_API_KEY=$(cat "$TOKEN_FILE" | tr -d '\n\r')
    else
        echo "ERROR: Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set, and $TOKEN_FILE was not found."
        exit 1
    fi
fi

source ~/miniconda3/etc/profile.d/conda.sh || source ~/.bashrc
conda activate pred-eqa

echo "=================================================================="
echo "Starting Pred-EQA Gemini Evaluation on 3 Failure Cases (1005, 1006, 103)"
echo "Model: gemini-robotics-er-2-preview"
echo "=================================================================="

# 1. Question 1005 (Is the wall in the bathroom still clean after the last cleaning?)
echo "\n--- Evaluating Question ID 1005 ---"
python run_express_bench_evaluation_vlm_only.py \
    -cf cfg/eval_pred_eqa_gemini.yaml \
    --start_ratio 0.0044032 \
    --end_ratio 0.0048924 \
    --vlm_provider gemini \
    --vlm_model gemini-robotics-er-2-preview

# 2. Question 1006 (What type of material is the wall made of in the bathroom?)
echo "\n--- Evaluating Question ID 1006 ---"
python run_express_bench_evaluation_vlm_only.py \
    -cf cfg/eval_pred_eqa_gemini.yaml \
    --start_ratio 0.0048925 \
    --end_ratio 0.0053816 \
    --vlm_provider gemini \
    --vlm_model gemini-robotics-er-2-preview

# 3. Question 103 (What is the small round object above the mail rack?)
echo "\n--- Evaluating Question ID 103 ---"
python run_express_bench_evaluation_vlm_only.py \
    -cf cfg/eval_pred_eqa_gemini.yaml \
    --start_ratio 0.0176126 \
    --end_ratio 0.0181018 \
    --vlm_provider gemini \
    --vlm_model gemini-robotics-er-2-preview

echo "\nAll 3 failure cases completed for Gemini!"
echo "Results saved under: results/Pred-EQA-Gemini/"
