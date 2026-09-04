#!/bin/bash
# Script to evaluate Pred-EQA on Episode 0 using Gemini VLM (gemini-robotics-er-2-preview)

set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_DIR"

if [ -z "$GEMINI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
    echo "ERROR: Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set in your environment."
    echo "Please set your API key before running this script:"
    echo "  export GEMINI_API_KEY='your_api_key_here'"
    exit 1
fi

echo "=================================================================="
echo "Starting Pred-EQA Gemini Evaluation on Episode 0"
echo "Model: gemini-robotics-er-2-preview"
echo "Config: cfg/eval_pred_eqa_gemini.yaml"
echo "=================================================================="

# Activate pred-eqa conda environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/.bashrc
conda activate pred-eqa

# Run Episode 0 (start_ratio 0.0, end_ratio 0.0005 evaluates question 0)
python run_express_bench_evaluation_vlm_only.py \
    -cf cfg/eval_pred_eqa_gemini.yaml \
    --start_ratio 0.0 \
    --end_ratio 0.0005 \
    --vlm_provider gemini \
    --vlm_model gemini-robotics-er-2-preview

echo "Episode 0 evaluation finished!"
echo "Results saved under: results/Pred-EQA-Gemini/"
