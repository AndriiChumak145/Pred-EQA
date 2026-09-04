#!/bin/bash
# start_pred_eqa_pipeline.sh
# Starts Qwen VLM server and Pred-EQA evaluation run in background tmux sessions.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 1
mkdir -p "$PROJECT_DIR/logs"

echo "=================================================="
echo " Starting Pred-EQA Overnight Pipeline"
echo "=================================================="

# 1. Start Qwen VLM Server if not running
if tmux has-session -t qwen_server 2>/dev/null; then
    echo "✔ tmux session 'qwen_server' is already running."
else
    echo "➜ Starting Qwen VLM server in tmux session 'qwen_server'..."
    tmux new-session -d -s qwen_server "source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen36 && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib && cd $PROJECT_DIR && python scripts/servers/llm_native_server.py --model Qwen/Qwen3-VL-8B-Instruct > logs/qwen_server.log 2>&1"
fi

# 2. Wait for Qwen VLM Server port 8000 to be ready (checking HTTP 200 OK)
echo "⏳ Waiting for Qwen server on http://127.0.0.1:8000/health to respond..."
max_retries=45
retries=0
until curl -s -f http://127.0.0.1:8000/health > /dev/null 2>&1 || curl -s -f http://127.0.0.1:8000/v1/models > /dev/null 2>&1; do
    retries=$((retries + 1))
    if [ $retries -ge $max_retries ]; then
        echo "❌ Server port 8000 failed to respond after $max_retries attempts."
        echo "Check logs/qwen_server.log for details:"
        tail -n 20 "$PROJECT_DIR/logs/qwen_server.log"
        exit 1
    fi
    sleep 2
    echo -n "."
done
echo ""
echo "✔ Qwen server is UP and ready on port 8000!"

# 3. Stabilization Delay (staggering launch to prevent VRAM memory allocation conflicts)
echo "⏳ Staggering launch: pausing 10 seconds for server VRAM to settle..."
sleep 10

# 4. Start Pred-EQA Evaluation run if not running
if tmux has-session -t pred_eqa_eval 2>/dev/null; then
    echo "✔ tmux session 'pred_eqa_eval' is already running."
else
    echo "➜ Starting 200-episode evaluation in tmux session 'pred_eqa_eval'..."
    tmux new-session -d -s pred_eqa_eval "source ~/miniconda3/etc/profile.d/conda.sh && conda activate pred-eqa && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib && cd $PROJECT_DIR && python run_express_bench_evaluation_vlm_only.py -cf cfg/eval_pred_eqa.yaml --start_ratio 0 --end_ratio 0.09785 >> logs/pred_eqa_200_episodes.log 2>&1"
fi

# 5. Post-launch Verification
sleep 3
if ! tmux has-session -t pred_eqa_eval 2>/dev/null; then
    echo "❌ FATAL: Evaluation session 'pred_eqa_eval' crashed immediately after start!"
    echo "Check logs/pred_eqa_200_episodes.log for details:"
    tail -n 25 "$PROJECT_DIR/logs/pred_eqa_200_episodes.log"
    exit 1
fi

echo "=================================================="
echo "Pipeline started successfully and verified running!"
echo ""
echo "Monitoring Commands:"
echo "  • View evaluation log:  tail -f $PROJECT_DIR/logs/pred_eqa_200_episodes.log"
echo "  • View server log:      tail -f $PROJECT_DIR/logs/qwen_server.log"
echo "  • Attach to eval session: tmux attach -t pred_eqa_eval"
echo "  • List tmux sessions:   tmux ls"
echo "=================================================="

