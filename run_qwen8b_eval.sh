#!/bin/bash
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate qwen36
export LD_LIBRARY_PATH=/home/dani/miniconda3/envs/qwen36/lib:$LD_LIBRARY_PATH

export USE_LOCAL_QWEN=1
export QWEN_QUANTIZATION_OVERRIDE=0
export QWEN_THINKING=0
export USE_LOCAL_GEMMA=0
export USE_LOCAL_VLLM=0

cd /home/dani/concept-scenesplat/EXPRESS-Bench
python3 eval_all_2044.py \
    --input_file /home/dani/concept-scenesplat/Pred-EQA/results/pred_eqa_200_results.pkl \
    --output_file /home/dani/concept-scenesplat/Pred-EQA/results/pred_eqa_200_qwen8b_scores.pkl \
    --episodes_per_batch 250 \
    --num_evals_per_episode 10 \
    --qwen_model "Qwen/Qwen3-VL-8B-Instruct"
