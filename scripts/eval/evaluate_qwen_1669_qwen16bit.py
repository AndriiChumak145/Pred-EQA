import os
import sys
from pathlib import Path
import glob
import json
import pickle
import numpy as np
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_DIR = str(REPO_ROOT)
os.chdir(PROJECT_DIR)

# 1. Load ground truth questions
with open("data/express-bench.json", "r") as f:
    gt_data = {str(item["question_id"]): item for item in json.load(f)}

# 2. Load Pred-EQA Qwen 1669 outputs
res_dir = "results/Pred-EQA"
with open(os.path.join(res_dir, "gpt_answer_0.3644815_0.3649707.json"), "r") as f:
    gpt_answers_list = json.load(f)
    gpt_answers = {str(item["question_id"]): item.get("answer", "") for item in gpt_answers_list}

with open(os.path.join(res_dir, "express_bench_info_0.3644815_0.3649707.json"), "r") as f:
    bench_info = json.load(f)

with open(os.path.join(res_dir, "path_length_list_0.3644815_0.3649707.pkl"), "rb") as f:
    path_lengths = pickle.load(f)

qid = "1669"
gt = gt_data.get(qid, {})
question = gt.get("question", "")
answer = gt.get("answer", "")
category = gt.get("category", "")
geodesic_dist = gt.get("geodesic_distance", np.nan)

agent_ans = gpt_answers.get(qid, "")
info = bench_info.get(qid, {})
goal_dis = info.get("goal_dis", np.nan)
is_success = (goal_dis <= 3.0) if not np.isnan(goal_dis) else False
path_len = path_lengths.get(qid, np.nan)

chosen_dir = f"{res_dir}/{qid}/chosen_snapshot"
img_path = None
if os.path.exists(chosen_dir):
    imgs = sorted(glob.glob(os.path.join(chosen_dir, "*.png")))
    if imgs:
        img_path = os.path.abspath(imgs[0])

record = {
    "question_ind": int(qid),
    "question": str(question),
    "answer": str(answer),
    "gen_answer": str(agent_ans),
    "category": str(category),
    "geodesic_distance": geodesic_dist,
    "goal_dis": goal_dis,
    "nav_success": is_success,
    "path_length": path_len,
    "cnt_step": 0,
    "img_path": img_path
}

input_pkl = os.path.join(PROJECT_DIR, "results/Pred-EQA/qwen_1669_results.pkl")
output_scores_pkl = os.path.join(PROJECT_DIR, "results/Pred-EQA/qwen_1669_qwen8b_16bit_scores.pkl")

with open(input_pkl, "wb") as f:
    pickle.dump([record], f)

env = os.environ.copy()
env["USE_LOCAL_QWEN"] = "1"
env["QWEN_QUANTIZATION_OVERRIDE"] = "0"
env["QWEN_THINKING"] = "0"
env["USE_LOCAL_GEMMA"] = "0"
env["USE_LOCAL_VLLM"] = "0"

cmd = [
    "python3",
    "/home/dani/concept-scenesplat/EXPRESS-Bench/eval_all_2044.py",
    "--input_file", input_pkl,
    "--output_file", output_scores_pkl,
    "--episodes_per_batch", "10",
    "--num_evals_per_episode", "10",
    "--qwen_model", "Qwen/Qwen3-VL-8B-Instruct"
]

subprocess.run(cmd, env=env, cwd="/home/dani/concept-scenesplat/EXPRESS-Bench", check=True)

with open(output_scores_pkl, "rb") as f:
    scores = pickle.load(f)

print(f"\nEvaluated Scores for Question 1669 ({len(scores.get(1669, []))} runs):")
for idx, s in enumerate(scores.get(1669, [])):
    print(f"  Run {idx+1}: {s}")
