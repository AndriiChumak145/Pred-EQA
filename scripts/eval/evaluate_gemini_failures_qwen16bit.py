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

qids = ["1005", "1006", "103"]

records = []
for qid in qids:
    gt = gt_data.get(qid, {})
    question = gt.get("question", "")
    answer = gt.get("answer", "")
    category = gt.get("category", "")
    geodesic_dist = gt.get("geodesic_distance", np.nan)

    # Gemini answer
    gem_ans = ""
    for f in glob.glob("results/Pred-EQA-Gemini/gpt_answer_*.json"):
        data = json.load(open(f))
        for item in data:
            if str(item.get("question_id")) == qid:
                gem_ans = item.get("answer", "")
                break

    # Gemini info
    gem_info = {}
    for f in glob.glob("results/Pred-EQA-Gemini/express_bench_info_*.json"):
        data = json.load(open(f))
        if qid in data:
            gem_info = data[qid]
            break

    goal_dis = gem_info.get("goal_dis", np.nan)
    is_success = (goal_dis <= 3.0) if not np.isnan(goal_dis) else False

    gem_traj_path = f"results/Pred-EQA-Gemini/{qid}/trajectory.json"
    gem_traj = json.load(open(gem_traj_path)) if os.path.exists(gem_traj_path) else {}
    path_len = gem_traj.get("path_length", np.nan)

    chosen_dir = f"results/Pred-EQA-Gemini/{qid}/chosen_snapshot"
    img_path = None
    if os.path.exists(chosen_dir):
        imgs = sorted(glob.glob(os.path.join(chosen_dir, "*.png")))
        if imgs:
            img_path = os.path.abspath(imgs[0])

    record = {
        "question_ind": int(qid),
        "question": str(question),
        "answer": str(answer),
        "gen_answer": str(gem_ans),
        "category": str(category),
        "geodesic_distance": geodesic_dist,
        "goal_dis": goal_dis,
        "nav_success": is_success,
        "path_length": path_len,
        "cnt_step": 0,
        "img_path": img_path
    }
    records.append(record)

input_pkl = os.path.join(PROJECT_DIR, "results/Pred-EQA-Gemini/gemini_failures_results.pkl")
output_scores_pkl = os.path.join(PROJECT_DIR, "results/Pred-EQA-Gemini/gemini_failures_qwen8b_16bit_scores.pkl")

with open(input_pkl, "wb") as f:
    pickle.dump(records, f)

env = os.environ.copy()
env["USE_LOCAL_QWEN"] = "1"
env["QWEN_QUANTIZATION_OVERRIDE"] = "0"
env["QWEN_THINKING"] = "0"

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

print("\nOfficial 10-Run Judge Scores for Gemini Output:")
for qid_int in [1005, 1006, 103]:
    s_list = scores.get(qid_int, [])
    print(f"  Question {qid_int}: {s_list[0] if s_list else 'N/A'}")
