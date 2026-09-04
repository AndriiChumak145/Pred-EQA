import os
import sys
from pathlib import Path
import glob
import json
import pickle
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_DIR = str(REPO_ROOT)
os.chdir(PROJECT_DIR)

# 1. Load data
with open("data/express-bench.json", "r") as f:
    gt_data = {str(item["question_id"]): item for item in json.load(f)}

with open("results/Pred-EQA/gpt_answer_0.0_0.09785.json", "r") as f:
    gpt_answers_list = json.load(f)
    gpt_answers = {str(item["question_id"]): item.get("answer", "") for item in gpt_answers_list}

with open("results/Pred-EQA/express_bench_info_0.0_0.09785.json", "r") as f:
    bench_info = json.load(f)

with open("results/Pred-EQA/success_list_0.0_0.09785.pkl", "rb") as f:
    success_list = pickle.load(f)

with open("results/Pred-EQA/path_length_list_0.0_0.09785.pkl", "rb") as f:
    path_lengths = pickle.load(f)

# Sort by question_id integer order
sorted_qids = sorted([str(k) for k in gpt_answers.keys()], key=lambda x: int(x))

compiled_rows = []
pkl_records = []

chosen_count = 0
none_count = 0

for idx, qid in enumerate(sorted_qids):
    gt = gt_data.get(qid, {})
    question = gt.get("question", "N/A")
    answer = gt.get("answer", "N/A")
    category = gt.get("category", "N/A")
    geodesic_dist = gt.get("geodesic_distance", np.nan)
    
    agent_ans = gpt_answers.get(qid, "")
    if agent_ans is None:
        agent_ans = ""
        
    info = bench_info.get(qid, {})
    goal_dis = info.get("goal_dis", np.nan)
    
    # Navigation success check
    is_success = (goal_dis <= 3.0) if not np.isnan(goal_dis) else False
    
    # Path length
    path_len = path_lengths.get(qid, np.nan)
    if np.isnan(path_len) and is_success:
        path_len = 0.0
        
    # Image resolution according to Pred-EQA evaluate_express_bench.py intent
    chosen_dir = f"results/Pred-EQA/{qid}/chosen_snapshot"
    img_path = None
    if os.path.exists(chosen_dir):
        imgs = sorted(glob.glob(os.path.join(chosen_dir, "*.png")))
        if imgs:
            img_path = os.path.abspath(imgs[0])
            chosen_count += 1

    if img_path is None:
        none_count += 1
        
    compiled_rows.append({
        "question_ind": int(qid),
        "category": category,
        "question": str(question),
        "gt_answer": str(answer),
        "agent_response": str(agent_ans),
        "nav_success": "SUCCESS" if is_success else "FAILED",
        "final_goal_dis_m": round(goal_dis, 2) if not np.isnan(goal_dis) else "N/A",
        "geodesic_dist_m": round(geodesic_dist, 2) if not np.isnan(geodesic_dist) else "N/A",
        "path_length_m": round(path_len, 2) if not np.isnan(path_len) else "N/A",
        "img_path": img_path
    })
    
    # Create EXPRESS-Bench compatible record
    pkl_records.append({
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
    })

print(f"Pred-EQA image selection: {chosen_count} using chosen_snapshot, {none_count} ungrounded (None)")

# Write Markdown summary
md_file = "results/pred_eqa_200_results_summary.md"
with open(md_file, "w") as f:
    f.write("# Pred-EQA 200-Episode Evaluation Results Summary\n\n")
    f.write(f"- **Total Episodes Processed**: {len(compiled_rows)}\n")
    f.write(f"- **Successful Navigation Episodes**: {sum(1 for r in compiled_rows if r['nav_success'] == 'SUCCESS')} / {len(compiled_rows)} ({sum(1 for r in compiled_rows if r['nav_success'] == 'SUCCESS')/len(compiled_rows)*100:.1f}%)\n")
    valid_lens = [r['path_length_m'] for r in compiled_rows if r['nav_success'] == 'SUCCESS' and isinstance(r['path_length_m'], (int, float))]
    f.write(f"- **Mean Path Length (Success)**: {np.mean(valid_lens):.2f} meters\n\n")
    f.write("---\n\n")
    f.write("| Q_Idx | Category | Question | Ground Truth Answer | Agent Response | Nav Status | Goal Dis (m) | Path Len (m) |\n")
    f.write("|:---:|:---:|:---|:---|:---|:---:|:---:|:---:|\n")
    for r in compiled_rows:
        q_text = r['question'].replace('|', '/')
        gt_text = r['gt_answer'].replace('|', '/')
        ag_text = r['agent_response'].replace('|', '/')
        f.write(f"| {r['question_ind']} | {r['category']} | {q_text} | {gt_text} | {ag_text} | **{r['nav_success']}** | {r['final_goal_dis_m']} | {r['path_length_m']} |\n")

print(f"Successfully generated Markdown summary: {md_file}")

# Write CSV file
csv_file = "results/pred_eqa_200_results_summary.csv"
df = pd.DataFrame(compiled_rows)
df.to_csv(csv_file, index=False)
print(f"Successfully generated CSV summary: {csv_file}")

# Write EXPRESS-Bench compatible PKL file
pkl_file = "results/pred_eqa_200_results.pkl"
with open(pkl_file, "wb") as f:
    pickle.dump(pkl_records, f)
print(f"Successfully generated EXPRESS-Bench compatible PKL: {pkl_file}")
