import os
import sys
from pathlib import Path
import re
import json
import pickle
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_DIR = str(REPO_ROOT)
os.chdir(PROJECT_DIR)

# 1. Load results and scores
with open("results/pred_eqa_200_results.pkl", "rb") as f:
    records = pickle.load(f)

scores_file = "results/pred_eqa_200_qwen8b_scores.pkl"
if not os.path.exists(scores_file):
    print(f"Error: {scores_file} not found yet!")
    exit(1)

with open(scores_file, "rb") as f:
    scores_dict = pickle.load(f)

def parse_mark_string(mark_str):
    if not mark_str or not isinstance(mark_str, str):
        return 0.0, 1.0
    
    # Try regex match for "Your mark: G, A" or "G, A"
    match = re.search(r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', mark_str)
    if match:
        try:
            g = float(match.group(1))
            a = float(match.group(2))
            # Validate ranges
            g = max(0.0, min(1.0, g))
            a = max(1.0, min(5.0, a))
            return g, a
        except ValueError:
            pass
    return 0.0, 1.0

compiled_rows = []
for r in records:
    qid = r["question_ind"]
    marks_list = scores_dict.get(qid, [])
    
    if marks_list:
        raw_mark = marks_list[0]
        grounding, accuracy = parse_mark_string(raw_mark)
    else:
        raw_mark = "N/A"
        grounding, accuracy = 0.0, 1.0
        
    grounded_acc = grounding * (accuracy / 5.0) * 100.0
    
    nav_status = "SUCCESS" if r["nav_success"] else "FAILED"
    goal_dis = round(r["goal_dis"], 2) if not np.isnan(r["goal_dis"]) else "N/A"
    path_len = round(r["path_length"], 2) if not np.isnan(r["path_length"]) else "N/A"
    
    compiled_rows.append({
        "question_ind": qid,
        "category": r["category"],
        "question": r["question"],
        "gt_answer": r["answer"],
        "agent_response": r["gen_answer"],
        "nav_success": nav_status,
        "goal_dis_m": goal_dis,
        "path_length_m": path_len,
        "grounding_score": grounding,
        "accuracy_score": accuracy,
        "grounded_acc_pct": round(grounded_acc, 1),
        "raw_judge_mark": str(raw_mark).strip()
    })

# Compute aggregate metrics
mean_grounding = np.mean([r["grounding_score"] for r in compiled_rows])
mean_accuracy = np.mean([r["accuracy_score"] for r in compiled_rows])
mean_c_pct = np.mean([r["grounded_acc_pct"] for r in compiled_rows])
succ_count = sum(1 for r in compiled_rows if r["nav_success"] == "SUCCESS")

# 2. Update Markdown file
md_file = "results/pred_eqa_200_results_summary.md"
with open(md_file, "w") as f:
    f.write("# Pred-EQA 200-Episode Evaluation Results (Qwen 8B Instruct Judge)\n\n")
    f.write(f"- **Total Episodes Processed**: {len(compiled_rows)}\n")
    f.write(f"- **Navigation Success Rate**: {succ_count} / {len(compiled_rows)} ({succ_count/len(compiled_rows)*100:.1f}%)\n")
    f.write(f"- **Mean Grounding Score**: **{mean_grounding:.2f}** / 1.0\n")
    f.write(f"- **Mean Accuracy Score**: **{mean_accuracy:.2f}** / 5.0\n")
    f.write(f"- **Mean Grounded Accuracy Metric (C%)**: **{mean_c_pct:.1f}%**\n\n")
    f.write("---\n\n")
    f.write("| Q_Idx | Category | Question | Ground Truth Answer | Agent Response | Nav Status | Grounding | Accuracy | Grounded Acc (C%) |\n")
    f.write("|:---:|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|\n")
    for r in compiled_rows:
        q_text = r['question'].replace('|', '/')
        gt_text = r['gt_answer'].replace('|', '/')
        ag_text = r['agent_response'].replace('|', '/')
        f.write(f"| {r['question_ind']} | {r['category']} | {q_text} | {gt_text} | {ag_text} | **{r['nav_success']}** | {r['grounding_score']} | {r['accuracy_score']} | **{r['grounded_acc_pct']}%** |\n")

print(f"Successfully generated Markdown report: {md_file}")

# 3. Update CSV file
csv_file = "results/pred_eqa_200_results_summary.csv"
df = pd.DataFrame(compiled_rows)
df.to_csv(csv_file, index=False)
print(f"Successfully generated CSV report: {csv_file}")

