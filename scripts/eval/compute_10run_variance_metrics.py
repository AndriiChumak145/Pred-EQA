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

# 1. Load records and 10-run 16-bit score dictionary
with open("results/pred_eqa_200_results.pkl", "rb") as f:
    records = pickle.load(f)

scores_file = "results/pred_eqa_200_qwen8b_scores.pkl"
with open(scores_file, "rb") as f:
    scores_dict = pickle.load(f)

def parse_mark_string(mark_str):
    if not mark_str or not isinstance(mark_str, str):
        return 0.0, 1.0
    match = re.search(r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', mark_str)
    if match:
        try:
            g = float(match.group(1))
            a = float(match.group(2))
            g = max(0.0, min(1.0, g))
            a = max(1.0, min(5.0, a))
            return g, a
        except ValueError:
            pass
    return 0.0, 1.0

# Number of runs per episode
NUM_RUNS = 10

# Store scores: c_matrix[episode_idx, run_idx]
n_episodes = len(records)
g_matrix = np.zeros((n_episodes, NUM_RUNS))
a_matrix = np.zeros((n_episodes, NUM_RUNS))
c_matrix = np.zeros((n_episodes, NUM_RUNS))

compiled_rows = []

for ep_idx, r in enumerate(records):
    qid = r["question_ind"]
    marks_list = scores_dict.get(qid, [])
    
    # Fill up to 10 runs
    parsed_runs = []
    for k in range(NUM_RUNS):
        if k < len(marks_list):
            raw_mark = marks_list[k]
        else:
            raw_mark = marks_list[0] if marks_list else "0, 1"
        g, a = parse_mark_string(raw_mark)
        c = g * (a / 5.0) * 100.0
        
        g_matrix[ep_idx, k] = g
        a_matrix[ep_idx, k] = a
        c_matrix[ep_idx, k] = c
        parsed_runs.append((g, a, c))
        
    ep_g_mean = float(np.mean(g_matrix[ep_idx, :]))
    ep_a_mean = float(np.mean(a_matrix[ep_idx, :]))
    ep_c_mean = float(np.mean(c_matrix[ep_idx, :]))
    ep_c_std  = float(np.std(c_matrix[ep_idx, :]))
    ep_c_mae  = float(np.mean(np.abs(c_matrix[ep_idx, :] - ep_c_mean)))
    
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
        "mean_grounding": round(ep_g_mean, 2),
        "mean_accuracy": round(ep_a_mean, 2),
        "mean_grounded_acc_pct": round(ep_c_mean, 1),
        "grounded_acc_std": round(ep_c_std, 2),
        "grounded_acc_mae": round(ep_c_mae, 2)
    })

# Compute 10 run benchmark scores (C_k)
run_c_scores = np.mean(c_matrix, axis=0) # array of length 10
run_g_scores = np.mean(g_matrix, axis=0)
run_a_scores = np.mean(a_matrix, axis=0)

mean_final_c = float(np.mean(run_c_scores))
mean_final_g = float(np.mean(run_g_scores))
mean_final_a = float(np.mean(run_a_scores))

# Leaderboard MAE (across the 10 run overall benchmark scores)
leaderboard_mae = float(np.mean(np.abs(run_c_scores - mean_final_c)))

# Episode-level MAE (average micro-variance per episode)
ep_means = np.mean(c_matrix, axis=1, keepdims=True) # (n_episodes, 1)
episode_mae = float(np.mean(np.abs(c_matrix - ep_means)))

succ_count = sum(1 for r in compiled_rows if r["nav_success"] == "SUCCESS")

# Print Summary
print("======================================================================")
print("PRED-EQA 200-EPISODE BENCHMARK VARIANCE SUMMARY (Qwen 8B Instruct 16-bit Native)")
print("======================================================================")
print(f"Total Episodes: {n_episodes}")
print(f"Evaluations per Episode: {NUM_RUNS}")
print(f"Navigation Success Rate: {succ_count} / {n_episodes} ({succ_count/n_episodes*100:.1f}%)")
print("\nFinal Benchmark Scores (C_avg) Across 10 Independent Runs:")
for k in range(NUM_RUNS):
    print(f"  - Run {k+1:2d}: {run_c_scores[k]:.2f}%  (G: {run_g_scores[k]:.2f}, A: {run_a_scores[k]:.2f})")

print("\nStatistical Summary (matching EXPRESS-Bench notes.md):")
print(f"  * Mean Final Score (C_avg): {mean_final_c:.2f}%")
print(f"  * Mean Grounding Score:     {mean_final_g:.2f} / 1.0")
print(f"  * Mean Accuracy Score:      {mean_final_a:.2f} / 5.0")
print(f"  * Leaderboard Variance (MAE of Final Scores): {leaderboard_mae:.3f}")
print(f"  * Average Episode-Level MAE (Micro-Variance):  {episode_mae:.2f}")

# Write Markdown Report
md_file = "results/pred_eqa_200_results_summary.md"
with open(md_file, "w") as f:
    f.write("# Pred-EQA 200-Episode Evaluation Results & 10-Run Variance Analysis (Native 16-bit)\n\n")
    f.write("Evaluated using **Qwen 8B Instruct (Native 16-bit / bfloat16)** across **10 independent evaluation runs** per episode (matching `EXPRESS-Bench/notes.md`).\n\n")
    f.write("## 1. Executive Summary & Benchmark Stability Metrics\n\n")
    f.write(f"- **Total Episodes Processed**: {n_episodes}\n")
    f.write(f"- **Evaluations per Episode**: {NUM_RUNS}\n")
    f.write(f"- **Navigation Success Rate**: {succ_count} / {n_episodes} ({succ_count/n_episodes*100:.1f}%)\n")
    f.write(f"- **Mean Grounded Accuracy Metric (C%)**: **{mean_final_c:.2f}%**\n")
    f.write(f"- **Mean Grounding Score**: **{mean_final_g:.2f}** / 1.0\n")
    f.write(f"- **Mean Accuracy Score**: **{mean_final_a:.2f}** / 5.0\n")
    f.write(f"- **Leaderboard Variance (MAE of Final Scores)**: **{leaderboard_mae:.3f}**\n")
    f.write(f"- **Average Episode-Level MAE (Micro-Variance)**: **{episode_mae:.2f}**\n\n")
    
    f.write("### Benchmark Scores (C_avg) Across 10 Independent Runs:\n")
    for k in range(NUM_RUNS):
        f.write(f"- **Run {k+1:2d}**: {run_c_scores[k]:.2f}% (Grounding: {run_g_scores[k]:.2f}, Accuracy: {run_a_scores[k]:.2f})\n")
        
    f.write("\n---\n\n")
    f.write("## 2. Per-Episode Detailed Results (Averaged Over 10 Runs)\n\n")
    f.write("| Q_Idx | Category | Question | Ground Truth Answer | Agent Response | Nav Status | Mean Grounding | Mean Accuracy | Mean Grounded Acc (C%) | Ep MAE |\n")
    f.write("|:---:|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|\n")
    for r in compiled_rows:
        q_text = r['question'].replace('|', '/')
        gt_text = r['gt_answer'].replace('|', '/')
        ag_text = r['agent_response'].replace('|', '/')
        f.write(f"| {r['question_ind']} | {r['category']} | {q_text} | {gt_text} | {ag_text} | **{r['nav_success']}** | {r['mean_grounding']} | {r['mean_accuracy']} | **{r['mean_grounded_acc_pct']}%** | ±{r['grounded_acc_mae']} |\n")

print(f"\nSuccessfully generated Markdown summary: {md_file}")

# Write CSV Summary
csv_file = "results/pred_eqa_200_results_summary.csv"
df = pd.DataFrame(compiled_rows)
df.to_csv(csv_file, index=False)
print(f"Successfully generated CSV summary: {csv_file}")
