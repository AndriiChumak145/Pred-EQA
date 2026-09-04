import os
import sys
from pathlib import Path
import json
import pickle

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPRESS_BENCH_DATA = str(WORKSPACE_ROOT / "EXPRESS-Bench/data/express-bench.json")
PRED_EQA_PKL = str(REPO_ROOT / "results/pred_eqa_200_results.pkl")
OUTPUT_JSON = str(WORKSPACE_ROOT / "EXPRESS-Bench/data/express-bench-pred-eqa-200.json")

# 1. Load full express-bench dataset
with open(EXPRESS_BENCH_DATA, "r", encoding="utf-8") as f:
    full_dataset = json.load(f)

# 2. Load the 200 evaluated episodes from Pred-EQA
with open(PRED_EQA_PKL, "rb") as f:
    records = pickle.load(f)

evaluated_qids = [int(r["question_ind"]) for r in records]

print(f"Loaded {len(evaluated_qids)} evaluated question indices from Pred-EQA results.")

# 3. Filter the exact 200 items by index
subset_dataset = []
for qid in evaluated_qids:
    item = full_dataset[qid].copy()
    item["question_ind"] = qid
    item["question_id"] = str(qid)
    subset_dataset.append(item)

print(f"Successfully created subset with {len(subset_dataset)} items.")
print(f"Sample item 0 question: {subset_dataset[0]['question']}")

# 4. Save the subset JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(subset_dataset, f, indent=2, ensure_ascii=False)

print(f"Successfully saved 200-episode subset JSON to: {OUTPUT_JSON}")
