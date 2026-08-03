# Pred-EQA Research & Experiment Findings Log

This file serves as an append-only chronological log of experiments, empirical discoveries, performance benchmarks, and rejected approaches in the `Pred-EQA` project.

---

## Log Format Standard
Each entry is dated and structured as follows:
* **Objective / Hypothesis:** What were we trying to evaluate or discover?
* **What Was Tried:** Details of the run, prompt setup, or parameters.
* **Results & Empirical Observations:** Key metrics, behavior, or failures.
* **Why Discarded / Adopted:** Rationale for keeping or abandoning the approach.
* **Key Lessons Learned:** Takeaways for future experiments.

---

## Log Entries

### [2026-07-21] Initial 11-Episode Evaluation Batch on EXPRESS-Bench
* **Objective / Hypothesis:** Benchmark Pred-EQA performance using local Qwen3-VL-8B-Instruct on the first subset of EXPRESS-Bench questions (`--start_ratio 0 --end_ratio 0.00587`, QIDs `0`, `1`, `10`, `100`, `1000`–`1006`).
* **What Was Tried:** Ran `run_express_bench_evaluation_vlm_only.py` with 11 episodes.
* **Results & Empirical Observations:**
  * **Navigation Success Rate:** **100.0%** (11 / 11 episodes succeeded).
  * **Mean Path Length:** **5.04 m** (vs Fine-EQA baseline **50.17 m** — a **~10x path reduction**).
  * **QA Accuracy:** **81.8%** (9 / 11 correct).
* **Why Adopted:** Verified that local Qwen3-VL-8B-Instruct can effectively serve as the VLM backend for Pred-EQA's multi-agent planning and memory management modules.
* **Key Lessons Learned:** In 4 out of 11 episodes (`0`, `1000`, `1001`, `1003`), Pred-EQA achieved immediate early stopping at 0.00 m (0 steps) because the Answerer recognized the target in initial snapshots, avoiding redundant wandering.

---

### [2026-07-22] Episode Duration & Latency Variance Analysis
* **Objective / Hypothesis:** Analyze runtime variance across episodes to understand why overnight evaluation throughput varied significantly.
* **What Was Tried:** Parsed execution timestamps and step counts across 17 completed episodes from `pred_eqa_12_episodes.log` and `pred_eqa_tmux_batch.log`.
* **Results & Empirical Observations:**
  * **Min Duration:** **2.13 min** (128 s) — 1 step (early stop).
  * **Max Duration:** **39.52 min** (2,371 s) — 16 steps (full exploration).
  * **Mean Duration:** **15.51 min** (930 s).
  * **Std Deviation:** **12.06 min** (723 s).
* **Why Kept:** Explains why overall batch duration depends heavily on the ratio of early-stopped vs deep-exploration episodes.
* **Key Lessons Learned:** Each navigation step triggers 6 sequential VLM queries (Snapshot Manager, Frontier Manager, High-Level Planner, Low-Level Planner, Answerer, Step Summarizer). Per-step latency is stable (~2–2.5 min), so total episode runtime scales linearly with step count.

---

### [2026-07-23] 40-Episode Benchmark Summary & Performance Metrics
* **Objective / Hypothesis:** Evaluate aggregate performance across 40 unique benchmark episodes (`express_bench_info_0.0_0.09785.json` and related runs).
* **What Was Tried:** Multi-batch evaluation up to question ratio `0.09785` (200 target episodes).
* **Results & Empirical Observations:**
  * **Overall Success Rate:** **87.5%** (35 / 40 unique episodes succeeded).
  * **Overall Mean Path Length:** **4.90 m**.
  * **Overall Mean Geodesic Distance:** **7.13 m**.
  * **200-Episode Active Batch (36 saved episodes):** **80.6% Success Rate** (29/36), **5.40 m** mean path length.
* **Key Lessons Learned:** Pred-EQA consistently maintains high navigation success rates (>80%) while keeping navigation paths extremely short (~5 m) compared to standard non-predictive baseline planners.

---

### [2026-07-24] GPU VRAM Allocation & Multi-Process Resource Limits
* **Objective / Hypothesis:** Diagnose why Qwen server failed to initialize during background startup (`ValueError: Some modules are dispatched on the CPU or the disk`).
* **What Was Tried:** Inspected GPU memory using `nvidia-smi` and process tree analysis.
* **Results & Empirical Observations:** Found an active Jupyter notebook kernel (`PID 8729`, env `concept_pose`) allocating **21.2 GB VRAM** on GPU 0, leaving only ~11 GB free. Qwen3-VL-8B 8-bit requires ~16–29 GB VRAM, causing allocation failure.
* **Why Documented:** Critical operational constraint for shared GPU environments.
* **Key Lessons Learned:** Always verify free GPU VRAM (`nvidia-smi`) before launching `start_pred_eqa_pipeline.sh`. The Qwen server requires at least 20 GB free VRAM to load successfully.
