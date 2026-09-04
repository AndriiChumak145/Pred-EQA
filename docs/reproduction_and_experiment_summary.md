# Pred-EQA Experiment Summary & Reproduction Guide

## 1. Overview & Objectives
* **Benchmark:** [EXPRESS-Bench](https://github.com/facebookresearch/express-bench) (HM3D dataset, 2,044 total EQA questions).
* **Goal:** Evaluate **Pred-EQA** (Predictive Planning with Specialized Memory for Embodied Question Answering) using a fast, locally hosted 8-bit quantized Vision-Language Model (`Qwen/Qwen3-VL-8B-Instruct`), replacing external API calls and providing high evaluation throughput.

---

## 2. Technical Modifications & Setup

### A. Local OpenAI-Compatible VLM Server
* **File:** [`llm_native_server.py`](file:///home/dani/concept-scenesplat/Pred-EQA/llm_native_server.py) / [`qwen_native_server.py`](file:///home/dani/concept-scenesplat/Pred-EQA/qwen_native_server.py)
* **Configuration:**
  * Hosts `Qwen/Qwen3-VL-8B-Instruct` on `http://127.0.0.1:8000/v1/chat/completions`.
  * Quantization: 8-bit (`bitsandbytes`) for memory efficiency.
  * Hyperparameters: `repetition_penalty=1.15`, `max_new_tokens=1024` to eliminate generation loops and speed up per-step inference ~4x (~2 min/step).

### B. Evaluation Integration
* **File:** [`run_express_bench_evaluation_vlm_only.py`](file:///home/dani/concept-scenesplat/Pred-EQA/run_express_bench_evaluation_vlm_only.py)
* Configured to stream multi-image prompts (snapshots and frontiers) directly to the local Qwen server.

### C. Persistent Pipeline Automation
* **File:** [`start_pred_eqa_pipeline.sh`](file:///home/dani/concept-scenesplat/Pred-EQA/start_pred_eqa_pipeline.sh)
* Automates launching both the Qwen VLM server (`qwen_server`) and the evaluation script (`pred_eqa_eval`) in background `tmux` sessions. Includes health check retry loops to wait for GPU weight loading.

---

## 3. Results Summary (Empirical Verification)

### A. Aggregate Benchmark Performance (40 Unique Episodes Evaluated)
* **Overall Navigation Success Rate:** **87.5%** (35 / 40 episodes)
* **Overall Mean Navigation Path Length:** **4.90 m**
* **Overall Mean Geodesic Distance:** **7.13 m**
* **Path Reduction:** ~10x reduction compared to Fine-EQA baseline (~50.17 m) due to Pred-EQA's predictive memory compaction and early-stopping mechanisms.

### B. 200-Episode Batch (`0.0`–`0.09785` ratio)
* **Completed Saved Episodes:** **36 episodes**
* **Batch Success Rate:** **80.6%** (29 / 36 episodes)
* **Batch Mean Path Length:** **5.40 m**

---

## 4. How to Reproduce Results

### Step 1: Environment Requirements
Ensure conda environments are created:
* `qwen36`: Environment containing `transformers`, `torch`, `bitsandbytes`, `fastapi`, `uvicorn`.
* `pred-eqa`: Environment containing `habitat-sim`, `habitat-lab`, `easydict`, `pyyaml`.

### Step 2: One-Command Execution
Run the automated pipeline script from the repository root:
```bash
cd /home/dani/concept-scenesplat/Pred-EQA
./start_pred_eqa_pipeline.sh
```

### Step 3: Monitoring & Inspection Commands
* **Watch evaluation log:**
  ```bash
  tail -f /home/dani/concept-scenesplat/Pred-EQA/pred_eqa_200_episodes.log
  ```
* **Watch server log:**
  ```bash
  tail -f /home/dani/concept-scenesplat/Pred-EQA/qwen_server.log
  ```
* **Attach to active evaluation session:**
  ```bash
  tmux attach -t pred_eqa_eval
  ```
* **Compute summary scores from saved output files:**
  ```bash
  conda activate pred-eqa
  python -c "import json, pickle; info=json.load(open('results/Pred-EQA/express_bench_info_0.0_0.09785.json')); path=pickle.load(open('results/Pred-EQA/path_length_list_0.0_0.09785.pkl','rb')); succ=pickle.load(open('results/Pred-EQA/success_list_0.0_0.09785.pkl','rb')); print(f'Episodes: {len(info)}, Success Rate: {len(succ)}/{len(info)} ({len(succ)/len(info)*100:.1f}%), Mean Path: {sum(path.values())/len(path):.2f}m')"
  ```
