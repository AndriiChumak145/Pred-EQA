# Architecture & Engineering Decision Records (ADRs)

This document records key architectural, implementation, and workaround decisions made in the `Pred-EQA` codebase. 

---

## Format Standard
Each record follows the lightweight ADR structure:
* **Status:** `[Accepted | Superseded | Proposed | Rejected]`
* **Context & Problem:** What was the issue or requirement?
* **Decision:** What specific change was implemented?
* **Rationale:** Why was this option chosen over alternatives?
* **Consequences:** What are the trade-offs, limitations, or downstream impacts?

---

## Decision Records

### [ADR-001] Local 8-Bit Qwen3-VL FastAPI Server for VLM Inference
* **Date:** 2026-07-21
* **Status:** Accepted
* **Context & Problem:** The original Pred-EQA pipeline relied on external cloud API endpoints (e.g. OpenAI / GPT-4o), which introduced high evaluation costs, network latency, rate limits, and non-reproducible external dependencies.
* **Decision:** Built a local OpenAI-compatible FastAPI server (`llm_native_server.py` / `qwen_native_server.py`) hosting `Qwen/Qwen3-VL-8B-Instruct` using 8-bit `bitsandbytes` quantization on GPU port 8000.
* **Rationale:** Provides zero API costs, deterministic local evaluation, full privacy, and fast local GPU inference while retaining high visual-question-answering capabilities.
* **Consequences:** Requires ~29 GB VRAM on GPU 0. Requires running the background server process before running evaluation.

---

### [ADR-002] Pure NumPy Vectorization in `tsdf_base.py` Replacing Numba `@njit`
* **Date:** 2026-07-22
* **Status:** Accepted
* **Context & Problem:** In Python 3.11 with recent `Numba` versions, calling `@njit(parallel=True)` on `vox2world()`, `cam2pix()`, and `integrate_tsdf()` threw a fatal runtime exception: `TypeError: can't unbox array from PyObject into native value`.
* **Decision:** Removed `@njit` decorators and explicit Python `for` loops in [`src/tsdf_base.py`](file:///home/dani/concept-scenesplat/Pred-EQA/src/tsdf_base.py#L124-L175). Replaced them with pure vectorized NumPy operations (e.g., `vol_origin + (vox_size * vox_coords)`) and added division-by-zero protection in `cam2pix()`.
* **Rationale:** Vectorized NumPy math is mathematically identical to the loop implementation and completely eliminates Numba C-unboxing type compilation crashes without needing complex Numba array type signatures or C-contiguous memory re-allocations.
* **Consequences:** Executes purely in NumPy on CPU without Numba parallel threading overhead. Performance remains fast for standard TSDF volume dimensions.

---

### [ADR-003] Restoration of Official `habitat_sim` `quat_to_angle_axis()` with Zero-Division Guard
* **Date:** 2026-08-03 (Updated from 2026-07-22)
* **Status:** Accepted
* **Context & Problem:** In the original Pred-EQA code, evaluating `angle = angle * axis[1] / np.abs(axis[1])` triggered a `RuntimeWarning: invalid value encountered in scalar divide` whenever an episode's initial orientation had no Y-axis rotation (i.e. `axis[1] == 0`), producing `nan` angle values that corrupted downstream trigonometric calculations.
* **Decision:** Restored the official library import `quat_to_angle_axis` from `habitat_sim.utils.common` in [`src/utils.py`](file:///home/dani/concept-scenesplat/Pred-EQA/src/utils.py#L49-L69) and added a clean zero-division guard (`if axis[1] != 0: ... else: angle = 0.0`), removing the temporary redundant `safe_quat_to_angle_axis` wrapper.
* **Rationale:** Preserves the official Habitat-Sim library API while cleanly handling the `nan` scalar division edge case.
* **Consequences:** Clean, minimal codebase change fully compatible with `habitat_sim` and `quaternion`.

---

### [ADR-004] VLM Generation Hyperparameters (`repetition_penalty=1.15`, `max_new_tokens=1024`)
* **Date:** 2026-07-22
* **Status:** Accepted
* **Context & Problem:** Default VLM generation settings caused `Qwen3-VL-8B-Instruct` to occasionally enter infinite reasoning repetition loops when formatting XML plan updates or memory management summaries, inflating step latency to >10 minutes per step.
* **Decision:** Set `repetition_penalty=1.15` and capped `max_new_tokens=1024` in `llm_native_server.py`.
* **Rationale:** Forces concise reasoning output and prevents repetitive token loops without degrading XML checklist or snapshot retention quality.
* **Consequences:** Reduced per-step latency ~4x down to ~2 minutes per step.

---

### [ADR-005] Automated `tmux` Pipeline Orchestration Script
* **Date:** 2026-07-24
* **Status:** Accepted
* **Context & Problem:** Manually launching the Qwen server, waiting for GPU VRAM allocation, and then starting `run_express_bench_evaluation_vlm_only.py` in separate terminals was error-prone and tedious across machine reboots.
* **Decision:** Created [`start_pred_eqa_pipeline.sh`](file:///home/dani/concept-scenesplat/Pred-EQA/start_pred_eqa_pipeline.sh) to check/launch `qwen_server` in `tmux`, poll `http://127.0.0.1:8000` until ready, and then launch `pred_eqa_eval` in `tmux`.
* **Rationale:** Provides single-command reproduction and reliable overnight background execution.
* **Consequences:** Dependent on `tmux` being installed on the host system.
