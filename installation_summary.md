# Pred-EQA Installation Summary

This document summarizes the steps taken to install the `pred-eqa` environment on an RTX 5090, detailing the exact commands used and explaining the critical deviations from the original README.

## Suggested Environment vs. Our Environment

| Component | Suggested (Original) | Ours (Actual) |
| :--- | :--- | :--- |
| **Python** | 3.9 | 3.11 |
| **CUDA Toolkit**| 11.8 | 12.8.1 |
| **PyTorch** | 2.0.1+cu118 | 2.12.0.dev20260408+cu128 (Nightly) |
| **Habitat-Sim** | 0.2.5 (Stable) | 0.3.3.2026.07.19 (aihabitat-nightly) |
| **PyTorch3D** | 0.7.4 | *Omitted* |
| **NumPy & SciPy**| Unpinned | NumPy < 2.0.0, SciPy == 1.12.0 |

### Why Did We Deviate?

1. **Hardware Constraints (CUDA 12.8):** The NVIDIA RTX 5090 (Blackwell architecture) strictly requires CUDA 12.8, which is only available in PyTorch nightly builds (`cu128`).
2. **Python Version Squeeze:** The PyTorch nightly `cu128` wheels require Python $\ge$ 3.10. However, the stable `habitat-sim=0.2.5` only supports Python 3.9. To bridge this gap, we upgraded to Python 3.11 and used a bleeding-edge `aihabitat-nightly` build of `habitat-sim` that includes a `py3.11_headless` binary.
3. **C-API Binary Incompatibility:** We pinned `numpy < 2.0.0` and downgraded `scipy == 1.12.0`. NumPy 2.0 introduces breaking C-API changes that cause legacy computer vision libraries (like `opencv-python` and `scikit-learn`) to instantly crash with `numpy.dtype size changed` errors.
4. **Conda Resolver Traps:** We split the installation, installing `habitat-sim` via Conda but `faiss-cpu` via `pip`. Combining them caused Conda's C++ resolver to crash because it tried to resolve C++ shared libraries against our forcefully injected pip-installed PyTorch.
5. **Omitting PyTorch3D:** Compiling `pytorch3d=0.7.4` from source using the required modern `gcc=14.3` compiler causes fatal C++ template errors deep inside PyTorch 2.12's C++ headers. Since we verified the predictive planner relies exclusively on a "pure VLM pipeline" and does not use the legacy `conceptgraph` detectors, we safely omitted it entirely.

---

## Final Installation Commands

To recreate this environment, run the following commands sequentially:

```bash
# 1. Base Environment & Compilers
conda create -n pred-eqa python=3.11 -y
conda activate pred-eqa
conda install -c nvidia -c conda-forge cuda-toolkit=12.8.1 gcc=14.3 gxx=14.3 -y

# 2. Force-Install PyTorch Nightly (cu128)
pip install torch==2.12.0.dev20260408+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
pip install torchvision==0.27.0.dev20260407+cu128 torchaudio==2.11.0.dev20260407+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128 --no-deps --no-cache-dir

# 3. Pin NumPy (1.x)
pip install "numpy<2.0.0" --no-deps

# 4. Install Habitat-Sim (Nightly Build)
conda install -c aihabitat-nightly -c conda-forge habitat-sim=*=*headless* -y

# 5. Install Remainder Dependencies via Pip
pip install faiss-cpu==1.7.4 scipy==1.12.0
pip install cmake ninja omegaconf==2.3.0 supervision==0.21.0 opencv-python-headless==4.10.* scikit-learn==1.4 scikit-image==0.22 open3d==0.18.0 hipart==1.0.4 openai==1.35.3 httpx==0.27.2 qwen_vl_utils
```
