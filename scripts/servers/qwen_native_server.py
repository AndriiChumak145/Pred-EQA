import sys
from pathlib import Path

# Add scripts/servers and REPO_ROOT to sys.path dynamically
SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Backward-compatible wrapper for qwen_native_server.py
# Delegates to llm_native_server.py with Qwen3-VL default model
if __name__ == "__main__":
    if "--model" not in sys.argv and "--model_id" not in sys.argv:
        sys.argv.extend(["--model", "Qwen/Qwen3-VL-8B-Instruct"])
    from llm_native_server import app, HOST, PORT, uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

