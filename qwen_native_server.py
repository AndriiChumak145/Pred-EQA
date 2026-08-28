import sys

# Backward-compatible wrapper for qwen_native_server.py
# Delegates to llm_native_server.py with Qwen3-VL default model
if __name__ == "__main__":
    if "--model" not in sys.argv and "--model_id" not in sys.argv:
        sys.argv.extend(["--model", "Qwen/Qwen3-VL-8B-Instruct"])
    from llm_native_server import app, HOST, PORT, uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
