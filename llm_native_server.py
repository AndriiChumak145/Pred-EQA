import argparse
import logging
import os
import threading
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
import torch
import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

import transformers
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

# Try importing Qwen VL utilities if available
try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False

# Try importing Qwen3-VL conditional generation class
try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3_VL = True
except ImportError:
    HAS_QWEN3_VL = False

# Setup timestamped logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm_native_server")


# Parse command line arguments
parser = argparse.ArgumentParser(description="Universal LLM/VLM Native FastAPI Server")
parser.add_argument(
    "--model",
    "--model_id",
    dest="model_id",
    type=str,
    default=os.getenv("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct"),
    help="HuggingFace model ID or local directory path (e.g. google/gemma-3-12b-it, Qwen/Qwen3-VL-8B-Instruct, google/gemma-2-9b-it)",
)
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind the server")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server")
parser.add_argument(
    "--quantization",
    type=str,
    choices=["8bit", "4bit", "none"],
    default="8bit",
    help="Quantization mode: 8bit, 4bit, or none",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="auto",
    help="Torch dtype (e.g. auto, bfloat16, float16, float32)",
)


def resolve_cached_model_id(model_id: str) -> str:
    if os.path.exists(model_id):
        return model_id

    hf_cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
    if not os.path.exists(hf_cache_dir):
        return model_id

    parts = model_id.split("/")
    if len(parts) == 2:
        org, repo = parts
        target_prefix = f"models--{org.lower()}--{repo.lower()}"
        candidate_folders = []
        try:
            for folder in os.listdir(hf_cache_dir):
                if folder.lower() == target_prefix:
                    folder_path = os.path.join(hf_cache_dir, folder)
                    blobs_path = os.path.join(folder_path, "blobs")
                    blob_size = 0
                    if os.path.exists(blobs_path):
                        for b in os.listdir(blobs_path):
                            b_file = os.path.join(blobs_path, b)
                            if os.path.isfile(b_file):
                                blob_size += os.path.getsize(b_file)
                    candidate_folders.append((folder, blob_size))
            if candidate_folders:
                candidate_folders.sort(key=lambda x: x[1], reverse=True)
                best_folder, best_size = candidate_folders[0]
                real_org_repo = best_folder.replace("models--", "").split("--", 1)
                if len(real_org_repo) == 2:
                    resolved = f"{real_org_repo[0]}/{real_org_repo[1]}"
                    if resolved != model_id:
                        logger.info(
                            "Auto-resolved '%s' -> '%s' from local cache (%.2f GB cached)",
                            model_id,
                            resolved,
                            best_size / (1024**3),
                        )
                    return resolved
        except Exception as e:
            logger.warning("Cache resolution encountered an issue: %s", e)
    return model_id


args = parser.parse_args()

MODEL_ID = resolve_cached_model_id(args.model_id)
HOST = args.host
PORT = args.port

app = FastAPI(title=f"LLM Native Server ({MODEL_ID})")


class TokenTracker:
    """Thread-safe cumulative token tracker supporting global and per-session metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.global_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def reset(self, session_id: Optional[str] = None):
        with self._lock:
            if session_id:
                if session_id in self.sessions:
                    del self.sessions[session_id]
            else:
                self.global_stats = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "requests": 0,
                }
                self.sessions.clear()
                self.start_time = time.time()

    def record_usage(self, session_id: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
        total_tokens = prompt_tokens + completion_tokens
        now = time.time()
        with self._lock:
            self.global_stats["prompt_tokens"] += prompt_tokens
            self.global_stats["completion_tokens"] += completion_tokens
            self.global_stats["total_tokens"] += total_tokens
            self.global_stats["requests"] += 1

            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "requests": 0,
                    "start_time": now,
                    "last_active": now,
                }
            s = self.sessions[session_id]
            s["prompt_tokens"] += prompt_tokens
            s["completion_tokens"] += completion_tokens
            s["total_tokens"] += total_tokens
            s["requests"] += 1
            s["last_active"] = now

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            uptime_sec = time.time() - self.start_time
            return {
                "uptime_seconds": round(uptime_sec, 2),
                "global": dict(self.global_stats),
                "active_sessions_count": len(self.sessions),
                "sessions": {k: dict(v) for k, v in self.sessions.items()},
            }


token_tracker = TokenTracker()


def load_image(url_or_path: str) -> Image.Image:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        res = requests.get(url_or_path, timeout=15)
        return Image.open(BytesIO(res.content)).convert("RGB")
    elif url_or_path.startswith("file://"):
        path = url_or_path[7:]
        return Image.open(path).convert("RGB")
    else:
        return Image.open(url_or_path).convert("RGB")


def load_model_and_processor(model_id: str, quantization: str = "8bit", dtype: str = "auto"):
    logger.info("Loading model '%s' (quantization=%s, dtype=%s)...", model_id, quantization, dtype)

    # 1. Load Processor / Tokenizer
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        logger.info("Loaded AutoProcessor.")
    except Exception as e:
        logger.info("AutoProcessor not found or failed (%s), falling back to AutoTokenizer...", e)
        processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        logger.info("Loaded AutoTokenizer.")

    # 2. Setup Quantization Config
    quantization_config = None
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype, "auto")

    # 3. Model Loading Strategy
    model = None
    model_lower = model_id.lower()

    # Load Qwen3-VL directly if available
    if "qwen3" in model_lower and "vl" in model_lower and HAS_QWEN3_VL:
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True,
            )
            logger.info("Successfully loaded model with Qwen3VLForConditionalGeneration.")
        except Exception as e:
            logger.warning("Qwen3VLForConditionalGeneration load attempt failed: %s", e)

    # Auto class fallbacks: AutoModelForMultimodalLM -> AutoModelForImageTextToText -> AutoModelForCausalLM -> AutoModel
    auto_classes = [
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForCausalLM",
        "AutoModel",
    ]

    if model is None:
        for cls_name in auto_classes:
            if hasattr(transformers, cls_name):
                cls = getattr(transformers, cls_name)
                try:
                    logger.info("Attempting to load model with %s...", cls_name)
                    model = cls.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                        quantization_config=quantization_config,
                        trust_remote_code=True,
                    )
                    logger.info("Successfully loaded model with %s!", cls_name)
                    break
                except Exception as e:
                    logger.warning("%s attempt failed: %s", cls_name, e)

    if model is None:
        raise RuntimeError(f"Failed to load model '{model_id}' with available HuggingFace model architectures.")

    return model, processor


model, processor = load_model_and_processor(MODEL_ID, quantization=args.quantization, dtype=args.dtype)
logger.info("Model '%s' loaded successfully and ready to serve requests!", MODEL_ID)


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 1.0
    session_id: Optional[str] = None


class ResetUsageRequest(BaseModel):
    session_id: Optional[str] = None


def format_fallback_chat(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


@app.get("/health")
@app.get("/v1/models")
async def get_health():
    return {"status": "ok", "data": [{"id": MODEL_ID}]}


@app.get("/v1/usage")
@app.get("/usage")
async def get_usage():
    return token_tracker.get_summary()


@app.post("/v1/usage/reset")
@app.post("/usage/reset")
async def reset_usage(request: Optional[ResetUsageRequest] = None):
    session_to_reset = request.session_id if request else None
    token_tracker.reset(session_id=session_to_reset)
    target = f"session '{session_to_reset}'" if session_to_reset else "all sessions and global stats"
    logger.info("Usage counters reset for %s.", target)
    return {"status": "ok", "message": f"Reset token counters for {target}"}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    # Determine session id from request parameter or header (defaults to "default")
    session_id = request.session_id or x_session_id or "default"
    messages = request.messages

    # Determine if prompt includes vision/image inputs
    has_images = False
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") in ("image_url", "image"):
                    has_images = True
                    break

    model_lower = MODEL_ID.lower()
    is_qwen_vl = "qwen" in model_lower and "vl" in model_lower

    if is_qwen_vl and HAS_QWEN_VL_UTILS and has_images:
        # Translate OpenAI format to Qwen VL format
        qwen_messages = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                qwen_content = []
                for item in msg["content"]:
                    if item.get("type") == "text":
                        qwen_content.append({"type": "text", "text": item["text"]})
                    elif item.get("type") in ("image_url", "image"):
                        url = item.get("image_url", {}).get("url") if item.get("type") == "image_url" else item.get("image")
                        qwen_content.append({"type": "image", "image": url})
                qwen_messages.append({"role": msg.get("role", "user"), "content": qwen_content})
            else:
                qwen_messages.append(msg)

        text = processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    elif has_images:
        # Generic multimodal model handling (e.g. Gemma 3, PaliGemma, Llama 3.2 Vision)
        formatted_messages = []
        raw_images = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            if isinstance(content, list):
                msg_content = []
                for item in content:
                    if item.get("type") == "text":
                        msg_content.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") in ("image_url", "image"):
                        url = item.get("image_url", {}).get("url") if item.get("type") == "image_url" else item.get("image")
                        try:
                            pil_img = load_image(url)
                            raw_images.append(pil_img)
                            msg_content.append({"type": "image", "image": pil_img})
                        except Exception as e:
                            logger.warning("Failed to load image from %s: %s", url, e)
                formatted_messages.append({"role": role, "content": msg_content})
            else:
                formatted_messages.append({"role": role, "content": content})

        if hasattr(processor, "apply_chat_template"):
            try:
                text = processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = format_fallback_chat(formatted_messages)
        else:
            text = format_fallback_chat(formatted_messages)

        if raw_images and callable(processor):
            inputs = processor(text=[text], images=raw_images, return_tensors="pt", padding=True)
        else:
            tokenizer = getattr(processor, "tokenizer", processor)
            inputs = tokenizer([text], return_tensors="pt", padding=True)
    else:
        # Standard text-only model handling (e.g. Gemma 12B IT, Llama, Qwen text, etc.)
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                text_str = "\n".join(text_parts)
            else:
                text_str = str(content)
            formatted_messages.append({"role": role, "content": text_str})

        if hasattr(processor, "apply_chat_template"):
            try:
                text = processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = format_fallback_chat(formatted_messages)
        else:
            text = format_fallback_chat(formatted_messages)

        tokenizer = getattr(processor, "tokenizer", processor)
        inputs = tokenizer([text], return_tensors="pt", padding=True)

    # Move tensor inputs to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

    with torch.inference_mode():
        max_tokens = min(request.max_tokens, 2048) if request.max_tokens else 1024
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": 1.15,
        }
        if request.temperature is not None and request.temperature > 0 and request.temperature != 1.0:
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["do_sample"] = True

        generated_ids = model.generate(**inputs, **gen_kwargs)

    # Calculate input token length to trim input prompt from generated text
    if "input_ids" in inputs:
        input_len = inputs["input_ids"].shape[1]
    else:
        input_len = 0

    generated_ids_trimmed = [out_ids[input_len:] for out_ids in generated_ids]

    tokenizer = getattr(processor, "tokenizer", processor)
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Clean up reasoning / thinking tags if present
    if "</think>" in output_text:
        after_think = output_text.split("</think>")[-1].strip()
        if after_think:
            output_text = after_think
        else:
            output_text = output_text.replace("</think>", "").replace("<think>", "").strip()

    output_text = output_text.strip()

    # Exact token calculation
    prompt_tokens = int(input_len)
    completion_tokens = int(generated_ids_trimmed[0].shape[0]) if len(generated_ids_trimmed) > 0 else 0
    usage_data = token_tracker.record_usage(
        session_id=session_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    logger.info(
        "Session: %s | Prompt tokens: %d | Completion tokens: %d | Total tokens: %d | Server Cumulative: %d",
        session_id,
        usage_data["prompt_tokens"],
        usage_data["completion_tokens"],
        usage_data["total_tokens"],
        token_tracker.global_stats["total_tokens"],
    )

    response = {
        "id": "chatcmpl-llm-native",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage_data,
    }
    return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
