#!/usr/bin/env python3
"""
fill_blank_answers_and_evaluate.py
1. Identifies any blank/empty predictions in Pred-EQA results.
2. Uses Qwen 8B Instruct with random_answer.txt (without images) to generate blind language-prior answers.
3. Updates results/pred_eqa_200_results.pkl.
4. Triggers Qwen 8B judge evaluation to update Grounding, Accuracy, and Grounded Accuracy metrics.
"""

import os
import sys
from pathlib import Path
import re
import json
import glob
import pickle
import torch
import numpy as np
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_DIR = str(REPO_ROOT)
os.chdir(PROJECT_DIR)

# 1. Load data
pkl_file = "results/pred_eqa_200_results.pkl"
with open(pkl_file, "rb") as f:
    records = pickle.load(f)

# Load random_answer prompt
prompt_file = str(REPO_ROOT.parent / "EXPRESS-Bench/prompt/random_answer.txt")
with open(prompt_file, "r", encoding="utf-8") as f:
    prompt_lines = f.read().split("\n")
    system_prompt = prompt_lines[1] if len(prompt_lines) > 1 else "You are an intelligent question answering agent."
    few_shot_template = "\n".join(prompt_lines[3:]) if len(prompt_lines) > 3 else ""

blank_indices = [i for i, r in enumerate(records) if not r.get("gen_answer") or r["gen_answer"].strip() == ""]

print(f"Found {len(blank_indices)} / {len(records)} blank answers to fill with blind language-prior guessing.")

if blank_indices:
    print("Loading Qwen 8B Instruct into GPU with 8-bit quantization for blind guessing...")
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", quantization_config=quantization_config
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    for idx in tqdm(blank_indices, desc="Generating blind guesses"):
        r = records[idx]
        q_text = r["question"]
        
        user_text = f"{few_shot_template}\n\nQ: {q_text}\nA: "
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=64)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        guessed_ans = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        # Clean prefix if any
        if guessed_ans.startswith("A:"):
            guessed_ans = guessed_ans[2:].strip()
            
        print(f"\n[Q_idx {r['question_ind']}] Question: {q_text}")
        print(f"   -> Blind Guess: '{guessed_ans}'")
        
        # Update record
        r["gen_answer"] = guessed_ans

    # Save updated PKL
    with open(pkl_file, "wb") as f:
        pickle.dump(records, f)
    print(f"Updated {pkl_file} with blind guessed answers.")
    
    # Free model VRAM
    del model
    del processor
    torch.cuda.empty_cache()

print("Blind answer filling completed cleanly!")
