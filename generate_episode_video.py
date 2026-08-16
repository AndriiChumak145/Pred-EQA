import os
import glob
import json
import re
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Panel Display Configuration Constants
MAX_OBSERVATION_LINES = 10         # Maximum lines of observation text to render in bottom card
MAX_OBSERVATION_CHAR_LEN = -1     # Maximum character length of observation text (truncated with '...')

def load_express_bench_metadata():
    """Load questions and GT answers from express-bench.json."""
    dataset_path = "Pred-EQA/data/express-bench.json"
    if not os.path.exists(dataset_path):
        return {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["question_id"]): item for item in data if "question_id" in item}

def load_model_answers(results_root="Pred-EQA/results/Pred-EQA"):
    """Load model answers from gpt_answer files."""
    answers = {}
    answer_files = sorted(set(
        glob.glob(f"{results_root}/gpt_answer*.json") + 
        glob.glob("Pred-EQA/results/**/gpt_answer*.json") + 
        glob.glob("Pred-EQA/gpt_answer*.json")
    ))
    for afile in answer_files:
        try:
            with open(afile, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    for item in content:
                        qid = str(item.get("question_id"))
                        ans = item.get("answer")
                        if ans and qid not in answers:
                            answers[qid] = ans
                elif isinstance(content, dict):
                    for qid, ans in content.items():
                        if ans and str(qid) not in answers:
                            answers[str(qid)] = ans
        except Exception:
            pass
    return answers

def parse_episode_plans(episode_id, results_root="Pred-EQA/results/Pred-EQA"):
    """Parse step-by-step high-level todo plans, chosen actions, step observations, and low-level reasons from log files."""
    step_plans = {}
    chosen_actions = {}
    step_reasons = {}
    step_observations = {}
    
    # Search results directory log files first, then fallback to root log files
    log_files = sorted(glob.glob(f"{results_root}/log_*.log")) + sorted(glob.glob("Pred-EQA/*.log"))
    
    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        pattern = rf'(Index:\s*{episode_id}\b|Question id\s*{episode_id}\b)'
        m = re.search(pattern, content)
        if m:
            start_pos = m.start()
            next_m = re.search(r'\nIndex:\s*\d+', content[start_pos+10:])
            block = content[start_pos:start_pos+10+next_m.start()] if next_m else content[start_pos:]
            
            step_chunks = re.split(r'== step:\s*(\d+)', block)
            if len(step_chunks) > 1:
                for i in range(1, len(step_chunks), 2):
                    step_num = int(step_chunks[i])
                    schunk = step_chunks[i+1]
                    
                    matches = re.findall(r'<todos>(.*?)</todos>', schunk, re.DOTALL)
                    real_matches = [m.strip() for m in matches if "Pending task description" not in m]
                    if real_matches:
                        step_plans[step_num] = real_matches[-1]
                        
                    res_matches = list(re.finditer(r'response:\s*(frontier|snapshot)\s*(\d+)', schunk, re.IGNORECASE))
                    if res_matches:
                        last_res = res_matches[-1]
                        chosen_actions[step_num] = {
                            "type": last_res.group(1).lower(),
                            "id": int(last_res.group(2))
                        }
                    
                    obs_match = re.search(r'Successfully recorded step \d+ summary:\s*(.*)', schunk)
                    if obs_match:
                        step_observations[step_num] = obs_match.group(1).strip()
                        
                    reasons = re.findall(r'reason:\s*(.*?)(?=\n\d\d:\d\d:\d\d\s*-|\n\n[A-Z]|$)', schunk, re.DOTALL | re.IGNORECASE)
                    cleaned_reasons = []
                    for r in reasons:
                        r_clean = r.strip()
                        r_clean = re.sub(r'^\d\d:\d\d:\d\d\s*-\s*', '', r_clean).strip()
                        if r_clean and '######' not in r_clean and len(r_clean) > 2:
                            cleaned_reasons.append(r_clean)
                    if cleaned_reasons:
                        step_reasons[step_num] = cleaned_reasons[-1]
                break
            else:
                matches = re.findall(r'<todos>(.*?)</todos>', block, re.DOTALL)
                real_matches = [m.strip() for m in matches if "Pending task description" not in m]
                for i, rmatch in enumerate(real_matches):
                    step_plans[i] = rmatch
                    
                res_matches = list(re.finditer(r'response:\s*(frontier|snapshot)\s*(\d+)', block, re.IGNORECASE))
                if res_matches:
                    last_res = res_matches[-1]
                    chosen_actions[0] = {
                        "type": last_res.group(1).lower(),
                        "id": int(last_res.group(2))
                    }
                    
                obs_match = re.search(r'Successfully recorded step \d+ summary:\s*(.*)', block)
                if obs_match:
                    step_observations[0] = obs_match.group(1).strip()
                
                reasons = re.findall(r'reason:\s*(.*?)(?=\n\d\d:\d\d:\d\d\s*-|\n\n[A-Z]|$)', block, re.DOTALL | re.IGNORECASE)
                cleaned_reasons = [re.sub(r'^\d\d:\d\d:\d\d\s*-\s*', '', r.strip()).strip() for r in reasons if r.strip()]
                if cleaned_reasons:
                    step_reasons[0] = cleaned_reasons[-1]
            break

    # Carry over active <todos> checklist state for missing steps
    max_step_index = 30
    if step_plans:
        max_k = max(step_plans.keys())
        max_step_index = max(max_step_index, max_k + 5)
        
    latest_plan = ""
    for s in range(max_step_index + 1):
        if s in step_plans and step_plans[s]:
            latest_plan = step_plans[s]
        elif latest_plan:
            step_plans[s] = latest_plan

    return step_plans, chosen_actions, step_reasons, step_observations

def wrap_text(text, font, max_width, draw):
    """Wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines

def format_todo_item(raw_line):
    """Parse raw todo line into status, description, and rationale."""
    raw_line = raw_line.strip()
    status = "[ ]"
    if raw_line.startswith("[x]"):
        status = "[x]"
        raw_line = raw_line[3:].strip()
    elif raw_line.startswith("[-]"):
        status = "[-]"
        raw_line = raw_line[3:].strip()
    elif raw_line.startswith("[ ]"):
        status = "[ ]"
        raw_line = raw_line[3:].strip()
        
    parts = raw_line.split("<!--")
    desc = parts[0].strip()
    rationale = ""
    if len(parts) > 1:
        rationale = parts[1].replace("-->", "").strip()
        
    return status, desc, rationale

def clean_rationale(text):
    """Clean rationale text by stripping log prefixes and extra whitespace."""
    if not text:
        return ""
    text = re.sub(r'^(Step-by-step Reasoning:|\d\d:\d\d:\d\d\s*-\s*)+', '', text, flags=re.IGNORECASE).strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return " ".join(lines)

def draw_reasoning_panel(draw, step, plan_text, action_info, reason_text, obs_text, panel_x, panel_w, canvas_h, font_title, font_text, font_small):
    """Render the high-level plan & low-level reasoning panel on the right side of the canvas."""
    # Background
    draw.rectangle([panel_x, 0, panel_x + panel_w, canvas_h], fill=(16, 20, 26))
    draw.line([(panel_x, 0), (panel_x, canvas_h)], fill=(0, 255, 120), width=3)
    
    # Header Banner
    draw.rectangle([panel_x + 15, 15, panel_x + panel_w - 15, 80], fill=(28, 35, 46))
    draw.text((panel_x + 30, 25), "HIGH-LEVEL PLAN & REASONING", fill=(255, 215, 0), font=font_title)
    draw.text((panel_x + 30, 55), f"Step {step} Hypotheses & Active Checklist", fill=(180, 200, 220), font=font_small)
    
    curr_y = 95
    max_content_w = panel_w - 60
    
    # --- 1. HIGH-LEVEL PLAN CHECKLIST (TOP SECTION) ---
    if not plan_text:
        draw.rectangle([panel_x + 20, curr_y, panel_x + panel_w - 20, curr_y + 60], fill=(26, 30, 38), outline=(50, 60, 75), width=1)
        draw.text((panel_x + 35, curr_y + 18), "• Plan initialized dynamically", fill=(180, 180, 180), font=font_text)
        curr_y += 75
    else:
        todo_lines = [line.strip() for line in plan_text.split("\n") if line.strip()]
        
        # Limit to top 4 items to ensure ample space for Low-Level Rationale panel below
        for line in todo_lines[:4]:
            status, desc, rationale = format_todo_item(line)
            
            if status == "[x]":
                icon = "✓"
                icon_color = (0, 230, 120)  # Green
                box_bg = (22, 36, 28)
                title_color = (180, 240, 200)
            elif status == "[-]":
                icon = "➔"
                icon_color = (0, 200, 255)  # Cyan
                box_bg = (20, 34, 45)
                title_color = (200, 240, 255)
            else:
                icon = "○"
                icon_color = (180, 180, 180) # Gray
                box_bg = (26, 30, 38)
                title_color = (220, 220, 220)
                
            card_start_y = curr_y
            
            desc_lines = wrap_text(desc, font_text, max_content_w - 40, draw)
            rationale_lines = wrap_text(f"Rationale: {rationale}", font_small, max_content_w - 40, draw) if rationale else []
            
            card_h = 22 + len(desc_lines) * 22 + len(rationale_lines) * 18
            
            draw.rectangle([panel_x + 20, card_start_y, panel_x + panel_w - 20, card_start_y + card_h], fill=box_bg, outline=(50, 60, 75), width=1)
            draw.text((panel_x + 32, card_start_y + 8), icon, fill=icon_color, font=font_title)
            
            text_y = card_start_y + 8
            for dline in desc_lines:
                draw.text((panel_x + 60, text_y), dline, fill=title_color, font=font_text)
                text_y += 22
                
            for rline in rationale_lines:
                draw.text((panel_x + 60, text_y + 2), rline, fill=(150, 180, 170), font=font_small)
                text_y += 18
                
            curr_y += card_h + 10
            if curr_y > 520:
                break

    # --- 2. LOW-LEVEL ACTION & RATIONALE PANEL (BOTTOM SECTION) ---
    curr_y = max(curr_y + 10, 530)
    card_x1 = panel_x + 20
    card_x2 = panel_x + panel_w - 20
    card_y2 = canvas_h - 20 # 1060
    
    # Outer Card Box
    draw.rectangle([card_x1, curr_y, card_x2, card_y2], fill=(22, 27, 36), outline=(0, 200, 255), width=2)
    
    # Title Banner inside Card
    draw.rectangle([card_x1, curr_y, card_x2, curr_y + 45], fill=(32, 42, 58))
    draw.line([(card_x1, curr_y + 45), (card_x2, curr_y + 45)], fill=(0, 200, 255), width=1)
    draw.text((card_x1 + 15, curr_y + 10), "CURRENT STEP ACTION & RATIONALE", fill=(0, 230, 255), font=font_title)
    
    content_y = curr_y + 55
    
    # Action Badge / Line
    action_type = action_info.get("type", "frontier")
    action_id = action_info.get("id", 0)
    action_str = f"Action: Chosen {action_type.capitalize()} {action_id}"
    
    draw.rectangle([card_x1 + 15, content_y, card_x1 + 320, content_y + 30], fill=(0, 140, 75))
    draw.text((card_x1 + 25, content_y + 4), action_str, fill=(255, 255, 255), font=font_title)
    
    content_y += 42
    
    # Observation Summary
    if obs_text:
        if MAX_OBSERVATION_CHAR_LEN != -1 and len(obs_text) > MAX_OBSERVATION_CHAR_LEN:
            obs_text_display = obs_text[:MAX_OBSERVATION_CHAR_LEN].strip() + "..."
        else:
            obs_text_display = obs_text
            
        draw.text((card_x1 + 15, content_y), "Observation:", fill=(0, 200, 255), font=font_small)
        content_y += 20
        obs_lines = wrap_text(obs_text_display, font_small, max_content_w - 30, draw)
        for oline in obs_lines[:MAX_OBSERVATION_LINES]:
            draw.text((card_x1 + 15, content_y), oline, fill=(200, 220, 240), font=font_small)
            content_y += 18
        content_y += 6

    # Rationale Header & Text
    clean_reason = clean_rationale(reason_text)
    if clean_reason:
        draw.text((card_x1 + 15, content_y), "Rationale:", fill=(255, 215, 0), font=font_small)
        content_y += 20
        reason_lines = wrap_text(clean_reason, font_small, max_content_w - 30, draw)
        max_lines = (card_y2 - content_y - 10) // 18
        for rline in reason_lines[:max_lines]:
            draw.text((card_x1 + 15, content_y), rline, fill=(220, 230, 240), font=font_small)
            content_y += 18

def create_episode_video(episode_id="0", results_root="Pred-EQA/results/Pred-EQA", output_dir=None, show_reasoning=True):
    if not os.path.exists(f"{results_root}/{episode_id}/visualization") and os.path.exists(f"Pred-EQA/results/Pred-EQA-10/{episode_id}/visualization"):
        results_root = "Pred-EQA/results/Pred-EQA-10"
        
    base_dir = f"{results_root}/{episode_id}"
    if not os.path.exists(base_dir):
        print(f"Error: Episode directory {base_dir} does not exist.")
        return False
    if output_dir is None:
        output_dir = f"Pred-EQA/videos/{episode_id}"
    frames_dir = f"{output_dir}/frames"
    if os.path.exists(frames_dir):
        import shutil
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Metadata & Plan lookup
    gt_data = load_express_bench_metadata()
    model_answers = load_model_answers(results_root)
    step_plans, chosen_actions, step_reasons, step_observations = parse_episode_plans(episode_id, results_root)
    
    ep_info = gt_data.get(str(episode_id), {})
    question = ep_info.get("question", "Question unavailable.")
    expected_answer = ep_info.get("answer", "Answer unavailable.")
    model_answer = model_answers.get(str(episode_id), "Model answer unavailable.")
    
    main_canvas_w = 1920
    panel_w = 640 if show_reasoning else 0
    canvas_w = main_canvas_w + panel_w
    canvas_h = 1080
    banner_h = 180
    img_area_h = canvas_h - banner_h # 900px
    
    map_w = 880
    map_h = 880
    
    grid_cell_w = 490
    grid_cell_h = 430
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.exists(font_path):
        font_title = ImageFont.truetype(font_path, 22)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        font_small = ImageFont.truetype(font_path, 16)
    else:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()

    map_files = sorted(glob.glob(f"{base_dir}/visualization/*_map.png"))
    num_steps = len(map_files)
    is_zero_step = False
    if num_steps == 0:
        snap_files = glob.glob(f"{base_dir}/snapshot/*") + glob.glob(f"{base_dir}/chosen_snapshot/*")
        if snap_files:
            num_steps = 1
            is_zero_step = True
        else:
            print(f"Error: No visualization maps found in {base_dir}/visualization/")
            return False
        
    print(f"Generating video for Episode {episode_id} ({num_steps} steps) [Reasoning Panel: {show_reasoning}]...")

    out_video_path = f"{output_dir}/video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_video_path, fourcc, 2.0, (canvas_w, canvas_h))
    
    if not video_writer.isOpened():
        print(f"Error: Failed to open VideoWriter for path {out_video_path}")
        return False

    try:
        for step in range(num_steps):
            canvas = Image.new("RGB", (canvas_w, canvas_h), color=(25, 28, 35))
            draw = ImageDraw.Draw(canvas)
            
            # 1. Load and place Chosen Frontier / Action (Large Left Hero Panel)
            action = chosen_actions.get(step, {})
            action_type = action.get("type", "frontier")
            action_id = action.get("id", 0)
            
            is_final_step = (step == num_steps - 1)
            chosen_snap_files = sorted(glob.glob(f"{base_dir}/chosen_snapshot/snapshot_*.png"))
            
            frontier_img = None
            hero_title = f"ACTION: Chosen {action_type.capitalize()}"
            hero_color = (0, 160, 75)
            hero_outline = (0, 255, 120)

            if is_final_step and chosen_snap_files:
                frontier_img = Image.open(chosen_snap_files[0]).convert("RGB")
                hero_title = "ACTION: Chosen Visual Snapshot"
                hero_color = (200, 150, 0)
                hero_outline = (255, 215, 0)
            elif action_type == "frontier":
                chosen_file = f"{base_dir}/frontier/{step}_{action_id}.png"
                if os.path.exists(chosen_file):
                    frontier_img = Image.open(chosen_file).convert("RGB")
                else:
                    frontier_raw_files = sorted(glob.glob(f"{base_dir}/frontier/{step}_*.png"))
                    if frontier_raw_files:
                        frontier_img = Image.open(frontier_raw_files[0]).convert("RGB")
            elif action_type == "snapshot":
                if chosen_snap_files:
                    frontier_img = Image.open(chosen_snap_files[0]).convert("RGB")
            
            if frontier_img is None:
                if chosen_snap_files:
                    frontier_img = Image.open(chosen_snap_files[0]).convert("RGB")
                else:
                    frontier_img = Image.new("RGB", (map_w, map_h), color=(40, 44, 52))

            frontier_hero = frontier_img.copy()
            frontier_hero.thumbnail((map_w, map_h), Image.Resampling.LANCZOS)
            
            fx = (920 - frontier_hero.width) // 2
            fy = (img_area_h - frontier_hero.height) // 2
            canvas.paste(frontier_hero, (fx, fy))
            
            draw.rectangle([fx, fy, fx + frontier_hero.width, fy + frontier_hero.height], outline=hero_outline, width=4)
            draw.rectangle([fx, fy, fx + 260, fy + 28], fill=hero_color)
            draw.text((fx + 10, fy + 5), hero_title, fill=(255, 255, 255), font=font_small)

            # 2. Collect Snapshots for Right side grid (slots 0, 1, 2)
            snapshots = []
            snap_files = sorted(glob.glob(f"{base_dir}/snapshot/{step}-view_*.png"))
            for sfile in snap_files[:3]:
                snapshots.append(Image.open(sfile).convert("RGB"))
                
            while len(snapshots) < 3:
                empty_img = Image.new("RGB", (grid_cell_w, grid_cell_h), color=(40, 44, 52))
                edraw = ImageDraw.Draw(empty_img)
                edraw.text((grid_cell_w//3, grid_cell_h//2), "No Snapshot", fill=(140, 140, 140), font=font_text)
                snapshots.append(empty_img)

            # 3. Load Map for Right side grid (Slot 3)
            map_path = f"{base_dir}/visualization/{step}_map.png"
            if os.path.exists(map_path):
                map_img = Image.open(map_path).convert("RGB")
                slot3_title = f"Step {step} Map"
                slot3_color = (255, 255, 255)
            else:
                map_img = Image.new("RGB", (grid_cell_w, grid_cell_h), color=(240, 240, 240))
                mdraw = ImageDraw.Draw(map_img)
                mdraw.text((60, 200), "Initial State (Step 0)\nAnswered at initialization", fill=(50, 50, 50), font=font_text)
                slot3_title = f"Step {step} Map"
                slot3_color = (255, 255, 255)

            grid_positions = [
                (940, 15),                    # Cell 0: Memory Snap 0
                (940 + grid_cell_w + 20, 15),   # Cell 1: Memory Snap 1
                (940, 15 + grid_cell_h + 15),   # Cell 2: Memory Snap 2
                (940 + grid_cell_w + 20, 15 + grid_cell_h + 15) # Cell 3: Map / Chosen Snap
            ]

            # Paste Memory Snapshots
            for i, s_img in enumerate(snapshots):
                gx, gy = grid_positions[i]
                s_img_copy = s_img.copy()
                s_img_copy.thumbnail((grid_cell_w, grid_cell_h), Image.Resampling.LANCZOS)
                
                cell_bg = Image.new("RGB", (grid_cell_w, grid_cell_h), color=(35, 38, 46))
                cx = (grid_cell_w - s_img_copy.width) // 2
                cy = (grid_cell_h - s_img_copy.height) // 2
                cell_bg.paste(s_img_copy, (cx, cy))
                canvas.paste(cell_bg, (gx, gy))
                
                draw.rectangle([gx, gy, gx + 140, gy + 25], fill=(0, 0, 0, 200))
                draw.text((gx + 8, gy + 4), f"Memory Snap {i}", fill=(200, 220, 255), font=font_small)

            # Paste Map (Cell 3)
            mx, my = grid_positions[3]
            map_copy = map_img.copy()
            map_copy.thumbnail((grid_cell_w, grid_cell_h), Image.Resampling.LANCZOS)
            
            cell_bg_map = Image.new("RGB", (grid_cell_w, grid_cell_h), color=(255, 255, 255))
            mcx = (grid_cell_w - map_copy.width) // 2
            mcy = (grid_cell_h - map_copy.height) // 2
            cell_bg_map.paste(map_copy, (mcx, mcy))
            canvas.paste(cell_bg_map, (mx, my))
            
            draw.rectangle([mx, my, mx + 130, my + 25], fill=(0, 0, 0, 200))
            draw.text((mx + 8, my + 4), slot3_title, fill=slot3_color, font=font_small)

            # 4. Bottom Overlay Banner
            draw.rectangle([0, img_area_h, main_canvas_w, canvas_h], fill=(18, 22, 28))
            draw.line([(0, img_area_h), (main_canvas_w, img_area_h)], fill=(0, 255, 120), width=3)

            draw.text((30, img_area_h + 20), "Question: ", fill=(255, 215, 0), font=font_title)
            draw.text((170, img_area_h + 22), question, fill=(255, 255, 255), font=font_text)

            draw.text((30, img_area_h + 70), "Expected GT: ", fill=(0, 230, 120), font=font_title)
            draw.text((200, img_area_h + 72), expected_answer, fill=(200, 255, 220), font=font_text)

            if step == num_steps - 1:
                draw.text((30, img_area_h + 120), "Model Response: ", fill=(0, 200, 255), font=font_title)
                draw.text((240, img_area_h + 122), model_answer, fill=(255, 255, 255), font=font_text)
            else:
                draw.text((30, img_area_h + 120), "Status: ", fill=(255, 165, 0), font=font_title)
                draw.text((130, img_area_h + 122), f"Exploring... (Step {step}/{num_steps - 1})", fill=(200, 200, 200), font=font_text)

            # 5. Right-Hand Reasoning & Plan Column (if enabled)
            if show_reasoning:
                plan_text = step_plans.get(step, "")
                action_info = chosen_actions.get(step, {})
                reason_text = step_reasons.get(step, "")
                obs_text = step_observations.get(step, "")
                draw_reasoning_panel(draw, step, plan_text, action_info, reason_text, obs_text, main_canvas_w, panel_w, canvas_h, font_title, font_text, font_small)
                
            # Save frame image
            frame_path = f"{output_dir}/frames/step_{step:02d}.png"
            canvas.save(frame_path)
            
            cv_img = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
            
            num_repeats = 2
            for _ in range(num_repeats):
                video_writer.write(cv_img)
                
            # Add 4s extra buffer at the very end (total ~5s)
            if step == num_steps - 1:
                for _ in range(8): # 4 seconds at 2fps
                    video_writer.write(cv_img)
    finally:
        video_writer.release()
    print(f"Successfully generated video: {out_video_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EQA Episode Step-by-Step Visualization Video")
    parser.add_argument("--episode-id", type=str, default="0", help="Episode/Question ID to visualize (e.g. 0, 1, 10)")
    parser.add_argument("--results-root", type=str, default="Pred-EQA/results/Pred-EQA", help="Path to results root folder")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for video and frames")
    parser.add_argument("--show-reasoning", action="store_true", default=True, help="Include High-Level Plan & Reasoning column on the right side")
    parser.add_argument("--no-reasoning", action="store_false", dest="show_reasoning", help="Disable Reasoning panel and keep 1920x1080 canvas")
    args = parser.parse_args()
    
    create_episode_video(episode_id=args.episode_id, results_root=args.results_root, output_dir=args.output_dir, show_reasoning=args.show_reasoning)
