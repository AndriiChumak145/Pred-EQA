import openai
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging
from src.const import *
from qwen_vl_utils import process_vision_info
import re
import torch
import numpy as np
from src.long_term_memory import TextLongTermMemory
from src.plan_extraction_utils import extract_predictive_plan

def safe_findall(pattern, string):
    """Safe regex findall function that handles cases where string may be None."""
    if string is None:
        return []
    return re.findall(pattern, string)

def safe_strip(string):
    """Safe string strip function that handles cases where string may be None."""
    if string is None:
        return ""
    return string.strip()

VLM_CONFIG = {
    "provider": "local_qwen",
    "model": "Qwen3-VL-8B-Instruct",
    "base_url": "http://127.0.0.1:8000/v1",
    "api_key": "EMPTY",
    "rate_limit_delay": 0.0,
}

client = OpenAI(
    api_key=VLM_CONFIG["api_key"],
    base_url=VLM_CONFIG["base_url"],
    timeout=3600
)

def set_vlm_config(provider="local_qwen", model=None, base_url=None, api_key=None, rate_limit_delay=None):
    """
    Configures VLM provider (local_qwen or gemini) dynamically.
    Fully backwards compatible: defaults to local Qwen setup.
    """
    global VLM_CONFIG, client
    if provider == "gemini":
        token_file = "/home/dani/concept-scenesplat/tokens/gemini_api_key"
        file_key = open(token_file).read().strip() if os.path.exists(token_file) else None
        
        VLM_CONFIG["provider"] = "gemini"
        VLM_CONFIG["model"] = model or os.getenv("GEMINI_MODEL", "gemini-robotics-er-2-preview")
        VLM_CONFIG["base_url"] = base_url or os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        VLM_CONFIG["api_key"] = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or file_key or "EMPTY"
        VLM_CONFIG["rate_limit_delay"] = float(rate_limit_delay) if rate_limit_delay is not None else 4.0
    else:
        VLM_CONFIG["provider"] = "local_qwen"
        VLM_CONFIG["model"] = model or "Qwen3-VL-8B-Instruct"
        VLM_CONFIG["base_url"] = base_url or "http://127.0.0.1:8000/v1"
        VLM_CONFIG["api_key"] = api_key or "EMPTY"
        VLM_CONFIG["rate_limit_delay"] = 0.0

    client = OpenAI(
        api_key=VLM_CONFIG["api_key"],
        base_url=VLM_CONFIG["base_url"],
        timeout=3600
    )


# encode tensor images to base64 format
def encode_tensor2base64(img):
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def format_content(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{c[1]}",
                    },
                }
            )
    return formated_content


# send information to openai / gemini
def call_openai_api(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]

    delay = VLM_CONFIG.get("rate_limit_delay", 0.0)
    if delay > 0:
        time.sleep(delay)

    while retry_count < max_tries:
        try:
            kwargs = {
                "model": VLM_CONFIG.get("model", "Qwen3-VL-8B-Instruct"),
                "messages": message_text,
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.8,
            }
            if VLM_CONFIG.get("provider") == "local_qwen":
                kwargs["presence_penalty"] = 1.5
                kwargs["extra_body"] = {
                    "repetition_penalty": 1.0,
                    "top_k": 20,
                }
            completion = client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            wait_time = 5 * (2 ** retry_count)
            print(f"Rate limit error, waiting for {wait_time}s")
            time.sleep(wait_time)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(3)
            retry_count += 1
            continue

    return None

def call_openai_api_text(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]

    delay = VLM_CONFIG.get("rate_limit_delay", 0.0)
    if delay > 0:
        time.sleep(delay)

    while retry_count < max_tries:
        try:
            kwargs = {
                "model": VLM_CONFIG.get("model", "Qwen3-VL-8B-Instruct"),
                "messages": message_text,
                "max_tokens": 1024,
                "temperature": 1.0,
                "top_p": 1.0,
            }
            if VLM_CONFIG.get("provider") == "local_qwen":
                kwargs["presence_penalty"] = 2.0
                kwargs["extra_body"] = {
                    "repetition_penalty": 1.0,
                    "top_k": 40,
                }
            completion = client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            wait_time = 5 * (2 ** retry_count)
            print(f"Rate limit error, waiting for {wait_time}s")
            time.sleep(wait_time)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(3)
            retry_count += 1
            continue

    return None


def parse_number_list(numbers_str: str) -> list:
    """Parse number list string, e.g. '0, 1, 2' or '0,1,2'."""
    if not numbers_str:
        return []

    # Special case: if the string is ".", retain all
    if numbers_str.strip() == ".":
        return ["ALL"]  # Return list containing "ALL" instead of string itself

    # Clean up string
    clean_str = re.sub(r'[{}()\[\]]', '', numbers_str)
    clean_str = clean_str.replace(' ', '')
    # Remove trailing punctuation, e.g. periods
    clean_str = clean_str.rstrip('.,;:!?')

    # If empty after cleaning, return empty list indicating retaining empty set
    if not clean_str:
        return []

    # Split and convert to integers
    number_strs = clean_str.split(',')
    numbers = []

    for num_str in number_strs:
        num_str = num_str.strip()
        if num_str.isdigit():
            numbers.append(int(num_str))

    return numbers


def parse_retain_response(response: str, prefix: str = "Retain Snapshots") -> list:
    """
    Generic parsing function to parse the number list following "Retain X:" from model response.
    
    Args:
        response: Model response string
        prefix: Prefix, e.g. "Retain Snapshots" or "Retain Frontiers"
    
    Handles multiple formats:
    1. "Retain X: {0, 1, 2}."
    2. "Retain X: 0, 1, 2."
    3. "Retain X: 0, 1, 2, 3, 4, 5, 6, 7, 8."
    
    Strategies:
    - Find all matches and use the last one (avoid matching format specifications)
    - Support both formats with and without curly braces
    - Handle trailing periods, newlines, etc.
    """
    if not response:
        return []
    
    select_id = []
    
    # Strategy 1: Match format with curly braces "Retain X: {0, 1, 2}."
    # Use stricter pattern to ensure matching actual number list rather than format instructions
    pattern_with_braces = rf'{prefix}:\s*{{([0-9,\s]+)}}'
    matches_with_braces = safe_findall(pattern_with_braces, response)
    
    if matches_with_braces:
        # Take the last match (most likely the actual answer)
        final_match = matches_with_braces[-1].strip()
        if final_match:
            select_id = parse_number_list(final_match)
            if select_id:
                return select_id
    
    # Strategy 2: Match format without curly braces "Retain X: 0, 1, 2."
    # Improved regex: require at least one number, and not a placeholder in format instructions
    # Exclude patterns like "{i, ...}" in format instructions
    pattern_without_braces = rf'{prefix}:\s*([0-9]+(?:\s*,\s*[0-9]+)*)'
    matches_without_braces = safe_findall(pattern_without_braces, response)
    
    if matches_without_braces:
        # Take the last match (most likely the actual answer)
        # But we need to inspect the whole line because numbers might be separated by text
        lines = response.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if f'{prefix}:' in line:
                # Extract entire content after colon
                after_colon = line.split(f'{prefix}:', 1)[-1].strip()
                # Extract all numbers (including those separated by text)
                numbers = re.findall(r'\d+', after_colon)
                if numbers:
                    try:
                        select_id = [int(n) for n in numbers]
                        return select_id
                    except ValueError:
                        continue
        # If reverse search above fails, fall back to original method
        final_match = matches_without_braces[-1].strip()
        if final_match:
            select_id = parse_number_list(final_match)
            if select_id:
                return select_id
    
    # Strategy 3: If none of the above matched, try a more lenient pattern
    # Search for lines containing actual numbers (not placeholders)
    lines = response.split('\n')
    for line in reversed(lines):  # Search from back to front
        line = line.strip()
        # Ensure line contains "Retain X:" followed by numbers
        if f'{prefix}:' in line:
            # Extract content after colon
            after_colon = line.split(f'{prefix}:', 1)[-1].strip()
            # Strip potential quotes and formatting symbols
            after_colon = re.sub(r'[{}()\[\]"]', '', after_colon)
            # Try extracting all numbers (including text-separated numbers)
            numbers = re.findall(r'\d+', after_colon)
            if numbers:
                try:
                    select_id = [int(n) for n in numbers]
                    return select_id
                except ValueError:
                    continue
    
    # If all strategies fail, return empty list
    return []


def parse_retain_snapshots_response(response: str) -> list:
    """Parse Retain Snapshots response."""
    return parse_retain_response(response, "Retain Snapshots")


def parse_retain_frontiers_response(response: str) -> list:
    """Parse Retain Frontiers response."""
    return parse_retain_response(response, "Retain Frontiers")

def remove_digits(text: str) -> str:
    """Replace all digits in string with spaces."""
    return re.sub(r'\d', ' ', text)


def generate_step_summary(step_num, agent_outputs, question, step=None):
    """
    Generate a comprehensive summary for a step, covering outputs of all agents.
    Optimization: enhance summary quality, add more contextual information.
    Args:
        step_num: Step number
        agent_outputs: List of outputs from all agents in this step
        question: Current question
        step: Current step object, used to retrieve memory info
    Returns:
        Comprehensive step summary
    """
    if not agent_outputs:
        return "No agent outputs for this step."
    agent_summaries = []
    for output in agent_outputs:
        agent_type = output['agent_type']
        content = output['content']
        agent_summaries.append(f"{agent_type}: {content}")

    combined_content = "\n".join(agent_summaries)

    sys_prompt = f"""You are a concise assistant that summarizes a step in an exploration process for long-term textual memory.

Task: Summarize the step's key information useful for future reasoning. Focus on:
1. Critical environmental or spatial observations (e.g., room layout, connectivity, notable objects).
2. Progress and current status.

Constraints:
Keep under 150 words.
Be specific and forward-looking—prioritize details that won't be available in future steps (e.g., visual or spatial context).
STRICTLY NEVER mention ANY snapshot or frontier identifiers (e.g., "Snapshot 2", "Frontier 0") - these labels are step-specific and will cause confusion in later steps when the current image is no longer available.
AVOID relative directional references tied to transient views (e.g., “left of the snapshot”, “right of the frontier”). Instead, describe spatial relationships using observable objects (e.g., “the chair is next to the table”).
If no meaningful activity or observation occurred, return "No significant activity in this step."
"""

    # Add memory info to prompt
    memory_info_str = ""
    if step is not None:
        try:
            memory_info_str = format_memory_info(step, max_steps=20, outside=False)
            memory_info_str = f"\nRelevant Memory Information:\n{memory_info_str}\n"
        except Exception as e:
            memory_info_str = "\nMemory information unavailable.\n"

    contents = [(f"Step {step_num} Agent Outputs:\n{combined_content}{memory_info_str}",)]

    summary = call_openai_api_text(sys_prompt, contents)

    if summary is None:
        return combined_content[:200] + "..." if len(combined_content) > 200 else combined_content
    summary = remove_digits(summary)

    return summary.strip()


def generate_response_summary(response: str, response_type: str, question: Optional[str] = None, step: Optional[dict] = None) -> str:
    """
    Use VLM to generate response summary, filtering relevant information based on question.
    Args:
        response: Full VLM response
        response_type: Response type
        question: Current question, used to filter relevant information
        step: Current step object, used to retrieve memory info
    Returns:
        Summary text of response
    """
    if not response or response.strip() == "":
        return "No response to summarize"

    # If no question is provided, use original logic
    if not question:
        question_context = ""
    else:
        question_context = f"Question Context: {question}\n"

    sys_prompt = f"""You are a concise and helpful assistant that converts visual observations into textual long-term memory.

Task: Summarize the current visual scene clearly and briefly for future reference. Focus ONLY on:
1. Environmental details (e.g., room layout, objects, walls, doors).
2. Spatial connectivity (e.g., how rooms or areas link to each other).

Constraints:
STRICTLY NEVER mention ANY snapshot or frontier identifiers (e.g., "Snapshot 2", "Frontier 0") - these labels are step-specific and will cause confusion in later steps when the current image is no longer available.
AVOID relative directional references tied to transient views (e.g., “left of the snapshot”, “right of the frontier”). Instead, describe spatial relationships using observable objects (e.g., “the chair is next to the table”).
Describe only observable environment and connectivity.
Keep under 100 words."""
    memory_info_str = ""
    if step is not None:
        try:
            memory_info_str = format_memory_info(step)
            memory_info_str = f"\nRelevant Memory Information:\n{memory_info_str}\n"
        except Exception as e:
            memory_info_str = "\nMemory information unavailable.\n"
    contents = [(f"{response_type.upper()} Response to convert:\n{response}{memory_info_str}",)]
    summary = call_openai_api_text(sys_prompt, contents)

    if summary is None:
        return response[:200] + "..." if len(response) > 200 else response
    summary = remove_digits(summary)
    return summary.strip()


def extract_structured_output_from_response(response: str, response_type: str, question: Optional[str] = None, step: Optional[dict] = None) -> dict:
    """
    Extract structured output information from VLM response, filtering relevant information based on question.
    Args:
        response: Full VLM response
        response_type: Response type
        question: Current question, used to filter relevant information
        step: Current step object, used to retrieve memory info
    Returns:
        Structured output dictionary
    """
    response_summary = generate_response_summary(response, response_type, question, step)
    return {
        "raw_response": response,
        "reasoning": response_summary,
        "response_type": response_type
    }



def format_question(step):
    question = step["question"]
    image_goal = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"], "rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode("utf-8")

    return question, image_goal


def get_step_info(step, verbose=False):
    # 1 get question data
    question, image_goal = format_question(step)

    # 2 get step information(egocentric, frontier, snapshot)
    # 2.1 get egocentric views
    egocentric_imgs = []
    if step.get("use_egocentric_views", False):
        for egocentric_view in step["egocentric_views"]:
            egocentric_imgs.append(encode_tensor2base64(egocentric_view))

    # 2.2 get frontiers
    frontier_imgs = []
    for frontier in step["frontier_imgs"]:
        frontier_imgs.append(encode_tensor2base64(frontier))

    # 2.3 get snapshots
    snapshot_imgs, snapshot_classes = [], []
    obj_map = step["obj_map"]
    seen_classes = set()
    for i, rgb_id in enumerate(step["snapshot_imgs"].keys()):
        snapshot_img = step["snapshot_imgs"][rgb_id]
        snapshot_imgs.append(encode_tensor2base64(snapshot_img))
        snapshot_class = [obj_map[int(sid)] for sid in step["snapshot_objects"][rgb_id]]
        # remove duplicates
        snapshot_class = sorted(list(set(snapshot_class)))
        seen_classes.update(snapshot_class)
        snapshot_classes.append(snapshot_class)


    keep_index = list(range(len(snapshot_imgs)))
    return (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        keep_index,
    )


def format_manage_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=False,
        use_snapshot_class=True,
        image_goal=None,
        step=None, 
):
    sys_prompt = """Task: You are an indoor MEMORY MANAGEMENT AGENT responsible for CURATING and PRESERVING visual snapshots and spatial information collected by the embodied agent during its navigation, working in tandem with your existing TEXTUAL MEMORY and high-level plan. 

Instructions:
1. CAREFULLY analyze the information needed to answer the question, paying special attention to location details, objectives, object relationships, and any mentioned or implied attributes.
2. Review all available snapshots thoroughly and cross-reference them with your TEXTUAL MEMORY. When deciding whether to retain a snapshot, adopt a conservative approach - if there is ANY potential visual relevance to the current question or its context, it should be preserved. Specifically, retain snapshots that include:
   - Any room types or spaces that may be related to the question's context, even indirectly.
   - Adjacent or connected areas that could provide spatial clues or lead to relevant locations.
   - Partial views or incomplete perspectives of objects, appliances, or features that might be useful in reasoning.
   - Environmental or contextual cues (e.g., lighting, layout, orientation) that help establish spatial understanding or support inference.
   - Objects or categories explicitly mentioned in the question, as well as those that are semantically or functionally associated.
   - Any image that provides visual background or situational information not fully captured by text, which could aid in answering the question or reconstructing the environment.

3. MEMORY COMPACTION (Textual Redundancy Filter): To prevent critical visual clues from being overwhelmed by redundant trajectory images, you may DISCARD a snapshot ONLY IF it meets BOTH of the following conditions:
   - It is completely irrelevant to the primary question or objective (contains no target objects or contextual clues).
   - Its environmental content, spatial relationships, or navigational cues are already adequately and comprehensively described in your existing textual memory.

4. When in doubt—especially if you are unsure whether the textual memory fully captures the visual nuances of the scene—err on the side of retention. Even seemingly minor or indirect visual clues can become valuable during later stages of reasoning or path reconstruction.
"""
    content = []
    # 1. Question
    content.append((f"Question: {question}\n",))

    # 2. Memory information 
    if step is not None:
        try:
            # memory_info = format_memory_info(step)
            # memory_info = format_memory_info(step, max_steps=3)
            memory_info = format_memory_info(step, only_high_level_plan=True)
            content.append((memory_info,))
        except Exception as e:
            content.append(("Memory information unavailable.\n",))

    # 3. Snapshots display
    content.append(("Available Snapshots:\n",))
    if not snapshot_imgs:
        content.append(("No snapshots available\n",))
    else:
        for i, img in enumerate(snapshot_imgs):
            content.append((f"Snapshot {i}: ", img))
            if use_snapshot_class:
                text = ", ".join(snapshot_classes[i])
                content.append((text,))
            content.append(("\n",))

    # 3. Format specification
    text = "Output Format:\n"
    text += "1. First, think step by step and explain your reasoning clearly.\n"
    text += "2. Then, provide your final answer in the exact format: \"Retain Snapshots: {i, ...}.\""
    content.append((text,))
    

    return sys_prompt, content

def format_answer_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=False,
        use_snapshot_class=True,
        image_goal=None,
        step=None, 
):

    sys_prompt = """Task: You are an indoor agent that needs to determine if the current collected information is sufficient to answer the question.
 
Instructions:   
1. CAREFULLY analyze the information needed to answer the question, especially location, objectives, relationships, and attributes.
2. CAREFULLY analyze ALL available snapshots (total observed clues).
3. If ANY snapshot contains information needed to answer the question, output Answer.
4. If NO snapshot provides sufficient information, output Continue Exploration.
"""

    content = []
    # 1. Question
    content.append((f"Question: {question}\n",))

    # 2. Memory information 
    if step is not None:
        try:
            # memory_info = format_memory_info(step)
            memory_info = format_memory_info(step, only_high_level_plan=True)
            content.append((memory_info,))
        except Exception as e:
            content.append(("Memory information unavailable.\n",))

    # 3. Snapshots display
    content.append(("Available Snapshots:\n",))
    if not snapshot_imgs:
        content.append(("No snapshots available\n",))
    else:
        for i, img in enumerate(snapshot_imgs):
            content.append((f"Snapshot {i}: ", img))
            if use_snapshot_class:
                text = ", ".join(snapshot_classes[i])
                content.append((text,))
            content.append(("\n",))

    # 3. Format specification
    text = "Output Format:\n"
    text += "1. First, think step by step and explain your reasoning clearly.\n"
    text += "2. If answerable, provide your final answer in the exact format: \"Answer: [your concise answer] (Evidence: Snapshot [index])\"\n"
    text += "If not, use format: \"Continue Exploration\""
    content.append((text,))


    return sys_prompt, content


def format_explore_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=False,
        use_snapshot_class=True,
        image_goal=None,
        step=None, 
):
    sys_prompt = """Task: You are an indoor agent that needs to PHYSICALLY NAVIGATE through sequential frontier selections to finally find information needed for answering the question.

Instructions:
1. Analyze the question's information requirements, especially locations, objectives, relationships, and attributes. Identify target objects and their typical locations based on common sense.
2. Assess the previously observed clues to determine already explored areas and objects.
3. Given question needs and current exploration progress, choose a frontier based on the following Core Principles and constraints:
principle 1: Use common room-object relationships to infer possible locations of the target object (e.g., "refrigerator" in kitchen, "bed" in bedroom). Use typical room connections to prioritize exploration directions (e.g., kitchen is often adjacent to living room or dining room).
principle 2: If you are in an unrelated area, choose the frontier leading to a potentially relevant area.  If previously observed clues do not suggest that the relevant area has already been explored, continue exploring without stopping until you reach the relevant area.
principle 3: Balance proximity with strategic long-range exploration when clues suggest distant frontiers.
constraint 1: If you find that you are still in an irrelevant area, you can only choose a frontier and continue walking in order to reach the relevant area.
constraint 2: You can only access to unvisited areas by selecting a frontier step-by-step.
constraint 3: Keep selecting a frontier for moving until you find conclusive evidence enough to answer the question. Note that the objects mentioned in all questions are definitely available.

"""

    content = []
    # 1. Context reminder
    content.append((f"Target Question: {question}\n",))
    
    # 2. Memory information (add before images)
    if step is not None:
        try:
            # memory_info = format_memory_info(step)
            memory_info = format_memory_info(step, only_high_level_plan=True)
            content.append((memory_info,))
        except Exception as e:
            content.append(("Memory information unavailable.\n",))


    content.append(("Previously Observed Clues:\n",))
    if not snapshot_imgs:
        content.append(("No snapshots available\n",))
    else:
        for i, img in enumerate(snapshot_imgs):
            content.append(("\n", img))
        content.append(("\n",))

    # 2. Frontiers display
    content.append(("\nAvailable Exploration Directions:\n",))
    if not frontier_imgs:
        content.append(("No frontiers available\n",))  # TODO
    else:
        for i, img in enumerate(frontier_imgs):
            content.append((f"Frontier {i}: ", img))
            content.append(("\n",))
        if len(frontier_imgs) == 1:
            content.append(("Available Frontier indices: 0\n",))
        else:
            content.append((f"Available Frontier indices: 0-{len(frontier_imgs) - 1}\n",))

    # # 3. Format specification
    text = "Output Format:\n"
    text += "1. First, think step by step and explain your reasoning clearly.\n"
    text += "2. Then, provide your final answer in the exact format: \"Next Step: Frontier i\" or \"Stop Exploration\", where i is the index of the frontier you choose."
    # text += "Ensure that your answer includes at least 3 snapshot indices.\n\n"
    content.append((text,))


    return sys_prompt, content

def format_high_level_plan_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=False,
        use_snapshot_class=True,
        image_goal=None,
        step=None,  # Add step parameter for memory info
):
    sys_prompt = """Task: You are a HIGH-LEVEL EXPLORATION PLANNER AGENT responsible for devising a long-term navigation and search plan to answer the user's question. Based on the question, you must break down the goal into a sequence of high-level tasks (e.g., go to a room, find an object, observe an attribute) and output them as an ordered to-do list. This plan will guide the low-level agents in subsequent steps.

Instructions:
1. Analyze the user's question and identify its type (object recognition, attribute recognition, spatial relationship, object state, functional reasoning, world knowledge, or object localization).
2. Decompose the question into subgoals. For example:
   - Object recognition: Determine which object to find and where it is likely located.
   - Attribute recognition: Identify the object and which attribute to check.
   - Spatial understanding: Decide which locations or objects need exploration to understand their spatial arrangement.
   - Object state recognition: Determine which object's state to verify and how to observe it.
   - Functional reasoning: Identify relevant objects that demonstrate the function in question.
   - World knowledge: Use typical associations (e.g., kitchen contains a fridge) to infer where to search.
   - Object localization: Plan a search sequence for locating the object in different rooms.
3. For each subgoal, create a clear task (e.g., “Go to the kitchen”, “Find the refrigerator”, “Check the microwave's door status”).
4. Create Parallel Prediction-Based Branches for the immediate next step. For the most immediate unresolved navigation or search task (e.g., figuring out how to get to the kitchen), do not output a single generic task. Instead, generate multiple parallel prediction-based exploration branches. Formulate these as testable hypotheses based on current observations and world knowledge (e.g., instead of [ ] Find the kitchen, create [ ] Explore the frontier leading to the hallway since it may lead to the kitchen and [ ] Explore the frontier leading to the living area since it may lead to the kitchen).
5. Combine these immediate predictive branches and the remaining downstream high-level tasks into a single, cohesive, ordered to-do list. Place the parallel predictive branches at the very top as the active starting point, followed by the subsequent tasks.
6. Use the updateable checklist format for output. Mark tasks as [ ] pending, [-] in progress, or [x] completed based on what has been done so far. When agents investigate and eliminate predictive branches, mark the incorrect or dead-end branches as completed [x] with a brief inline explanation. Add new tasks immediately when they become apparent. Do not remove unfinished tasks unless they are truly irrelevant to the goal.
Core Principles:
- Before updating, always confirm which todos have been completed or invalidated since the last update.
- You may update multiple statuses in a single update (e.g., mark the previous as completed, invalidate obsolete tasks, and mark the new one as in progress).
- Dynamic Replanning: Because the environment is partially observable, new observations may completely invalidate your previous assumptions or downstream plans. If this happens, you MUST actively overhaul the plan. You are allowed to restructure, replace, or pivot away from previously planned unfinished tasks to align with the newly discovered ground truth.
- When a prediction-based branch proves incorrect (a dead-end), OR when a downstream task becomes obsolete due to a plan overhaul, mark it as [x] AND append a brief inline comment explaining why it failed or was discarded (e.g., [x] Search the kitchen ).
- Once ONE predictive branch successfully locates the target, immediately mark all other parallel predictive branches for that same goal as [x] with an explanation.
- When a completely new actionable path is discovered that pivots the entire strategy, add the new tasks immediately into the sequence and mark the old, now-irrelevant tasks as [x] with a brief explanation of the pivot.
- For regular tasks that remain relevant to the current valid strategy, only mark them as completed [x] when fully accomplished successfully (no partials, no unresolved dependencies).
Content Constraints:
STRICTLY NEVER mention ANY snapshot or frontier identifiers (e.g., "Snapshot 2", "Frontier 0") - these labels are step-specific and will cause confusion in later steps when the current image is no longer available.
AVOID relative directional references tied to transient views (e.g., “left of the snapshot”, “right of the frontier”). Instead, describe spatial relationships using observable objects (e.g., “the chair is next to the table”).
Example:
[ ] Go through the doorway into the kitchen.
[x] Explore the frontier leading to the hallway to check if it leads to the kitchen. <!-- Irrelevant; kitchen is confirmed via doorway -->
[x] Explore the frontier leading to the living area to check if it leads to the kitchen. <!-- Also irrelevant; kitchen is already identified -->
[x] Retain the view through the kitchen doorway as it leads to the target location. <!-- Already completed; serves as navigation anchor -->

""" 
    content = []
    # 1. Context reminder
    content.append((f"Target Question: {question}\n",))
    
    # 2. Memory information (add before images)
    if step is not None:
        try:
            memory_info = format_memory_info(step)
            content.append((memory_info,))
        except Exception as e:
            content.append(("Memory information unavailable.\n",))

    # # 3. Format specification
    text = '''Output Format:
1. First, think step by step and explain your reasoning clearly.
2. Always output your tasks in the following XML checklist format:
<update_todo_list>
<todos>
[ ] Pending task description
[-] In progress task description <!-- status; rationale -->
[x] Completed or pruned task description <!-- status; rationale -->
</todos>
</update_todo_list>
'''
    content.append((text,))


    return sys_prompt, content



def format_plan_manager_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=False,
        use_snapshot_class=True,
        image_goal=None,
        step=None,  
):
    sys_prompt = '''Task: You are an EXPLORATION DIRECTION MANAGEMENT AGENT responsible for STRATEGICALLY SELECTING and PRUNING potential frontiers based on observed visual snapshots.  Your goal is to eliminate directions that have BOTH OBVIOUSLY BEEN EXPLORED AND ARE IRRELEVANT to answering the question.

Instructions:
1.   CAREFULLY analyze the provided visual snapshots to identify areas that have already been explored.
2.   Determine which frontiers (exploration directions) can be safely removed because they MEET BOTH CRITERIA:
- They lead to areas ALREADY CONFIRMED AS VISITED with high certainty.
- The area or objects within them are CLEARLY UNRELATED TO THE QUESTION or its context.
3.   ONLY remove such frontiers if BOTH conditions above are MET.  If ANY DOUBT exists about either exploration status or relevance, KEEP THE FRONTIER.
4.   Retain all other frontiers, including those where there is ANY UNCERTAINTY regarding their exploration status or their relevance to the question.
5.   Maintain spatial awareness: even partially visible rooms or ambiguous paths should be preserved unless you are ABSOLUTELY CERTAIN about their irrelevance.
6.   REMEMBER, the key is to avoid deleting potentially useful information.  When in doubt, err on the side of caution and retain the frontier.
'''
    content = []
    # 1. Context reminder
    content.append((f"Target Question: {question}\n",))

    # # 2. Memory information (add before images)
    if step is not None:
        try:
            # memory_info = format_memory_info(step)
            memory_info = format_memory_info(step, only_high_level_plan=True)
            content.append((memory_info,))
        except Exception as e:
            content.append(("Memory information unavailable.\n",))

    content.append(("Previously Observed Clues:\n",))
    if not snapshot_imgs:
        content.append(("No snapshots available\n",))
    else:
        for i, img in enumerate(snapshot_imgs):
            content.append(("\n", img))
        content.append(("\n",))

    # 2. Frontiers display
    content.append(("\nAvailable Exploration Directions:\n",))
    if not frontier_imgs:
        content.append(("No frontiers available\n",))  # TODO
    else:
        for i, img in enumerate(frontier_imgs):
            content.append((f"Frontier {i}: ", img))
            content.append(("\n",))
        if len(frontier_imgs) == 1:
            content.append(("Available Frontier indices: 0\n",))
        else:
            content.append((f"Available Frontier indices: 0-{len(frontier_imgs) - 1}\n",))

    # 3. Format specification
    text = "Output Format:\n"
    text += "1. First, think step by step and explain your reasoning clearly.\n"
    text += "2. Then, provide your final answer in the exact format: \"Retain Frontiers: {i, ...}\" (retain at least 1 frontiers)."
    content.append((text,))

    return sys_prompt, content


def format_force_answer_prompt(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
    step=None,  
):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings. To answer the question, you are required to choose a Snapshot as the answer.\n"
    sys_prompt += "Definitions:\n"
    sys_prompt += "Snapshot: A focused observation of several objects. Choosing a Snapshot means that this snapshot image contains enough information for you to answer the question. "
    sys_prompt += "If you choose a Snapshot, you need to directly give an answer to the question. Your answer is mandatory and you must select one of the available Snapshots.\n"

    content = []
    # 1 first is the question
    text = f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    text = "Select the Snapshot that would help find the answer of the question.\n"
    content.append((text,))

    # 3 here is the snapshot images
    text = "The followings are all the snapshots that you can choose\n"
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No Snapshot is available\n",))
    else:
        for i in range(len(snapshot_imgs)):
            content.append((f"Snapshot {i} ", snapshot_imgs[i]))
            content.append(("\n",))

    # 5 here is the format of the answer
    text = "Please provide your answer in the following format: 'Snapshot i\n[Answer]', where i is the index of the snapshot you choose. "
    text += "Your answer is mandatory and you must select one of the available Snapshots. "
    text += "For example, if you choose the first snapshot, you can return 'Snapshot 0\nThe fruit bowl is on the kitchen counter.'. "
    text += "Note that if you choose a snapshot to answer the question, (1) you should give a direct answer that can be understood by others. Don't mention words like 'snapshot', 'on the left of the image', etc; "
    text += "(2) you can also utilize other snapshots and egocentric views to gather more information, but you should always choose one most relevant snapshot to answer the question.\n"
    content.append((text,))

    return sys_prompt, content





def get_agent_outputs_by_step_and_type(step, step_num, agent_types=None):
    """
    Optimized memory retrieval function: get outputs of specified agent types in a given step in one go.
    Args:
        step: Current step object
        step_num: Step number to retrieve
        agent_types: List of agent types to retrieve; if None, retrieves all types
    Returns:
        Dictionary where keys are agent types and values are all outputs of that type
    """
    if 'scene' not in step or step['scene'] is None:
        return {}

    # agent_execution_order = ["frontier_manager", "snapshot_manager", "answerer", "planner", "forced_answerer"]
    agent_execution_order = ["snapshot_manager", "frontier_manager", "answerer", "planner", "forced_answerer"]
    
    # If no agent types specified, use default order
    if agent_types is None:
        agent_types = agent_execution_order
    
    # Retrieve all required agent outputs at once
    agent_outputs = {}
    try:
        all_current_step_outputs = []
        for agent_type in agent_types:
            outputs = step['scene'].long_term_memory.retrieve_by_type(f"{agent_type}_output", top_k=50)
            for output in outputs:
                if output.step == step_num:
                    all_current_step_outputs.append((agent_type, output))
        
        # Organize outputs by agent type
        for agent_type, output in all_current_step_outputs:
            if agent_type not in agent_outputs:
                agent_outputs[agent_type] = []
            
            if (output.structured_decision and
                'raw_response_summary' in output.structured_decision and
                output.structured_decision['raw_response_summary']):
                agent_outputs[agent_type].append({
                    'content': output.structured_decision['raw_response_summary'],
                    'timestamp': getattr(output, 'timestamp', None),
                    'raw_output': output  # Keep raw output for later use if needed
                })
    except Exception as e:
        logging.warning(f"Error retrieving agent outputs for step {step_num}: {e}")
    
    return agent_outputs


def format_memory_info(step, max_steps=50, outside=True, only_high_level_plan=False):
    """
    Format memory info for previous N steps using step-level comprehensive summaries.
    
    Functionality:
    - Prioritize retrieving pre-generated summaries to avoid redundant generation
    - Add detailed memories of agents already executed in current step
    - Improve retrieval efficiency, add caching mechanism, and improve information organization
    - Add structured info from high-level planner (prioritize current step's; if not present, use previous step's)
    
    Args:
        step: Current step object, containing scene information
            Required fields: 'scene', 'current_step', 'question', 'current_position'
            
        max_steps: int, default 50
            Maximum number of historical steps to display. Controls how many historical step summaries are shown in "Previous Steps Summary".
            Larger values show more history, but may make the prompt too long.
            
        outside: bool, default True
            Whether to include historical step summaries.
            - True: Full output, including "Previous Steps Summary" section (showing previous max_steps historical steps)
            - False: Exclude historical step summaries; only show current step info and high-level plan
            Mainly used for passing information between agents within the current step, avoiding duplicate historical info.
            
        only_high_level_plan: bool, default False
            Whether to only return high-level plan information.
            - True: Only return "High-Level Plan" section, ignoring current step progress and historical summaries
            - False: Return complete memory info (historical summaries included depending on outside parameter)
            Suitable for scenarios where only planning information is needed, significantly reducing prompt length.
    
    Returns:
        str: Formatted memory information string containing the following sections (controlled by arguments):
            1. High-Level Plan: Task planning list from current or previous step (if present)
            2. Current Step Progress: Execution status of agents in current step (only when only_high_level_plan=False)
            3. Previous Steps Summary: Historical step summaries (only when outside=True and only_high_level_plan=False)
            
        If no available info, returns corresponding prompt message.
    """
    if 'scene' not in step or step['scene'] is None:
        return "No scene information available for memory retrieval.\n"

    try:
        memory_info = []
        current_step = step.get('current_step', 0)
        question = step.get('question', '')

        # Define agent execution order
        # agent_execution_order = ["frontier_manager", "snapshot_manager", "answerer", "planner", "forced_answerer"]
        agent_execution_order = ["snapshot_manager", "frontier_manager", "answerer", "planner", "forced_answerer"]

        # 1. Retrieve pre-generated step summaries (for previous steps and current step if available)
        step_summaries = {}
        try:
            # Optimization: retrieve all needed step_summary_outputs at once to reduce repeated calls
            summary_outputs = step['scene'].long_term_memory.retrieve_by_type("step_summary_output", top_k=50)
            if summary_outputs:
                for output in summary_outputs:
                    if (output.structured_decision and
                        'raw_response_summary' in output.structured_decision and
                        output.structured_decision['raw_response_summary']):
                        step_summaries[output.step] = output.structured_decision['raw_response_summary']
        except Exception as e:
            logging.warning(f"Error retrieving step summaries: {e}")

        # 2. Retrieve high-level planner output
        high_level_plan_info = None
        try:
            # First check if current step has high-level plan
            current_planner_outputs = step['scene'].long_term_memory.retrieve_by_step_and_type(current_step, "high_level_planner_output")
            if current_planner_outputs:
                # Use latest high-level plan
                latest_planner_output = current_planner_outputs[-1]
                if (latest_planner_output.structured_decision and
                    'todo_list' in latest_planner_output.structured_decision):
                    high_level_plan_info = {
                        'step': current_step,
                        'todo_list': latest_planner_output.structured_decision['todo_list'],
                        'raw_output': latest_planner_output.structured_decision.get('raw_response', '')
                    }
            else:
                # If current step has none, check previous step
                if current_step > 0:
                    previous_planner_outputs = step['scene'].long_term_memory.retrieve_by_step_and_type(current_step - 1, "high_level_planner_output")
                    if previous_planner_outputs:
                        # Use latest high-level plan from previous step
                        latest_planner_output = previous_planner_outputs[-1]
                        if (latest_planner_output.structured_decision and
                            'todo_list' in latest_planner_output.structured_decision):
                            high_level_plan_info = {
                                'step': current_step - 1,
                                'todo_list': latest_planner_output.structured_decision['todo_list'],
                                'raw_output': latest_planner_output.structured_decision.get('raw_response', '')
                            }
        except Exception as e:
            logging.warning(f"Error retrieving high-level planner info: {e}")

        # 3. Optimization: use dedicated function to retrieve all agent outputs for current step
        current_step_agents = {}
        try:
            # Use optimized memory retrieval function
            agent_outputs_by_type = get_agent_outputs_by_step_and_type(step, current_step, agent_execution_order)
            
            # Add latest output of each agent type to current_step_agents
            for agent_type in agent_execution_order:
                if agent_type in agent_outputs_by_type and agent_outputs_by_type[agent_type]:
                    # Use latest output
                    latest_output = agent_outputs_by_type[agent_type][-1]
                    current_step_agents[agent_type] = latest_output
        except Exception as e:
            logging.warning(f"Error retrieving current step agents: {e}")

        # 4. Format output - optimize information structure to highlight most important info
        if not step_summaries and not current_step_agents and not high_level_plan_info:
            return "No memory available.\n"

        # Prioritize showing high-level plan info, if present
        if high_level_plan_info:
            memory_info.append("High-Level Plan:\n")
            memory_info.append(f"- Plan from Step {high_level_plan_info['step']}:\n")
            for task in high_level_plan_info['todo_list']:
                status = task.get('status', 'unknown')
                task_desc = task.get('task', '')
                memory_info.append(f" * [{status}] {task_desc}\n")
            memory_info.append("\n")
        
        # If only high-level plan info is needed, return directly
        if only_high_level_plan:
            if high_level_plan_info:
                return "".join(memory_info)
            else:
                return "No high-level plan available.\n"

        # Then show current step agent info, as it is most relevant
        if current_step_agents:
            memory_info.append("Current Step Progress:\n")
            for agent_type in agent_execution_order:
                if agent_type in current_step_agents:
                    agent_info = current_step_agents[agent_type]
                    agent_name = agent_type.replace('_', ' ').title()
                    memory_info.append(f"- {agent_name}: {agent_info['content']}\n")
            memory_info.append("\n")

        # Then show previous steps (using pre-generated summaries)
        if outside:
            memory_info.append("Previous Steps Summary:\n")
            previous_steps = sorted([s for s in step_summaries.keys() if s < current_step], reverse=True)
            recent_steps = previous_steps[:max_steps]
            if recent_steps:
                for step_num in recent_steps:
                    summary = step_summaries[step_num]
                    memory_info.append(f"Step {step_num}: {summary}\n\n")
            else:
                memory_info.append("(No previous step summaries available)\n\n")

        return "".join(memory_info)

    except Exception as e:
        logging.error(f"Error in format_memory_info: {str(e)}")
        return f"Error retrieving memory information: {str(e)}\n"





def explore_step(step, cfg, verbose=False):
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        snapshot_id_mapping,
    ) = get_step_info(step, verbose)
    ##################################################
    ### -1  Snapshot Manager: manage the snapshots ###
    ##################################################
    snapshot_select_id = list(range(len(snapshot_imgs)))
    numbers_int = None  
    if len(snapshot_imgs) > 3:
        sys_prompt, content = format_manage_prompt(
            question,
            egocentric_imgs,
            frontier_imgs,
            snapshot_imgs,
            snapshot_classes,
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            step=step,
        )

        if verbose:
            logging.info(f"Input prompt:")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)

        full_response = call_openai_api(sys_prompt, content)
        logging.info(f"MANAGER: {full_response}")
        snapshot_select_id = parse_retain_snapshots_response(full_response)
        if snapshot_select_id:
            snapshot_select_id = [i for i in snapshot_select_id if 0 <= i < len(snapshot_imgs)]
            if not snapshot_select_id:
                logging.warning(f"All parsed snapshot indices were out of bounds, keeping no snapshots")
        else:
            logging.info(f"No snapshots selected from response (empty result), keeping no snapshots")
        
        logging.info(f"Snapshot selection parsed: keep {snapshot_select_id} out of {len(snapshot_imgs)}")
                
        if 'scene' in step and step['scene'] is not None:
            try:
                structured_output = extract_structured_output_from_response(full_response or "", "frontier_manager", question)

                current_position = step.get('current_position', [0, 0])
                position = np.array(current_position)
                
                if hasattr(step['scene'], 'text_memory_system'):
                    snapshot_descriptions = {}
                    for i in range(len(snapshot_imgs)):
                        snapshot_descriptions[f"snapshot_{i}"] = f"Snapshot {i} from step {step.get('current_step', 0)}"
                    
                    actual_step = step.get('current_step', 0)
                    
                    step['scene'].text_memory_system.record_structured_agent_output(
                        step=actual_step,
                        agent_type="snapshot_manager",
                        structured_output=structured_output,
                        raw_response=full_response or "",
                        position=position
                    )
                
                    all_snapshot_outputs = step['scene'].long_term_memory.retrieve_by_type("snapshot_manager_output", top_k=50)
                    if all_snapshot_outputs and len(all_snapshot_outputs) > 1: 
                        logging.info(f"Snapshot Manager History (All Steps):")
                        for output in all_snapshot_outputs[-5:]:  
                            logging.info(f" - Step {output.step}: {output.content[:100]}...") 
                                
                    logging.info(f"=== End Snapshot Manager Update ===")
            except Exception as e:
                logging.error(f"Error recording snapshot manager decision to memory: {e}")
        else:
            logging.warning(f"Step does not have scene attribute, skipping memory recording for snapshot manager. Scene in step: {'scene' in step}, scene value: {step.get('scene', 'Not found')}")
        numbers_int = set(range(len(snapshot_imgs))) - set(snapshot_select_id)

        logging.info(numbers_int)
        try:
            filtered_imgs = [img for i, img in enumerate(snapshot_imgs) if i not in numbers_int]
            filtered_cls = [cls for i, cls in enumerate(snapshot_classes) if i not in numbers_int]
            snapshot_id_mapping = [snapshot_id_mapping[i] for i in snapshot_select_id]
            snapshot_imgs = filtered_imgs  
            snapshot_classes = filtered_cls
        except Exception as e:
            logging.info(f"Error in exclude snapshots: {numbers_int}")
            logging.info(e)
    else:
        logging.info("could not find {}")
        # filtered_imgs = []
        # filtered_cls = []
        logging.info('there is no useful snapshots')
        full_response = 'continue exploration'


    ##################################################
    ### 0  Frontier Manager: manage the frontiers ###
    ##################################################
    frontier_select_id = list(range(len(frontier_imgs)))
    if len(frontier_imgs) > 1:
        sys_prompt, content = format_plan_manager_prompt(
            question,
            egocentric_imgs,
            frontier_imgs,
            snapshot_imgs,
            snapshot_classes,
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            step=step,
        )
        if verbose:
            logging.info(f"Input prompt:")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)

        full_response = call_openai_api(sys_prompt, content)
        logging.info(f"PLAN MANAGER: {full_response}")
        
        if full_response is None:
            logging.info("VLM response is None, using default frontier selection")
            frontier_select_id = list(range(len(frontier_imgs)))
        else:
            frontier_select_id = parse_retain_frontiers_response(full_response)
            if frontier_select_id:
                frontier_select_id = [i for i in frontier_select_id if 0 <= i < len(frontier_imgs)]
                if not frontier_select_id:
                    frontier_select_id = list(range(len(frontier_imgs)))
                    logging.warning(f"All parsed frontier indices were out of bounds, falling back to keep all {len(frontier_imgs)} frontiers")
            else:
                frontier_select_id = list(range(len(frontier_imgs)))
                logging.warning(f"Failed to parse frontier selection from response or empty result, falling back to keep all {len(frontier_imgs)} frontiers")
                    
            if 'scene' in step and step['scene'] is not None:
                try:
                    structured_output = extract_structured_output_from_response(full_response or "", "frontier_manager", question)
                    
                    current_position = step.get('current_position', [0, 0])
                    position = np.array(current_position)
                    
                    if hasattr(step['scene'], 'text_memory_system'):
                        frontier_descriptions = {}
                        for i in range(len(frontier_imgs)):
                            frontier_descriptions[i] = f"Frontier {i} at step {step.get('current_step', 0)}"
                        
                        actual_step = step.get('current_step', 0)
                        
                        step['scene'].text_memory_system.record_structured_agent_output(
                            step=actual_step,
                            agent_type="frontier_manager",
                            structured_output=structured_output,
                            raw_response=full_response or "",
                            position=position
                        )
                        
                        all_frontier_outputs = step['scene'].long_term_memory.retrieve_by_type("frontier_manager_output", top_k=50)
                        if all_frontier_outputs and len(all_frontier_outputs) > 1:  
                            logging.info(f"Frontier Manager History (All Steps):")
                            for output in all_frontier_outputs[-5:]:  
                                logging.info(f" - Step {output.step}: {output.content[:100]}...") 
                                    
                        logging.info(f"=== End Frontier Manager Update ===")
                except Exception as e:
                    logging.error(f"Error recording frontier manager decision to memory: {e}")
            else:
                logging.warning(f"Step does not have scene attribute, skipping memory recording for frontier manager. Scene in step: {'scene' in step}, scene value: {step.get('scene', 'Not found')}")

        # TODO if retain
        frontier_out_id = set(range(len(frontier_imgs))) - set(frontier_select_id)
        logging.info(frontier_select_id)
        logging.info(frontier_out_id)
        try:
            filtered_imgs = [img for i, img in enumerate(frontier_imgs) if i not in frontier_out_id]
            frontier_imgs = filtered_imgs  # TODO
    
        except Exception as e:
            logging.info(f"Error in exclude frontier: {frontier_out_id}")
            logging.info(e)



    ##################################################
    ### 1  Answerer: check enough info to answer? ###
    ##################################################
    if snapshot_imgs:
        sys_prompt, content = format_answer_prompt(
            question,
            egocentric_imgs,
            frontier_imgs,
            snapshot_imgs,  
            snapshot_classes,  
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            step=step,
        )

        if verbose:
            logging.info(f"Input prompt:")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)
        full_response = call_openai_api(sys_prompt, content)
        logging.info(f"ANSWERER: {full_response}")
        
        if 'scene' in step and step['scene'] is not None:
            try:
                structured_output = extract_structured_output_from_response(full_response or "", "answerer", question)
                
                current_position = step.get('current_position', [0, 0])
                position = np.array(current_position)
                
                if hasattr(step['scene'], 'text_memory_system'):
                    snapshot_descriptions = {}
                    for i in range(len(snapshot_imgs)):
                        snapshot_descriptions[f"snapshot_{i}"] = f"Snapshot {i} from step {step.get('current_step', 0)}"
                    
                    actual_step = step.get('current_step', 0)
                    
                    # Record structured output; this automatically handles internal logic
                    step['scene'].text_memory_system.record_structured_agent_output(
                        step=actual_step,
                        agent_type="answerer",
                        structured_output=structured_output,
                        raw_response=full_response or "",
                        position=position
                    )
                
                    all_answerer_outputs = step['scene'].long_term_memory.retrieve_by_type("answerer_output", top_k=50)
                    if all_answerer_outputs and len(all_answerer_outputs) > 1: 
                        logging.info(f"Answerer History (All Steps):")
                        for output in all_answerer_outputs[-5:]:  
                            logging.info(f" - Step {output.step}: {output.content[:100]}...") 

                    logging.info(f"=== End Answerer Update ===")
            except Exception as e:
                logging.error(f"Error recording answerer decision to memory: {e}")
        else:
            logging.warning(f"Step does not have scene attribute, skipping memory recording for answerer. Scene in step: {'scene' in step}, scene value: {step.get('scene', 'Not found')}")
    else:
        logging.info('there is no useful snapshots')
        full_response = 'continue exploration'

    ##################################################
    ### 2  Planner: if [can not answer] & [frontier_imgs] is not none --> continue explore ###
    ##################################################
    if 'continue exploration' in safe_strip(full_response).lower():
        logging.info('###### high-level plan ######')
        sys_prompt, content = format_high_level_plan_prompt(
            question,
            egocentric_imgs,
            frontier_imgs,
            snapshot_imgs,
            snapshot_classes,
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            step=step,
        )
        if verbose:
            logging.info(f"Input prompt:")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)
        full_response = call_openai_api_text(sys_prompt, content)
        full_response = safe_strip(full_response)
        logging.info(f"{full_response}")
        
        try:
            todo_list = extract_predictive_plan(full_response)
            if todo_list:
                logging.info(f"Extracted {len(todo_list)} tasks from high-level planner:")
                for i, task in enumerate(todo_list):
                    status = task.get('status', 'unknown')
                    task_desc = task.get('task', '')
                    logging.info(f"  {i+1}. [{status}] {task_desc}")
                
                if 'scene' in step and step['scene'] is not None and hasattr(step['scene'], 'text_memory_system'):
                    current_position = step.get('current_position', [0, 0])
                    position = np.array(current_position)
                    actual_step = step.get('current_step', 0)
                    
                    structured_output = {
                        "raw_response": full_response,
                        "reasoning": f"High-level plan with {len(todo_list)} tasks extracted",
                        "response_type": "high_level_plan",
                        "structured_output": {
                            "todo_list": todo_list
                        }
                    }
                    
                    step['scene'].text_memory_system.record_structured_agent_output(
                        step=actual_step,
                        agent_type="high_level_planner",
                        structured_output=structured_output,
                        raw_response=full_response,
                        position=position
                    )
                    logging.info(f"High-level plan with todo list recorded to memory for step {actual_step}")
        except Exception as e:
            logging.error(f"Error extracting todo list from high-level planner output: {e}")
        
        logging.info('###### continue exploration ######')
        sys_prompt, content = format_explore_prompt(
            question,
            egocentric_imgs,
            frontier_imgs,
            snapshot_imgs,
            snapshot_classes,
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            step=step,
        )
  
        if verbose:
            logging.info(f"Input prompt:")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)

        full_response = call_openai_api(sys_prompt, content)
        full_response = safe_strip(full_response)
        if full_response is None:
            response = "stop exploration"
            reason = "No response from model"
        else:
            lines = full_response.strip().split('\n')
            answer_pattern = re.compile(
                r'(?:next\s+step\s*:\s*frontier\s+(\d+)|stop\s+exploration)',
                re.IGNORECASE
            )
            
            answer_line_index = None
            answer_match = None
            
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                match = answer_pattern.search(line)  
                if match:
                    answer_line_index = i
                    answer_match = match
                    break

            if answer_match:
                if answer_match.group(1):  # Frontier case
                    frontier_index = answer_match.group(1)
                    response = f"frontier {frontier_index}"
                else:  # Stop Exploration case
                    response = "stop exploration"
                reason = '\n'.join(lines[:answer_line_index]).strip() or "No reasoning provided."
            else:
                found_response = False
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if 'next step: frontier' in line_lower:
                        # Extract number after frontier
                        frontier_match = re.search(r'frontier\s+(\d+)', line_lower)
                        if frontier_match:
                            response = f"frontier {frontier_match.group(1)}"
                            reason = '\n'.join(lines[:i]).strip() or "No reasoning provided."
                            found_response = True
                            break
                    elif 'stop exploration' in line_lower:
                        response = "stop exploration"
                        reason = '\n'.join(lines[:i]).strip() or "No reasoning provided."
                        found_response = True
                        break
                
                if not found_response:
                    response = "stop exploration"
                    reason = full_response  # Or record error

            response = response.lower().strip()

        if 'scene' in step and step['scene'] is not None:
            try:
                structured_output = extract_structured_output_from_response(full_response or "", "planner", question)
                
                current_position = step.get('current_position', [0, 0])
                position = np.array(current_position)
                
                if hasattr(step['scene'], 'text_memory_system'):
                    snapshot_descriptions = {}
                    for i in range(len(snapshot_imgs)):
                        snapshot_descriptions[f"snapshot_{i}"] = f"Snapshot {i} from step {step.get('current_step', 0)}"
                    
                    frontier_descriptions = {}
                    for i in range(len(frontier_imgs)):
                        frontier_descriptions[i] = f"Frontier {i} at step {step.get('current_step', 0)}"
                    
                    actual_step = step.get('current_step', 0)
                    
                    step['scene'].text_memory_system.record_structured_agent_output(
                        step=actual_step,
                        agent_type="planner",
                        structured_output=structured_output,
                        raw_response=full_response or "",
                        position=position
                    )
                    
                    all_planner_outputs = step['scene'].long_term_memory.retrieve_by_type("planner_output", top_k=50)
                    if all_planner_outputs and len(all_planner_outputs) > 1:  
                        logging.info(f"Planner History (All Steps):")
                        for output in all_planner_outputs[-5:]: 
                            logging.info(f"  - Step {output.step}: {output.content[:100]}...")  # Limit length
                    
                    logging.info(f"=== End Planner Update ===")
            except Exception as e:
                logging.error(f"Error recording planner decision to memory: {e}")
        else:
            logging.warning(f"Step does not have scene attribute, skipping memory recording for planner. Scene in step: {'scene' in step}, scene value: {step.get('scene', 'Not found')}")
        
        if 'scene' in step and step['scene'] is not None:
            try:
                current_position = step.get('current_position', [0, 0])
                position = np.array(current_position)
                actual_step = step.get('current_step', 0)
                question = step.get('question', '')

                # Check if summary for this step is already recorded to prevent duplicates
                existing_summaries = step['scene'].long_term_memory.retrieve_by_type("step_summary_output", top_k=10)
                already_recorded = any(output.step == actual_step for output in existing_summaries)

                if already_recorded:
                    logging.info(f"Step {actual_step} summary already recorded, skipping...")
                else:
                    # Collect outputs of all agents for current step
                    step_agents = []
                    # agent_execution_order = ["frontier_manager", "snapshot_manager", "answerer", "planner", "forced_answerer"]
                    agent_execution_order = ["snapshot_manager", "frontier_manager", "answerer", "planner", "forced_answerer"]

                    try:
                        agent_outputs_by_type = get_agent_outputs_by_step_and_type(step, actual_step, agent_execution_order)
                        
                        # Add all agent outputs to step_agents list
                        for agent_type in agent_execution_order:
                            if agent_type in agent_outputs_by_type:
                                for output in agent_outputs_by_type[agent_type]:
                                    step_agents.append({
                                        'agent_type': agent_type,
                                        'content': output['content']
                                    })
                    except Exception as e:
                        logging.warning(f"Error retrieving {agent_type} outputs for step {actual_step}: {e}")

                    if step_agents:
                        logging.info(f"Generating step summary for step {actual_step} with {len(step_agents)} agent outputs")

                        # Generate step summary
                        step_summary = generate_step_summary(actual_step, step_agents, question)

                        if step_summary and len(step_summary.strip()) > 10:  # Ensure summary is non-empty and meaningful
                            # Record to long-term memory
                            structured_output = {
                                "raw_response": step_summary,
                                "reasoning": step_summary,
                                "response_type": "step_summary"
                            }

                            step['scene'].text_memory_system.record_structured_agent_output(
                                step=actual_step,
                                agent_type="step_summary",
                                structured_output=structured_output,
                                raw_response=step_summary,
                                position=position
                            )

                            logging.info(f"Successfully recorded step {actual_step} summary: {step_summary[:100]}...")
                        else:
                            logging.warning(f"Generated step summary is too short or empty for step {actual_step}")
                    else:
                        logging.info(f"No agent outputs found for step {actual_step}, skipping summary generation")

            except Exception as e:
                logging.error(f"Error generating and recording step summary for step {step.get('current_step', 'unknown')}: {e}")


            logging.info(f"response: {response}")
            logging.info(f"reason: {reason}")

            try:
                response_parts = response.split(" ")
                choice_type = response_parts[0]
                choice_id = response_parts[1] if len(response_parts) > 1 else None
                response_valid = False

                if (
                        choice_type == "frontier"
                        and choice_id is not None
                        and choice_id.isdigit()
                        and 0 <= int(choice_id) < len(frontier_imgs)
                ):
                    try:
                        choice_id = frontier_select_id[int(choice_id)]
                        response = choice_type + ' ' + str(choice_id)
                        response_valid = True
                    except (ValueError, IndexError):
                        logging.info(f"Error in frontier selection: choice_id={choice_id}, frontier_select_id={frontier_select_id}")
                        response_valid = False

                if 'stop' in response.lower():
                # if 'stop' in reason.lower() or 'stop' in response.lower():
                    logging.info(f"######  stop exploring  #######")
                    response_valid = False

                if response_valid:
                    final_response = response
                    final_reason = reason
                    return final_response, snapshot_id_mapping, final_reason, len(snapshot_imgs), numbers_int if numbers_int else None

            except Exception as e:
                logging.info(f"Error in splitting response: {response}")
                logging.info(e)
                if 'stop exploration' in response.lower():
                    response_valid = False
                else:
                    try:
                        response_parts = response.split(" ")
                        if len(response_parts) >= 2 and response_parts[0] == "frontier":
                            choice_type = response_parts[0]
                            choice_id = response_parts[1]
                            if choice_id.isdigit() and 0 <= int(choice_id) < len(frontier_imgs):
                                choice_id = frontier_select_id[int(choice_id)]
                                response = choice_type + ' ' + str(choice_id)
                                response_valid = True
                    except (ValueError, IndexError):
                        response_valid = False

    ##################################################
    ### 3  Forced Answerer: if [no need to explore] or [no frontier_imgs to explore]
    ##################################################
    logging.info(f"######  force a answer  #######")
    sys_prompt, content = format_force_answer_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs, # filtered_imgs,  # snapshot_imgs,
        snapshot_classes, # filtered_cls,  # snapshot_classes,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
        step=step,
    )

    if verbose:
        logging.info(f"Input prompt:")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    full_response = call_openai_api(sys_prompt, content)
    full_response = safe_strip(full_response)
    if "\n" in full_response:
        full_response_list = full_response.split("\n")
        response, reason = full_response_list[0], full_response_list[-1]
        response, reason = safe_strip(response), safe_strip(reason)
    else:
        response = full_response
        reason = ""
    response = response.lower()

    logging.info(f"response: {response}")
    logging.info(f"reason: {reason}")

    if 'stop' in safe_strip(response).lower() and len(response.split(" ")) > 1:
        try:
            choice_type, choice_id = response.split(" ")
            choice_id = snapshot_select_id[int(choice_id)]
        except (ValueError, IndexError):
            logging.info(f"Error in snapshot selection: response={response}, snapshot_select_id={snapshot_select_id}")
            choice_type = "stop"
            choice_id = ""
        response = choice_type + ' ' + str(choice_id)

    final_response = response
    final_reason = reason

    return final_response, snapshot_id_mapping, final_reason, len(snapshot_imgs), numbers_int if numbers_int else None

