"""
Step 2: Use GPT-4o to evaluate and score Blind LLM answers.
Read answers generated in Step 1, use prompt from evaluation.txt, generate scores.
"""

import json
import os
import time
from openai import OpenAI
from typing import Optional
import re

# Initialize OpenAI client
client = OpenAI(
    api_key='',
    base_url=''
)

def load_prompt(prompt_file):
    """Load prompt file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def load_answers(answers_file):
    """Load answers data."""
    with open(answers_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def call_gpt4o(system_prompt, user_prompt, max_retries=5):
    """Call GPT-4o API."""
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call error (attempt {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(3)
            else:
                print(f"Max retries reached, skipping this question")
                return None
    
    return None

def extract_system_and_user_prompt(prompt_content):
    """Extract system and user parts from prompt file."""
    lines = prompt_content.strip().split('\n')
    
    system_prompt = ""
    user_prompt = ""
    current_section = None
    
    for line in lines:
        if line.strip() == "[system]:":
            current_section = "system"
            continue
        elif line.strip() == "[user]:":
            current_section = "user"
            continue
        
        if current_section == "system":
            system_prompt += line + "\n"
        elif current_section == "user":
            user_prompt += line + "\n"
    
    return system_prompt.strip(), user_prompt.strip()

def parse_scores(response):
    """
    Extract two scores from GPT-4o response.
    Expected format: "0.5, 3" or "1, 5", etc.
    Returns: (image_alignment_score, accuracy_score)
    """
    if not response:
        return None, None
    
    # Try multiple regex patterns to extract scores
    patterns = [
        r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)',  # Matches "0.5, 3" or "1, 5"
        r'(\d+\.?\d*)\s+(\d+\.?\d*)',      # Matches "0.5 3"
        r'(\d+\.?\d*).*?(\d+\.?\d*)',      # Lenient match
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            try:
                score1 = float(matches[0][0])
                score2 = float(matches[0][1])
                return score1, score2
            except (ValueError, IndexError):
                continue
    
    print(f"Warning: Failed to parse scores from response: {response}")
    return None, None

def evaluate_blind_llm_answers(answers_file, prompt_file, output_metrics_file):
    """Evaluate Blind LLM answers."""
    
    print("Loading answers data...")
    answers = load_answers(answers_file)
    print(f"Answers data loaded, {len(answers)} total answers")
    
    print("Loading evaluation prompt...")
    prompt_content = load_prompt(prompt_file)
    system_prompt, user_prompt_template = extract_system_and_user_prompt(prompt_content)
    print("Prompt loaded successfully")
    
    metrics = {}
    detailed_results = []
    
    for idx, item in enumerate(answers):
        question = item['question']
        answer = item['answer']
        response = item['blind_llm_response']
        question_id = item['question_id']
        
        print(f"\nEvaluating question {idx + 1}/{len(answers)} (ID: {question_id})")
        print(f"Question: {question}")
        print(f"Ground truth answer: {answer}")
        print(f"Model response: {response}")
        
        # Construct user prompt for evaluation
        # Note: Because this is Blind LLM with no image, Image alignment score should be 0
        # But we still evaluate following the original prompt format
        eval_user_prompt = f"{user_prompt_template}\n"
        eval_user_prompt += f"Question: {question}\n"
        eval_user_prompt += f"Answer: {answer}\n"
        eval_user_prompt += f"Response: {response}\n"
        eval_user_prompt += f"Image: [No image provided - Blind LLM]\n"
        
        # Call GPT-4o for evaluation
        eval_response = call_gpt4o(system_prompt, eval_user_prompt)
        
        if eval_response:
            print(f"Evaluation response: {eval_response}")
            
            # Parse scores
            image_score, accuracy_score = parse_scores(eval_response)
            
            if image_score is not None and accuracy_score is not None:
                print(f"Image alignment score: {image_score}, Accuracy score: {accuracy_score}")
                
                # Save to metrics dictionary (using question_id as key)
                metrics[str(question_id)] = accuracy_score
                
                # Save detailed results
                detailed_results.append({
                    "question_id": question_id,
                    "question": question,
                    "answer": answer,
                    "response": response,
                    "image_alignment_score": image_score,
                    "accuracy_score": accuracy_score,
                    "evaluation_response": eval_response
                })
            else:
                print("Score parsing failed")
                detailed_results.append({
                    "question_id": question_id,
                    "question": question,
                    "answer": answer,
                    "response": response,
                    "image_alignment_score": None,
                    "accuracy_score": None,
                    "evaluation_response": eval_response,
                    "error": "Score parsing failed"
                })
        else:
            print("Evaluation failed")
            detailed_results.append({
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "response": response,
                "error": "Evaluation API call failed"
            })
        
        # Save intermediate results every 10 questions
        if (idx + 1) % 10 == 0:
            print(f"\nSaving intermediate results...")
            with open(output_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            detailed_output = output_metrics_file.replace('.json', '-detailed.json')
            with open(detailed_output, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    # Final save
    print(f"\nSaving final results to {output_metrics_file}...")
    with open(output_metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    detailed_output = output_metrics_file.replace('.json', '-detailed.json')
    with open(detailed_output, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    # Compute summary statistics
    valid_scores = [r['accuracy_score'] for r in detailed_results if r.get('accuracy_score') is not None]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"\nEvaluation complete!")
        print(f"Successfully evaluated: {len(valid_scores)}/{len(answers)}")
        print(f"Average accuracy score: {avg_score:.2f}")
        print(f"Scores saved to: {output_metrics_file}")
        print(f"Detailed results saved to: {detailed_output}")
    else:
        print("\nWarning: No successfully evaluated results")

