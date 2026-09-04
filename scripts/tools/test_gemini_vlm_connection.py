import os
import sys
from pathlib import Path
import argparse
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def test_connection(model_name="gemini-robotics-er-2-preview"):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.")
        print("Please export your API key before running:")
        print("  export GEMINI_API_KEY='your_api_key_here'")
        sys.exit(1)

    print(f"Initializing OpenAI client for Gemini API...")
    print(f"Target model: {model_name}")
    print(f"Base URL: https://generativelanguage.googleapis.com/v1beta/openai/")

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    prompt = "Hello! You are an embodied AI agent assisting with navigation. Please acknowledge this test message briefly in 1 sentence."

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful robotic navigation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=128,
            temperature=0.7
        )
        response_text = completion.choices[0].message.content
        print("\nSUCCESS! Received response from Gemini API:")
        print("-" * 50)
        print(response_text)
        print("-" * 50)
    except Exception as e:
        print(f"\nERROR: Failed to query Gemini API model '{model_name}': {e}")
        print("Fallback tip: If 'gemini-robotics-er-2-preview' is restricted on your tier, try testing with 'gemini-2.0-flash' or 'gemini-1.5-flash'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-robotics-er-2-preview", type=str, help="Gemini model name")
    args = parser.parse_args()
    test_connection(args.model)
