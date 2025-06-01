import openai
import re
import numpy as np
from tqdm import tqdm
import os
import math

from dotenv import load_dotenv
load_dotenv()

# ========== Configuration ==========
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # or replace with your key directly

PROMPT = "Describe a moment you felt proud of yourself. Ensure your response is 130-150 characters including spaces."
MODEL = "gpt-3.5-turbo"
N_SAMPLES = 5
CHAR_COUNT_MIN = 130
CHAR_COUNT_MAX = 150

# ========== Reward Function ==========
# binary reward function
# def compute_reward(text: str, min_chars: int, max_chars: int) -> float:
#     char_count = len(text)
#     return 1.0 if min_chars <= char_count <= max_chars else 0.0

# exponential penalty for deviation
def compute_reward(text: str, min_chars: int, max_chars: int) -> float:
    char_count = len(text)
    target = (min_chars + max_chars) / 2
    scale = (max_chars - min_chars) / 2

    # penalize larger deviations more harshly
    deviation = abs(char_count - target) / scale
    reward = math.exp(-deviation ** 2)  # sharper than linear decay
    return reward


# ========== Sampling Function ==========
def generate_samples(prompt: str, n: int, model: str):
    outputs = []
    for _ in tqdm(range(n)):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=100,
        )
        outputs.append(response.choices[0].message.content)
    return outputs

# ========== Main Pipeline ==========
if __name__ == "__main__":
    print("Generating samples...")
    samples = generate_samples(PROMPT, N_SAMPLES, MODEL)

    print("Calculating rewards...")
    rewards = [compute_reward(sample, CHAR_COUNT_MIN, CHAR_COUNT_MAX) for sample in samples]
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    reward_variance = np.var(rewards)
    reward_std = np.std(rewards)

    print(f"\nSummary:")
    print(f"  Prompt: {PROMPT}")
    print(f"  Model: {MODEL}")
    print(f"  Samples: {N_SAMPLES}")
    print(f"  Mean reward: {mean_reward:.4f}")
    print(f"  Reward variance: {reward_variance:.4f}")
    print(f"  Reward std dev: {reward_std:.4f}")
    print(f"  Reward vector: {[f'{r:.4f}' for r in rewards]}")

    for i, (text, score) in enumerate(zip(samples[:3], rewards[:3])):
        print(f"\n--- Sample {i+1} ---")
        print(f"Reward: {score:.4f}")
        print(f"Character count: {len(text)}")
        print(text.strip())