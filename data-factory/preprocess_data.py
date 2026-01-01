import json
import random
from typing import Dict, List

INPUT_FILE = "raw_reasoning_data.jsonl"
TRAIN_FILE = "train_data.jsonl"
VAL_FILE = "val_data.jsonl"

SEED = 42
TRAIN_RATIO = 0.90

GLOBAL_PROMPT_HEADER = """You are a reasoning asssistant. 

Follow these rules:
- Think step by step
- Respond ONLY in the following format:
<reasoning>...</reasoning>
<answer>...</answer>
"""

def build_prompt(sample: Dict) -> str:
    task = sample["source"]
    user_input = sample["input"]
    
    prompt = (
        GLOBAL_PROMPT_HEADER + f"\n[Task: {task}]\n" + user_input.strip()
    )
    
    return prompt

def load_and_convert() -> List[Dict]:
    processed = []
    
    with open(INPUT_FILE, "r") as f:
        for line in f:
            sample = json.loads(line)
            
            processed.append({
                "prompt": build_prompt(sample), 
                "response": sample["output"].strip(),
                "task": sample["source"]
            })
            
    return processed

def split_and_save(data: List[Dict]) -> None:
    random.seed(SEED)
    
    task_groups: Dict[str, List[Dict]] = {}
    for sample in data:
        task = sample["task"]
        task_groups.setdefault(task, []).append(sample)
        
    train_data: List[Dict] = []
    val_data: List[Dict] = []
    
    for task, samples in task_groups.items():
        random.shuffle(samples)
        
        split_idx = int(len(samples) * TRAIN_RATIO)
        task_train = samples[:split_idx]
        task_val = samples[split_idx:]
        
        train_data.extend(task_train)
        val_data.extend(task_val)
        
        print(
            f"Task '{task}': "
            f"total={len(samples)}, "
            f"train={len(task_train)}, "
            f"val={len(task_val)}"
        )
    
    with open(TRAIN_FILE, "w") as f:
        for row in train_data:
            f.write(json.dumps(row) + "\n")

    with open(VAL_FILE, "w") as f:
        for row in val_data:
            f.write(json.dumps(row) + "\n")

    print(f"\nTotal train samples: {len(train_data)} -> {TRAIN_FILE}")
    print(f"Total val samples: {len(val_data)} -> {VAL_FILE}")

    
def main():
    data = load_and_convert()
    split_and_save(data)
    
if __name__ == "__main__":
    main()