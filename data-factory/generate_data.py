import json
import asyncio
import os
import re
from typing import Optional, List, Literal
from openai import AsyncOpenAI
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "deepseek/deepseek-chat"

OUTPUT_FILE = "raw_reasoning_data.jsonl"

# Target Counts Needed
LIMIT_MATH = 0          # Target: 2000
LIMIT_CODE = 0          # Target: 1500
LIMIT_SCIENCE = 0       # Target: 1000
LIMIT_SUMMARIZATION = 0 # Target: 1000
LIMIT_CREATIVE = 0      # Target: 1000

class LLMStructuredOutput(BaseModel):
    """Validates the model followed the required XML format"""
    raw_text: str = Field(..., description="Raw LLM Output")
    reasoning: str = ""
    answer: str = ""
    
    @field_validator("raw_text")
    @classmethod
    def extract_sections(cls, text: str) -> str:
        reasoning = re.search(r"<reasoning>(.*?)</reasoning>", text, re.S)
        answer = re.search(r"<answer>(.*?)</answer>", text, re.S)
        
        if not reasoning or not answer:
            raise ValueError("Missing <reasoning> or <answer> tags")
        return text
    
    def parsed(self) -> "LLMStructuredOutput":
        self.reasoning = re.search(
            r"<reasoning>(.*?)</reasoning>", self.raw_text, re.S
        ).group(1).strip()
        
        return self
    
class TunixSample(BaseModel):
    instruction: str
    input: str
    output: str
    source: Literal["math", "code", "science", "summarization", "creative", "synthetic"]

PROMPT_MATH = """You are an expert reasoning engine.
FORMAT: <reasoning> [Step 1: ... Step 2: ...] </reasoning> <answer> [Final Answer] </answer>
STYLE: Identify variables -> State formula -> Calculate -> Verify.
Concise (3-7 steps). No text outside tags."""

PROMPT_CODE = """You are an expert coding assistant.
FORMAT: <reasoning> [Step 1: ... Step 2: ...] </reasoning> <answer> [Code Block] </answer>
STYLE: Analyze Request -> Choose Algo -> Logic Outline -> Edge Cases.
No text outside tags."""

PROMPT_SCIENCE = """You are a scientific reasoning expert.
FORMAT: <reasoning> [Step 1: ... Step 2: ...] </reasoning> <answer> [Answer] </answer>
STYLE: Identify Scientific Principle -> Apply to Scenario -> Deduce Outcome.
No text outside tags."""

PROMPT_SUMMARIZATION = """You are an expert summarizer.
FORMAT: <reasoning> [Step 1: ... Step 2: ...] </reasoning> <answer> [Summary] </answer>
STYLE: Scan Text -> Identify Key Entities -> Filter Noise -> Synthesize.
No text outside tags."""

PROMPT_CREATIVE = """You are an expert creative writer.
FORMAT: <reasoning> [Step 1: ... Step 2: ...] </reasoning> <answer> [Content] </answer>
STYLE: Analyze Intent -> Structure Content -> Key Themes -> Refine Draft.
No text outside tags."""

GLOBAL_SYNTHETIC_PROMPT = """You are a reasoning assistant.
Think step by step and explain your reasoning clearly.
FORMAT: <reasoning> [Step 1: ... Step 2: ...] </reasoning> <answer> [Content] </answer>
No text outside tags.
"""

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost",  
        "X-Title": "Tunix Synthetic Reasoning Generator"
    }
)



async def generate_sample(system_prompt: str, user_prompt: str, source: TunixSample.__annotations__["source"]) -> Optional[TunixSample]:
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4, 
            max_tokens=1024
        )
        
        raw_text = response.choices[0].message.content or ""
        structured = LLMStructuredOutput(raw_text=raw_text).parsed()
        
        return TunixSample(
            instruction=system_prompt, 
            input=user_prompt, 
            output=structured.raw_text, 
            source=source
        )
        
    except (ValidationError, ValueError) as e:
        return None
    except Exception as e:
        print(f"API error: {e}")
        return None

async def process_dataset() -> None:
    final_data: List[TunixSample] = []

    print("Generating MATH...")
    ds = load_dataset("gsm8k", "main", split="test")
    tasks = [
        generate_sample(PROMPT_MATH, item["question"], "math")
        for item in ds.select(range(LIMIT_MATH))
    ]
    final_data.extend(filter(None, await asyncio.gather(*tasks)))

    print("Generating CODE...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    tasks = [
        generate_sample(PROMPT_CODE, item["prompt"], "code")
        for item in ds.select(range(LIMIT_CODE))
    ]
    final_data.extend(filter(None, await asyncio.gather(*tasks)))

    print("Generating SCIENCE...")
    ds = load_dataset("sciq", split="train")
    tasks = [
        generate_sample(PROMPT_SCIENCE, item["question"], "science")
        for item in ds.select(range(LIMIT_SCIENCE))
    ]
    final_data.extend(filter(None, await asyncio.gather(*tasks)))

    print("Generating SUMMARIZATION...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
    tasks = []
    for item in ds.select(range(LIMIT_SUMMARIZATION)):
        snippet = item["article"][:1500]
        tasks.append(
            generate_sample(
                PROMPT_SUMMARIZATION,
                f"Summarize:\n{snippet}",
                "summarization"
            )
        )
    final_data.extend(filter(None, await asyncio.gather(*tasks)))

    print("Generating CREATIVE...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    tasks = [
        generate_sample(PROMPT_CREATIVE, item["prompt"], "creative")
        for item in ds.select(range(LIMIT_CREATIVE))
    ]
    final_data.extend(filter(None, await asyncio.gather(*tasks)))
    
    print("Generating SYNTHETIC MIXED TASKS...")

    with open("synthetic_prompts.json") as f:
        synthetic_prompts = json.load(f)

    tasks = [
        generate_sample(
            system_prompt=GLOBAL_SYNTHETIC_PROMPT,
            user_prompt=item["prompt"],
            source="synthetic"
        )
        for item in synthetic_prompts
    ]

    final_data.extend(filter(None, await asyncio.gather(*tasks)))
    
    print(f"Saving {len(final_data)} samples...")
    with open(OUTPUT_FILE, "w") as f:
        for sample in final_data:
            f.write(sample.model_dump_json() + "\n")

    print("Done.")

if __name__ == "__main__":
    asyncio.run(process_dataset())