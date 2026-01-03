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

MAX_CONCURRENT = 10

# Target Counts Needed
LIMIT_MATH = 5          # Target: 2000
LIMIT_CODE = 0          # Target: 1500
LIMIT_SCIENCE = 0       # Target: 2000
LIMIT_SUMMARIZATION = 0 # Target: 1000
LIMIT_CREATIVE = 0      # Target: 2500

class DPOStructuredOutput(BaseModel):
    raw_text: str = Field(..., description="Raw LLM Output")
    chosen: str = ""
    rejected: str = ""
    metadata: dict = {}
    
    def parsed(self) -> "DPOStructuredOutput":
        # Extract the perfect answer
        chosen_match = re.search(r"<chosen>(.*?)</chosen>", self.raw_text, re.S)
        if chosen_match:
            self.chosen = chosen_match.group(1).strip()
            
        # Extract the Flawed Answer (Loser)
        rejected_match = re.search(r"<rejected>(.*?)</rejected>", self.raw_text, re.S)
        if rejected_match:
            self.rejected = rejected_match.group(1).strip()
            
        # Extract the Loophole Metadata (Grading Key)
        meta_match = re.search(r"<metadata>(.*?)</metadata>", self.raw_text, re.S)
        if meta_match:
            try:
                self.metadata = json.loads(meta_match.group(1).strip())
            except:
                self.metadata = {}
        
        return self
    
class TunixSample(BaseModel):
    instruction: str
    input: str
    chosen_response: str
    rejected_response: str
    metadata: dict  
    source: Literal["math", "code", "science", "summarization", "creative", "synthetic"]

XML_TEMPLATE = """
RESPONSE TEMPLATE:
<chosen>
<reasoning>
[Your step-by-step logic here]
</reasoning>
<answer>
[Final Answer here]
</answer>
</chosen>

<rejected>
<reasoning>
[Your flawed logic here]
</reasoning>
<answer>
[Final Answer here]
</answer>
</rejected>

<metadata>
{ "key": "value" }
</metadata>
"""

# A menu of broad failure categories. 
ERROR_TYPES = f"""
You are generating a DPO Pair.
{XML_TEMPLATE}
For the <rejected> response, you MUST simulate a "Plausible Failure." 
Randomly select ONE failure mode from the list below:

1. **The Subtle Hallucinator:** Invent a plausible-sounding fact or library function to bridge a gap in knowledge.
2. **The Logic Jumper:** Skip a crucial intermediate step. State "Therefore, X is true" when X hasn't actually been proven yet.
3. **The Edge-Case Ignorer:** Write a solution that works for the general case but fails for specific inputs (e.g., negative numbers, empty lists).
4. **The Constraint Drifter:** Start following the constraints but lose track of them by the end of the response.
5. **The Pseudo-Parallel:** Write a reasoning trace that looks distinct from the answer, but the answer doesn't actually follow from the reasoning.
"""

PROMPT_MATH = f"""You are an Expert Math Reasoning Engine.
{ERROR_TYPES}

YOU MUST FOLLOW THE RESPONSE TEMPLATE EXACTLY.
BEGIN OUTPUT IMMEDIATELY WITH <chosen>

TASK: Solve the user's math problem.

<chosen>
- STYLE: Auditable & Explicit.
- Do not just state the formula; explain *why* it applies here.
- Show the intermediate verification of each step (e.g., "Checking units...").
- FORMAT: Follow the RESPONSE TEMPLATE exactly.
</chosen>

<rejected>
- Simulate a "Confident Error."
- **Guidance:** Perform the right setup but make a logical slip in the execution (e.g., applying a property of addition to multiplication). 
- The tone should be highly confident so the model learns to distrust confidence without verification.
</rejected>

<metadata>
Output a JSON with: "checkpoints" (list of 2 correct intermediate numbers), "final_answer".
</metadata>
"""

PROMPT_CODE = f"""You are a Senior Python Developer.
{ERROR_TYPES}

YOU MUST FOLLOW THE RESPONSE TEMPLATE EXACTLY.
BEGIN OUTPUT IMMEDIATELY WITH <chosen>

TASK: Write code to solve the user's request.

<chosen>
- STYLE: Robust & Explained.
- You can use advanced logic, but you must EXPLAIN the algorithmic choice in the reasoning trace.
- Explicitly mention how you are handling edge cases (empty inputs, extensive values).
- FORMAT: Follow the RESPONSE TEMPLATE exactly.
</chosen>

<rejected>
- Simulate a "Junior Developer" mistake.
- **Guidance:** Write code that looks clean but has a hidden bug (e.g., off-by-one error, shallow copy vs deep copy issue, or unhandled nulls).
- Or, import a heavy library (like pandas) for a trivial task.
</rejected>

<metadata>
Output a JSON with: "required_functions" (list), "forbidden_concepts" (list).
</metadata>
"""

PROMPT_SCIENCE = f"""You are a Scientific Reasoning Expert.
{ERROR_TYPES}

YOU MUST FOLLOW THE RESPONSE TEMPLATE EXACTLY.
BEGIN OUTPUT IMMEDIATELY WITH <chosen>

TASK: Explain a scientific phenomenon or solve a science problem.

<chosen>
- STYLE: First Principles.
- Derive the answer from fundamental laws. Connect the dots: "Because A is true, B happens, which leads to C."
- Avoid "Just-so" stories.
- FORMAT: Follow the RESPONSE TEMPLATE exactly.
</chosen>

<rejected>
- Simulate a "Plausible Misconception."
- **Guidance:** Use correct terminology but incorrect relationships (e.g., "The heavier object falls faster because it has more gravity," which sounds right to a novice but is wrong in a vacuum).
</rejected>

<metadata>
Output a JSON with: "key_principles" (list), "common_misconceptions" (list).
</metadata>
"""

PROMPT_SUMMARIZATION = f"""You are an Expert Summarizer.
{ERROR_TYPES}

YOU MUST FOLLOW THE RESPONSE TEMPLATE EXACTLY.
BEGIN OUTPUT IMMEDIATELY WITH <chosen>

TASK: Summarize the provided text.

<chosen>
- STYLE: High Information Density.
- Identify the core narrative thread. Discard fluff.
- Synthesize facts rather than just copying sentences.
- FORMAT: Follow the RESPONSE TEMPLATE exactly.
</chosen>

<rejected>
- Simulate a "Drift" failure.
- **Guidance:** Start strong, but then hallucinate a detail that *could* have been in the text but wasn't. Or, fixate on a minor anecdote and miss the headline.
</rejected>

<metadata>
Output a JSON with: "keywords" (list of critical entities), "extraneous" (list of 1 detail to avoid).
</metadata>
"""

PROMPT_CREATIVE = f"""You are a Creative Writing Expert.
{ERROR_TYPES}

YOU MUST FOLLOW THE RESPONSE TEMPLATE EXACTLY.
BEGIN OUTPUT IMMEDIATELY WITH <chosen>

TASK: Write creative content based on the prompt.

<chosen>
- STYLE: Intentional Structure.
- The reasoning trace should be a "Blueprint": Outline the themes, tone, and constraints before writing a single word of the content.
- Execute the blueprint perfectly.
- FORMAT: Follow the RESPONSE TEMPLATE exactly.
</chosen>

<rejected>
- Simulate a "Constraint Amnesia."
- **Guidance:** Acknowledge the constraints in the reasoning, but fail to implement them in the final answer. (e.g., "I will avoid the word 'dark'", but then use "darkness" in the story).
</rejected>

<metadata>
Output a JSON with: "must_include" (list), "must_avoid" (list).
</metadata>
"""

GLOBAL_SYNTHETIC_PROMPT = f"""You are a General Reasoning Assistant.
{ERROR_TYPES}

YOU MUST FOLLOW THE RESPONSE TEMPLATE EXACTLY.
BEGIN OUTPUT IMMEDIATELY WITH <chosen>

TASK: Solve the logic puzzle or general query.

<chosen>
- STYLE: Step-by-step Logic.
- The most important requirement is the XML Format: <reasoning> followed by <answer>.
- Reasoning must self-correct if it spots a potential error.
</chosen>

<rejected>
- Simulate a "Format" or "Lazy" failure.
- **Guidance:** Provide the right answer for the wrong reasons, or put the answer inside the reasoning block.
</rejected>

<metadata>
Output a JSON with: "reasoning_steps_estimate" (int), "tags_checked" (boolean).
</metadata>
"""

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost",  
        "X-Title": "Tunix Synthetic Reasoning Generator"
    }
)

sem = asyncio.Semaphore(MAX_CONCURRENT)

async def generate_sample(system_prompt: str, user_prompt: str, source: str) -> Optional[TunixSample]:
    async with sem:  
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4, 
                max_tokens=2500  
            )
            
            raw_text = response.choices[0].message.content or ""
            print("RAW OUTPUT:\n", raw_text[:500])
            structured = DPOStructuredOutput(raw_text=raw_text).parsed()
            
            # Ensure we have a complete pair
            if structured.chosen and structured.rejected:
                return TunixSample(
                    instruction=system_prompt,
                    input=user_prompt,
                    chosen_response=structured.chosen,
                    rejected_response=structured.rejected,
                    metadata=structured.metadata,
                    source=source
                )
            else:
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

    # print("Generating CODE...")
    # ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    # tasks = [
    #     generate_sample(PROMPT_CODE, item["prompt"], "code")
    #     for item in ds.select(range(LIMIT_CODE))
    # ]
    # final_data.extend(filter(None, await asyncio.gather(*tasks)))

    # print("Generating SCIENCE...")
    # ds = load_dataset("sciq", split="train")
    # tasks = [
    #     generate_sample(PROMPT_SCIENCE, item["question"], "science")
    #     for item in ds.select(range(LIMIT_SCIENCE))
    # ]
    # final_data.extend(filter(None, await asyncio.gather(*tasks)))

    # print("Generating SUMMARIZATION...")
    # ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
    # tasks = []
    # for item in ds.select(range(LIMIT_SUMMARIZATION)):
    #     snippet = item["article"][:1500]
    #     tasks.append(
    #         generate_sample(
    #             PROMPT_SUMMARIZATION,
    #             f"Summarize:\n{snippet}",
    #             "summarization"
    #         )
    #     )
    # final_data.extend(filter(None, await asyncio.gather(*tasks)))

    # print("Generating CREATIVE...")
    # ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    # tasks = [
    #     generate_sample(PROMPT_CREATIVE, item["prompt"], "creative")
    #     for item in ds.select(range(LIMIT_CREATIVE))
    # ]
    # final_data.extend(filter(None, await asyncio.gather(*tasks)))
    
    # print("Generating SYNTHETIC MIXED TASKS...")

    # with open("synthetic_prompts.json") as f:
    #     synthetic_prompts = json.load(f)

    # tasks = [
    #     generate_sample(~
    #         system_prompt=GLOBAL_SYNTHETIC_PROMPT,
    #         user_prompt=item["prompt"],
    #         source="synthetic"
    #     )
    #     for item in synthetic_prompts
    # ]
 
    # final_data.extend(filter(None, await asyncio.gather(*tasks)))
    
    print(f"Saving {len(final_data)} samples...")
    with open(OUTPUT_FILE, "w") as f:
        for sample in final_data:
            f.write(sample.model_dump_json() + "\n")

    print("Done.")

if __name__ == "__main__":
    asyncio.run(process_dataset())