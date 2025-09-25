# data_prep.py
from datasets import load_dataset
from transformers import AutoTokenizer
import json

MODEL = "meta-llama/Llama-2-7b-hf"  # replace with your chosen base and check license
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)

def convert_to_instruction_format(raw_examples):
    out = []
    for ex in raw_examples:
        inp = ex.get("question") or ex.get("title") or ex.get("input") or ""
        out_text = ex.get("answer") or ex.get("text") or ex.get("body") or ""
        # Basic safety filter
        if any(bad in inp.lower() for bad in ["brute force", "exploit", "shellcode"]):
            continue
        out.append({"input": inp, "output": out_text})
    return out

def main():
    ds = load_dataset("stackexchange", "stackoverflow")  # example; adapt to local/curated sources
    raw = ds["train"].select(range(10000))  # small sample for dev
    processed = convert_to_instruction_format(raw)
    with open("train.jsonl", "w", encoding="utf-8") as f:
        for r in processed:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()