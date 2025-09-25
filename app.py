# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()

MODEL_DIR = "out/peft-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

class Req(BaseModel):
    prompt: str

def safety_check(prompt: str) -> bool:
    forbidden = []
    low = prompt.lower()
    return not any(x in low for x in forbidden)

@app.post("/generate")
def generate(req: Req):
    if not safety_check(req.prompt):
        return {"error":"I can't help with that. I can provide legal, ethical guidance or recommend testing in a sanctioned lab."}
    out = generator(req.prompt, max_new_tokens=256, do_sample=False)
    return {"text": out[0]["generated_text"]}