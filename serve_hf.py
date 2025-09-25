from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 150

app = FastAPI()

# Load base + adapter
BASE_MODEL = "facebook/opt-350m"
ADAPTER_PATH = "./outputs"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

@app.post("/generate")
def generate_text(query: Query):
    inputs = tokenizer(query.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=query.max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}