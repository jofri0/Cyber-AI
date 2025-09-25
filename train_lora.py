# train_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype="auto", load_in_4bit=True, quantization_config=None)

# Prepare for 4-bit k-bit training (if using bitsandbytes)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load JSONL dataset
dataset = load_dataset("json", data_files={"train":"train.jsonl", "validation":"val.jsonl"})
def tokenize_fn(example):
    prompt = f"### Instruction:\n{example['input']}\n\n### Response:\n"
    tokenized = tokenizer(prompt + example['output'], truncation=True, max_length=1024)
    return tokenized

dataset = dataset.map(tokenize_fn, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_total_limit=2,
    learning_rate=2e-4,
    optim="adamw_torch"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"], eval_dataset=dataset["validation"], data_collator=data_collator)
trainer.train()
model.save_pretrained("out/peft-model")
tokenizer.save_pretrained("out/peft-model")