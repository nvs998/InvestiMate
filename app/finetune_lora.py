from datasets import load_dataset
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
dataset_path = "app/qa_data.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Load dataset
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Preprocess: Combine prompt + response
def format(example):
    return {
        "text": f"### Question:\n{example['prompt']}\n\n### Answer:\n{example['response']}"
    }

dataset = dataset.map(format)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True)

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Model + LoRA Config
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./finetuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()
model.save_pretrained("./finetuned-model")
