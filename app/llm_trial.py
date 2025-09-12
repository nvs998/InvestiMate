import torch
from transformers import BitsAndBytesConfig
from transformers import Mxfp4Config
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import os, socket, torch

def get_mistral_llm():
    """
    Loads mistralai/Mistral-7B-Instruct-v0.1 model and wraps it for LangChain usage.
    Uses 4-bit quantization for efficient memory usage (via bitsandbytes).
    """
    # model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
    # model_id = "unsloth/Kimi-K2-Base-BF16"
    model_id = 'openai/gpt-oss-20b'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}") 
    else:
        print("Using CPU device.")

    quantization_config = Mxfp4Config()
    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        device_map="auto",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer("Do you have time", return_tensors="pt").input_ids.to(0)
    print("Inputs:", inputs)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto"
    )

    
    # Hugging Face pipeline configuration
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        # device=0 if device == "cuda" else -1
    )

    # Wrap the pipeline for LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

llm2 = get_mistral_llm()
normal_answer = llm2.invoke('Should I invest in gold through Sovereign Gold Bonds or buy physical gold this year?')
print("normal answer===========",normal_answer)
