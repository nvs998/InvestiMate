import torch
from transformers import BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def get_llama3_llm():
    """
    Loads meta-llama/Meta-Llama-3.1-8B-Instruct model and wraps it for LangChain usage.
    """

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # --- Configure Quantization (NEW) ---
    # This configuration tells transformers to load the model in 4-bit precision
    # using the NF4 quantization type with double quantization.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Compute dtype for operations (A2 supports bfloat16)
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config, # <--- NEW ARGUMENT for 4-bit loading
        # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Create HF pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    # Wrap in LangChain LLM interface
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm
