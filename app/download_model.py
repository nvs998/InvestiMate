# download_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# --- IMPORTANT: CHOOSE YOUR MODEL ID ---
# Pick the specific Llama 3.1 model you want to download.
# The 8B (8 Billion parameters) is the smallest and fastest to download/run.
# The 70B (70 Billion parameters) is much larger and requires significantly more VRAM and disk space.
# Ensure you have accepted the license for THIS SPECIFIC model ID on Hugging Face.

# Example for Llama 3.1 8B Instruct model:
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Example for Llama 3.1 70B Instruct model (requires much more resources!):
# model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# --- OPTIONAL: Specify a custom cache directory ---
# By default, models are downloaded to ~/.cache/huggingface/hub/
# If your home directory has limited space, you might want to use a /scratch or /data directory.
# Ensure this directory exists and you have write permissions.
# Example: os.environ['TRANSFORMERS_CACHE'] = '/scratch/trgl1183/huggingface_models'

print(f"Attempting to download model: {model_id}")
print(f"Model will be cached in: {os.environ.get('TRANSFORMERS_CACHE', '~/.cache/huggingface/hub/')}")

try:
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch using device: {device}")
    if device == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Load tokenizer (this will download if not cached)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Tokenizer downloaded successfully.")

    # Load model (this will download if not cached)
    # Use torch_dtype=torch.bfloat16 if your GPU supports it (modern NVIDIA GPUs like A2 do)
    # or torch_dtype=torch.float16 for other GPUs, to save VRAM and speed up inference.
    # If you face memory issues, consider loading in 8-bit or 4-bit precision (requires `bitsandbytes` library).
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, # Or torch.float16, or just leave it for default float32
        # low_cpu_mem_usage=True, # Recommended for large models to reduce CPU RAM during loading
        device_map="auto" # Automatically moves model parts to GPU/CPU as needed
    )
    print("Model downloaded successfully.")
    print("Model is now ready to be used!")

except Exception as e:
    print(f"An error occurred during download: {e}")
    print("Please double-check:")
    print("1. That you have accepted the model's license on its Hugging Face page.")
    print("2. That you successfully ran `huggingface-cli login` and entered the correct token.")
    print("3. That you have enough disk space.")
    print("4. That your `smolagent_env` has `transformers` and `torch` installed correctly.")

print("Download script finished.")