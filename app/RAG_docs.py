from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA

from llm_models.llama3_wrapper import get_llama3_llm, get_mistral_llm

documents = [
    {"source": "Investopedia", "content": "ELSS stands for Equity Linked Savings Scheme. It allows tax deduction under section 80C."},
    {"source": "SEBI", "content": "Mutual funds pool money from investors to invest in equity or debt instruments."},
    {"source": "Finshots", "content": "Index funds mirror a stock market index like NIFTY 50 or Sensex."}
]

# Convert to LangChain documents
docs = [Document(page_content=doc["content"], metadata={"source": doc["source"]}) for doc in documents]

# Split into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# Use SentenceTransformers for embedding
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedder)

retriever = vectorstore.as_retriever(search_type="similarity", k=3)

# Load small model for demo (replace with LLaMA, Mistral, etc.)
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.7)
# llm = HuggingFacePipeline(pipeline=pipe)

# Step 1: Load LLaMA 3.1 model
llm = get_mistral_llm()

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

query = "What is ELSS and what tax benefit does it offer?"
response = rag_chain(query)

print("AnswerRAG:\n", response["result"])
# print("\nSources:")
# for doc in response["source_documents"]:
#     print("-", doc.metadata["source"])

# vanilla_prompt = f"Question: {query}\nHelpful Answer:"

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch

# Base + LoRA paths
base_model = "mistralai/Mistral-7B-Instruct-v0.1"  # Same as used during training
lora_weights = "./finetuned-model"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# Load base model with 4-bit quantization (same config as training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map="auto")

# Attach LoRA weights
model = PeftModel.from_pretrained(model, lora_weights)

# Setup generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# Query using same format you trained on
query = "What is ELSS and what tax benefit does it offer?"
prompt = f"### Question:\n{query}\n\n### Answer:\n"
output = generator(prompt)[0]["generated_text"]

print("📊 Fine-Tuned Answer:\n", output)

response = llm.invoke(query)
print("Vanilla LLM Answer:\n", response)
