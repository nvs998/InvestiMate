import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3"  # or any other model you've pulled in Ollama

def call_llm(conversation_history):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": conversation_history,
            "stream": False
        }
    )
    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        raise RuntimeError(f"LLM call failed: {response.text}")
