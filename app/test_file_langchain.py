# test_llama3_wrapper.py

from llm_models.llama3_wrapper import get_llama3_llm

def test_llm():
    llm = get_llama3_llm()

    print("🤖 LLaMA 3.1 is ready. Type a question.")
    while True:
        query = input("🧠 You: ")
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting.")
            break

        try:
            response = llm(query)
            print(f"🤖 LLaMA: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_llm()
