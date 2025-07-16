from agent.llm_agent import run_agent_conversation

if __name__ == "__main__":
    prompt = input("🧑‍💼 How can I help you with investing today?\n> ")
    response = run_agent_conversation(prompt)
    print("\n🤖 Response:\n", response)
