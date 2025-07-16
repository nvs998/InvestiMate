from agent import run_agent

if __name__ == "__main__":
    user_input = input("💬 Ask something: ")
    result = run_agent(user_input)
    print("\n🤖 Agent:\n", result)
