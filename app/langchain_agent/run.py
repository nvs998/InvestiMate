# run.py

from agent_runner import build_agent

if __name__ == "__main__":
    agent = build_agent()

    while True:
        user_input = input("\n💬 Ask something (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        response = agent.run(user_input)
        print("\n🤖 Response:\n", response)

# This has potential and if nothing works out I will use this as a fallback.