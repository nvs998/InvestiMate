from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from llm_models.llama3_wrapper import get_llama3_llm
# from rag.retriever import search_docs_tool

from tools.ml_tools import analyze_user_tool, apply_rules_tool, format_allocation_tool


def main():
    # Step 1: Load LLaMA 3.1 model
    llm = get_llama3_llm()

    # Step 2: Register tools
    tools = [
        analyze_user_tool,
        # apply_rules_tool,
        # format_allocation_tool,
        # search_docs_tool,  # RAG
    ]

    # Step 3: Initialize agent with tools
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Enables tool calling
        verbose=True,
        handle_parsing_errors=True
    )

    # Step 4: Start interaction loop
    print("📈 AI Financial Advisor (LangChain + LLaMA 3.1)")
    while True:
        query = input("\n🧠 You: ")
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        response = agent.invoke(query)
        print(f"\n🤖 Advisor: {response}")

if __name__ == "__main__":
    main()
