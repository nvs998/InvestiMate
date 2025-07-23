# from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from llm_models.llama3_wrapper import get_llama3_llm, get_mistral_llm
from prompt import prompt
# from rag.retriever import search_docs_tool

from tools.ml_tools import analyze_user_tool, apply_rules_tool, format_allocation_tool

from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_tool_messages
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain

from langchain.schema.output_parser import OutputParserException
import traceback
def main():
    # Step 1: Load LLaMA 3.1 model
    llm = get_mistral_llm()

    # Step 2: Register tools
    tools = [
        analyze_user_tool,
        # apply_rules_tool,
        # format_allocation_tool,
        # search_docs_tool,  # RAG
    ]

    # Step 3: Initialize agent with tools
    # agent = initialize_agent(
    #     tools=tools,
    #     llm=llm,
    #     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # Enables tool calling
    #     verbose=True,
    #     handle_parsing_errors=True
    # )

    # agent = StructuredChatAgent.from_llm_and_tools(
    #     llm=llm,
    #     tools=tools,
    #     prompt=prompt,
    # )

    agent = StructuredChatAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        output_parser=StructuredChatOutputParser(),
        allowed_tools=[tool.name for tool in tools]
    )

    executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # Step 4: Start interaction loop
    print("📈 AI Financial Advisor (LangChain + LLaMA 3.1)")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            query = input("🧠 You: ").strip()
            if query.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break

            try:
                # response = agent.invoke({"input": query})
                response = executor.invoke({"input": query})
                print(f"\n🤖 Advisor: {response}\n")

            except OutputParserException as e:
                print("⚠️ Could not parse the model's output.")
                print("👉 Tip: Try rephrasing the input or reducing complexity.")
                print(f"Error: {e}\n")

            except Exception as e:
                print("❌ An unexpected error occurred while processing your request.")
                print(traceback.format_exc())

    except KeyboardInterrupt:
        print("\n👋 Session interrupted. Exiting...")

if __name__ == "__main__":
    main()
    # llm = get_mistral_llm()
    # print(llm("What is asset allocation in finance?"))

