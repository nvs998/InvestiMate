
import sys
# print("Python Executable:", sys.executable)
# print("Python Version:", sys.version)

from tools.ml_tools import analyze_user_tool, format_allocation_tool
from llm_models.llama3_wrapper import get_llama3_llm, get_mistral_llm
from prompt import prompt
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor
from langchain.agents import create_structured_chat_agent

from langchain.schema.output_parser import OutputParserException

import traceback

# Import the necessary callback class
import json
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish # For type hinting
from uuid import UUID
from typing import Any, Dict, List
from langchain_core.outputs import LLMResult

class CleanAgentStepCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts running."""
        print("\n--- 💡 LLM START ---")
        print(f"--- Prompts sent to LLM ({len(prompts)} total):")
        for i, prompt_text in enumerate(prompts):
            # For ChatPromptTemplate, prompts[0] might be a string representation of all messages
            # For direct message lists, it's simpler.
            # print(f"--- Prompt {i+1}:\n{prompt_text}")
            # A more robust print for ChatPromptTemplate structure (if it passes a list of BaseMessages as prompts)
            if isinstance(prompt_text, str):
                print(f"--- Prompt {i+1} (Raw String):\n{prompt_text}")
            else: # Assuming it might be a list of BaseMessage objects for chat models
                print(f"--- Prompt {i+1} (Messages):")
                for msg in prompt_text:
                    print(f"    - {msg.type.upper()}: {msg.content}")
        print("--- END LLM START ---")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        print("\n--- ⚡ LLM END ---")
        print(f"--- Raw LLM Response:")
        if response.generations and response.generations[0] and response.generations[0][0]:
            print(response.generations[0][0].text)
        else:
            print("No text generation found in response.")
        print("--- END LLM END ---")

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Run on agent action."""
        print("==========================================================================")
        print(type(action), action)
        print(f"\n---Thought: {action.log.strip().split('Action:')[0].replace('Thought:', '').strip()}")
        print(f"---Action: {action.tool}")
        try:
            # Attempt to parse action_input as JSON for cleaner display
            action_input_json = json.dumps(action.tool_input, indent=2)
            print(f"---Action Input:\n{action_input_json}\n ====action_input_type{type(action.tool_input)}")
        except TypeError:
            print(f"---Action Input: {action.tool_input}")

    def on_tool_start(self, serialized: dict, input_str: str, run_id: UUID, **kwargs) -> None:
        """Run on tool start."""
        print("which toollll=====",{serialized},"runID==========",{run_id})
        print(f"--- 📊 Start input for tool:\n{input_str}") # Print raw output for clarity

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Run on tool end."""
        print(f"--- 📊 Observation:\n{output}") # Print raw output for clarity

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Run on agent finish."""
        print("\n--- ✅ Agent Finished ---")
        print(f"--- 🏁 Final Answer:\n{finish.return_values['output']}")
        print("-------------------------\n")

def main():
    # Step 1: Load LLaMA 3.1 model
    llm = get_mistral_llm()

    # Step 2: Register tools
    tools = [
        analyze_user_tool,
        format_allocation_tool
    ]
    
    agent = create_structured_chat_agent(
        llm = llm,
        tools = tools,
        prompt = prompt,
        stop_sequence=["\nObservation:"]
    )

    executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        callbacks=[CleanAgentStepCallbackHandler()]
    )

    # Step 4: Start interaction loop
    try:
        response = executor.invoke({
            "input": "I have ₹60,000 and moderate risk. What should I do?",
            "tool_names": ["analyze_user_tool", "format_allocation_tool"]
        })
        print(f"\n🤖 Advisor: {response['output']}\n")
    except Exception as e:
        print("❌ An unexpected error occurred while processing your request.")
        print(traceback.format_exc())

    # try:
    #     while True:
    #         query = input("🧠 You: ").strip()
    #         if query.lower() in {"exit", "quit"}:
    #             print("👋 Goodbye!")
    #             break

    #         try:
    #             # response = agent.invoke({"input": query})
    #             print(f"\n🤖 Advisor: {response}\n")

    #         except OutputParserException as e:
    #             print("⚠️ Could not parse the model's output.")
    #             print("👉 Tip: Try rephrasing the input or reducing complexity.")
    #             print(f"Error: {e}\n")

    #         except Exception as e:
    #             print("❌ An unexpected error occurred while processing your request.")
    #             print(traceback.format_exc())

    # except KeyboardInterrupt:
    #     print("\n👋 Session interrupted. Exiting...")

if __name__ == "__main__":
    main()
    # llm = get_mistral_llm()
    # print(llm("What is asset allocation in finance?"))

