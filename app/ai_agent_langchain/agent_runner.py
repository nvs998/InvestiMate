# agent_runner.py

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.tools import StructuredTool
from langchain_community.llms import Ollama
from tools import analyze_user, fetch_interest_rate, format_allocation

def build_agent():
    llm = Ollama(model="llama3")  # or change model name

    tools = [
        StructuredTool.from_function(
            name="analyze_user",
            description="Analyze user by providing surplus (number) and risk (low/moderate/high)",
            func=lambda x: analyze_user(**eval(x)),
            return_direct=False,
        ),
        StructuredTool.from_function(
            name="fetch_interest_rate",
            description="Fetch current interest rate",
            func=lambda x: fetch_interest_rate(),
            return_direct=False,
        ),
        StructuredTool.from_function(
            name="format_allocation",
            description="Format asset allocation nicely",
            func=lambda x: format_allocation(eval(x)["weights"]),
            return_direct=False,
        ),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent
