from langgraph.prebuilt import create_react_agent
from llm_models.llama3_wrapper import get_llama3_llm, get_mistral_llm
def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

llm = get_mistral_llm()

agent = create_react_agent(
    model=llm,  
    tools=[get_weather],  
    prompt="You are a helpful assistant"  
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)