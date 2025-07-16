import json
import requests
from ml_tools.ml_engine import analyze_user
from logic_tools.custom_tools import fetch_interest_rate, format_allocation
from agent.tool_registry import tool_schema

def route_tool(name, args):
    if name == "analyze_user":
        return analyze_user(**args)
    elif name == "fetch_interest_rate":
        return fetch_interest_rate()
    elif name == "format_allocation":
        return format_allocation(**args)
    return "Unknown tool"

def run_agent_conversation(user_prompt):
    messages = [
        {"role": "system", "content": "You are an AI investment advisor."},
        {"role": "user", "content": user_prompt}
    ]

    res = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3",
        "messages": messages,
        "tools": tool_schema,
        "tool_choice": "auto"
    })

    try:
        data = res.json()
    except Exception:
        print("❌ Invalid JSON from LLM server:", res.text)
        return "LLM returned invalid response."

    # Log error if 'choices' is missing
    if "choices" not in data:
        print("❌ LLM API Error:", data)
        return f"LLM error: {data.get('error', 'No choices returned.')}"

    choice = data["choices"][0]
    tool_call = choice.get("tool_calls", [None])[0]

    if not tool_call:
        return choice["message"]["content"]

    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])
    result = route_tool(name, args)

    messages.append({
        "role": "assistant",
        "tool_call_id": tool_call["id"],
        "content": str(result)
    })

    follow_up = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3",
        "messages": messages
    }).json()

    return follow_up['choices'][0]['message']['content']
