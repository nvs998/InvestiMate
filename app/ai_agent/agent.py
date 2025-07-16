import json
import requests
import re
from tools import analyze_user, fetch_interest_rate, format_allocation

def extract_json_block(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(match.group()) if match else None
    except Exception as e:
        raise ValueError(f"Could not extract JSON: {e}")

def prompt_llm(messages):
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3",
        "messages": messages
    })
    full_output = ""
    for line in response.text.strip().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            full_output += obj.get("message", {}).get("content", "")
        except json.JSONDecodeError:
            continue

    return full_output.strip()

def route_tool_call(json_obj):
    name = json_obj.get("tool")
    args = json_obj.get("args", {})

    if name == "analyze_user":
        return analyze_user(**args)
    elif name == "fetch_interest_rate":
        return fetch_interest_rate()
    elif name == "format_allocation":
        return format_allocation(**args)
    else:
        return {"error": "Unknown tool"}

def run_agent(user_input):
    # Step 1: Ask LLM what tool it wants to call
    system_prompt = {
        "role": "system",
        "content": (
            "You are a financial advisor AI that uses tools to reason.\n"
            "When given a user question, respond ONLY with a JSON tool call like:\n"
            '{"tool": "analyze_user", "args": {"surplus": 60000, "risk": "moderate"}}\n\n'
            "After receiving a tool result from the assistant, explain it in natural language.\n"
            "You can use these tools:\n"
            "- analyze_user(surplus, risk)\n"
            "- fetch_interest_rate()\n"
            "- format_allocation(weights)\n"
            "If you are asking for a tool respond ONLY with JSON like:\n"
            '{"tool": "TOOL_NAME", "args": {"arg1": ..., "arg2": ...}}'
        )
    }

    messages = [system_prompt, {"role": "user", "content": user_input}]
    response_text = prompt_llm(messages)
    print("\n🔎 LLM raw response:\n", response_text)

    try:
        tool_json = extract_json_block(response_text)
        if not tool_json:
            raise ValueError("❌ LLM did not return valid JSON tool call.")
    except Exception as e:
        raise ValueError(f"❌ Failed to parse tool call: {e}")
    
    tool_result = route_tool_call(tool_json)
    print("🛠️ Tool executed, result:\n", tool_result)
    
    # Step 2: Send result back to LLM for final answer
    messages.append({"role": "assistant", "content": json.dumps(tool_result)})
    print("\n📜 Messages so far:\n", json.dumps(messages, indent=2))
    final_response = prompt_llm(messages)
    print("\n🤖 Final response from LLM:\n", final_response)
    return final_response