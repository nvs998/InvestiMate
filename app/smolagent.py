
import json
import re
import time
import requests

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

analyze_user: Get allocation based on user's surplus and risk
fetch_interest_rate: Get the current interest rate
format_allocation: Format asset allocation into a readable string

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
analyze_user: Get allocation based on user's surplus and risk, args: {"surplus": {"type": "int"}, "risk": {"type": "string"}}
fetch_interest_rate: Get the current interest rate, args: {}
format_allocation: Format asset allocation into a readable string, args: {"weights": {"type": "dict"}}
example use : 

{{
  "action": "analyze_user",
  "action_input": {"surplus": 60000, "risk": "high"}
}}


ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
(this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer."""


# --- Tools ---
def analyze_user(surplus, risk):
    score = 0
    if surplus > 50000:
        score += 2
    if risk == 'high':
        score += 2
    elif risk == 'moderate':
        score += 1

    if score >= 3:
        profile = "Aggressive Growth Investor"
        weights = {"Equity": 0.7, "Gold": 0.2, "Bonds": 0.1}
    elif score == 2:
        profile = "Balanced Investor"
        weights = {"Equity": 0.5, "Gold": 0.3, "Bonds": 0.2}
    else:
        profile = "Conservative Saver"
        weights = {"Equity": 0.3, "Gold": 0.2, "Bonds": 0.5}

    return {"profile": profile, "weights": weights}

def fetch_interest_rate():
    return {"interest_rate": 6.5}

def format_allocation(weights):
    return {"formatted": ", ".join([f"{k}: {v*100:.0f}%" for k, v in weights.items()])}

# TOOLS = {
#     "analyze_user": analyze_user,
#     "fetch_interest_rate": fetch_interest_rate,
#     "format_allocation": format_allocation,
# }

tools = {
    "analyze_user": analyze_user,
    "fetch_interest_rate": fetch_interest_rate,
    "format_allocation": format_allocation,
}

'''
#Smolagent implementation using TGI inferenece of huggingface llama3.1 model.

class SmolAgent:
    def __init__(self, model_url="http://localhost:8080"):
        self.model_url = model_url

    def call_llm(self, prompt: str) -> str:
        url = f"{self.model_url}/generate"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "stop": ["Observation:", "Final Answer:"]
            }
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        return result["generated_text"]

    def parse_action(self, llm_output: str) -> str | None:
        for line in llm_output.splitlines():
            if line.strip().lower().startswith("action:"):
                return line.split(":", 1)[1].strip()
        return None

    def run(self, question: str, tools: dict) -> str:
        history = f"Question: {question}\n"

        while True:
            llm_output = self.call_llm(history)
            print("\n🤖 LLM Output:\n", llm_output)

            action = self.parse_action(llm_output)
            if not action:
                # Final answer was reached
                return llm_output.strip()

            if action not in tools:
                return f"[ERROR] Unknown tool: {action}"

            # Run the tool
            observation = tools[action]()

            # Append new step to the full history
            history += f"{llm_output}\nObservation: {observation}\n"

agent = SmolAgent()
final_answer = agent.run("How should I invest ₹1,00,000?", tools)

print("\n✅ Final Answer:\n", final_answer)

'''

# --- LLM Wrapper (Ollama) ---
# def call_llm(messages,step):
#     response = requests.post("http://localhost:11434/api/chat", json={
#         "model": "llama3",
#         "messages": messages,
#         "stream": False,
#     })
#     print(f"🕒 LLM Response Time (Step {step+1}): {response.elapsed.total_seconds()} seconds")
#     content = response.json()["message"]["content"]
#     return content.strip()

# --- Parser ---
# def parse_action_block(output):
#     try:
#         action = re.search(r"action:\s*(.*)", output).group(1).strip()
#         action_input = re.search(r"action_input:\s*(.*)", output).group(1).strip()
#         args = json.loads(action_input)
#         return action, args
#     except Exception as e:
#         raise ValueError(f"Failed to parse Action or Action Input: {e}\nOutput was:\n{output}")

# --- Agent Loop ---
# def smol_agent(user_input):
#     messages = [
#         {
#             "role": "system",
#             "content": SYSTEM_PROMPT
#         },
#         {"role": "user", "content": user_input}
#     ]

#     for step in range(5):
#         llm_output = call_llm(messages,step)
#         print(f"\n🧠 LLM Output (Step {step+1}):\n{llm_output}")

#         if "Final Answer:" in llm_output:
#             print("\n✅ Final Answer from LLM:")
#             return llm_output.split("Final Answer:")[-1].strip()

#         try:
#             action, args = parse_action_block(llm_output)
#             print(f"\n⚙️ Running tool: {action} with args: {args}")
#             if action in TOOLS:
#                 result = TOOLS[action](**args)
#                 print("✅ Tool executed successfully. Result:")
#                 print(result)
#                 observation = f"Observation: {json.dumps(result)}"
#                 messages.append({"role": "assistant", "content": llm_output})
#                 messages.append({"role": "user", "content": observation})
#             else:
#                 return f"❌ Unknown tool: {action}"
#         except Exception as e:
#             return f"❌ Error during tool execution: {e}"

#     return "❌ Agent exceeded maximum steps without producing a final answer."

# --- CLI ---
# if __name__ == "__main__":
#     print("💬 Welcome to SmolAgent (LLM + Tools)")
#     while True:
#         query = input("\nAsk a question (or type 'exit'): ")
#         if query.lower() in ("exit", "quit"):
#             break
#         result = smol_agent(query)
#         print(f"\n🤖 Agent Response:\n{result}")

from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class SmolAgent:
    def __init__(self, model_path):
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device='cpu')

    def call_llm(self, prompt: str, stop_tokens=None) -> str:
        stop_tokens = stop_tokens or ["Observation:", "Final Answer:"]
        result = self.generator(prompt, max_new_tokens=256)[0]["generated_text"]

        # Manually stop at first matching stop token
        for stop in stop_tokens:
            if stop in result:
                result = result.split(stop)[0].strip()
                break

        return result

    def parse_action(self, llm_output: str) -> Optional[str]:
        for line in llm_output.splitlines():
            if line.strip().lower().startswith("action:"):
                return line.split(":", 1)[1].strip()
        return None

    def run(self, question: str, tools: dict) -> str:
        history = f"Question: {question}\n"

        while True:
            llm_output = self.call_llm(history)
            print("\n🤖 LLM Output:\n", llm_output)

            action = self.parse_action(llm_output)
            if not action:
                return llm_output.strip()  # Final answer

            if action not in tools:
                return f"[ERROR] Unknown tool: {action}"

            observation = tools[action]()  # Run the tool
            history += f"{llm_output}\nObservation: {observation}\n"


model_path = "llama3.1"  # update this to your actual model folder path

agent = SmolAgent(model_path)
final_answer = agent.run("How should I invest ₹1,00,000?", tools)

print("\n✅ Final Answer:\n", final_answer)