import re
import json
import time
import torch
import requests
from typing import Optional, Dict

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

Tool 1 = { 
            Name: analyze_user,
            Purpose: Get allocation based on user's surplus and risk tolerance,
            args: {"surplus": {"type": "int"}, "risk": {"type": "string"}}
        }
Tool 2 = { 
            Name: fetch_interest_rate,
            Purpose: Get the current interest rate,
            args: {}
        }
Tool 3 = { 
            Name: format_allocation,
            Purpose: Format asset allocation into a readable string,
            args: {"weights": {"type": "dict"}}
        }

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with its value as the exact name of the tool you want to use) and 
an `action_input` key (which will contain the arguments for the tool in exact format as shown in above schema).

example use : 

{{
  "action": "analyze_user",
  "action_input": {"surplus": 60000, "risk": "high"}
}}


ALWAYS use the following format while thinking about the answer:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:{"action": "tool_name", "action_input": {"param": "value"}}
(Replace with your actual action JSON. Ensure the JSON is a single valid object directly following "Action: ")

JSON here in the format shown above as example use.

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
(this Thought/Action/Observation can repeat N times, you should take several steps when needed. Each "Action:" must be followed by a SINGLE valid JSON object.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer."""


# --- Tools ---
def analyze_user(action_input_dict: Dict) -> Dict:
    surplus = action_input_dict.get('surplus')
    risk = action_input_dict.get('risk')
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

def fetch_interest_rate(action_input_dict: Dict) -> Dict:
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
from transformers import BitsAndBytesConfig
from parser import get_json_before_first_observation, remove_after_first_observation, extract_json_from_llm_code_block, extract_action_json

class SmolAgent:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # --- Configure Quantization (NEW) ---
        # This configuration tells transformers to load the model in 4-bit precision
        # using the NF4 quantization type with double quantization.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 # Compute dtype for operations (A2 supports bfloat16)
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=bnb_config, # <--- NEW ARGUMENT for 4-bit loading
            device_map="auto" # Keep device_map="auto" to handle model placement
        )
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)


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
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {"role": "user", "content": question}
        ]

        while True:
            llm_output = self.call_llm(messages)
            print("=============LLM output==================")
            for item in llm_output:
                print(json.dumps(item, indent=2)) # indent=2 makes the JSON output nicely formatted
                print("-" * 50)
            tool_json = extract_action_json(llm_output[-1])
            print("===++++",tool_json,"+++++====")
            # action = self.parse_action(llm_output[-1])
            if not tool_json:
                return llm_output.strip()  # Final answer
            action = tool_json['action']
            action_input = tool_json['action_input']
            if action not in tools:
                return f"[ERROR] Unknown tool: {action}"

            observation = tools[action](action_input)  # Run the tool
            print("=========",observation,'\n')
            history = remove_after_first_observation(llm_output[-1])
            print("=========",history,'\n')
            content1 = history['content']
            print("=========",content1,'\n',type(content1),'\n')
            content2 = history['content'] + json.dumps(observation) + '.Thought:'
            history['content'] = content2
            print("=========",content2,'\n',type(content2),'\n')
            print("=========",history,'\n',type(history),'\n')
            llm_output[-1] = history
            messages = str(llm_output)
            print("====messages",messages,'\n')


model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # update this to your actual model folder path

agent = SmolAgent(model_path)
final_answer = agent.run("How should I invest ₹1,00,000?", tools)

print("\n✅ Final Answer:\n", final_answer)