import sys
from tools.ml_tools import analyze_user_tool, format_allocation_tool
from llm_models.llama3_wrapper import get_llama3_llm, get_mistral_llm, get_kimi
# from prompt import prompt
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor
from langchain.agents import create_structured_chat_agent

from langchain_core.runnables import RunnableLambda
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

from langchain.schema.output_parser import OutputParserException

import traceback

# Import the necessary callback class
import json
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish # For type hinting
from uuid import UUID
from typing import Any, Dict, List
from langchain_core.outputs import LLMResult

from langchain.tools import tool
from pydantic import BaseModel, Field


import torch
from transformers import BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# llm = get_mistral_llm()
llm = get_kimi()

# RAG Setup: loaders, splitter, embeddings, vector store
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1) Load your PDFs
# pdf_paths = [
#     "../query1_data/Gold ETF vs Physical Gold.pdf",
#     "../query1_data/Sovereign Gold Bond Scheme 2025-26.pdf"
# ]

# docs = []
# for p in pdf_paths:
#     loader = PyPDFLoader(p)
#     # Each page comes with metadata; loader returns Document objects
#     docs.extend(loader.load())

# # 2) Split into chunks (keep overlaps for context continuity)
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=900, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
# )
# splits = splitter.split_documents(docs)

# # 3) Embeddings (no API key needed)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # 4) Build FAISS index
# vecstore = FAISS.from_documents(splits, embeddings)

# # 5) Create a retriever (top_k configurable inside the tool)
# retriever = vecstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# # Define your query
# query = "What are the benefits of Sovereign Gold Bonds compared to physical gold?"

# # Search for relevant chunks
# retrieved_docs = vecstore.similarity_search(query, k=3)

# # Combine retrieved content
# context = "\n\n".join([doc.page_content for doc in retrieved_docs])

# print(context)

class DecomposeQueryInput(BaseModel):
    # This Pydantic model defines the *expected input* for the query_decomposition_tool
    query: str = Field(..., description="User's query which was received as input")

@tool(args_schema=DecomposeQueryInput)
def query_decomposition_tool(query: str) -> str:
    """
    Decompose user query into several smaller steps.
    Takes the full input string from user query.
    Example input (as a string):
    {
        "query": Should I invest in gold through Sovereign Gold Bonds or buy physical gold this year?
    }
    """

    # Access arguments directly as passed by Pydantic
    query = query

    # Use f-strings for formatting and comma for thousands separator
    return (
        f"Based on the question that user asked, "
        f"We should analyse factors like:\n"
        f"1. Duration for which user plan to hold the investment?\n2. User's convenience and safety.\n3. Taxation.\n4. Costs and Charges"
    )


from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain.tools import tool
import json

# class RAGSearchInput(BaseModel):
#     query: str = Field(..., description="Natural-language question to search in the document corpus.")
#     k: int = Field(6, description="How many passages to return (after de-dup/diversity).")
#     filters: Optional[Dict[str, Any]] = Field(
#         default=None,
#         description="Optional metadata filters, e.g., {'source': 'Sovereign Gold Bond Scheme 2025-26.pdf'}"
#     )

# @tool(args_schema=RAGSearchInput)
# def rag_search_tool(query: str, k: int = 6, filters: Optional[Dict[str, Any]] = None) -> str:
#     """
#     Search the indexed PDFs and return top-k quotable passages with source metadata.
#     Returns a JSON string (list of passages) for the agent to read in Observation.
#     """
#     # Apply filters (simple example: source exact match)
#     # FAISS retriever doesn't natively filter; we filter after retrieval using Document.metadata.
#     # Overfetch to improve precision@k, then diversify.
#     overfetch_k = max(k * 3, 5)
#     raw_docs: List = vecstore.similarity_search(query, k=overfetch_k)

#     if filters and "source" in filters:
#         src = filters["source"]
#         raw_docs = [d for d in raw_docs if (d.metadata.get("source") == src or d.metadata.get("file_path") == src)]

#     # Deduplicate by (source,page) to keep variety; then truncate to k
#     seen = set()
#     picked = []
#     for d in raw_docs:
#         key = (d.metadata.get("source") or d.metadata.get("file_path"), d.metadata.get("page"))
#         if key not in seen:
#             seen.add(key)
#             picked.append(d)
#         if len(picked) >= k:
#             break

#     # Prepare compact, quotable chunks with clean metadata for citation
#     passages = []
#     for d in picked:
#         print("=======passages==========",d,"/n")
#         passages.append({
#             "text": d.page_content.strip(),
#             "source": d.metadata.get("source") or d.metadata.get("file_path") or "unknown.pdf",
#             "page": d.metadata.get("page"),
#             "chunk_id": d.metadata.get("chunk", None)
#         })
#     # IMPORTANT: return a STRING (JSON) so your structured-chat agent prints it in Observation verbatim
#     return json.dumps(passages, ensure_ascii=False)


tools = [
    query_decomposition_tool,
    # rag_search_tool
]

from langchain_core.prompts import ChatPromptTemplate
messages = [
    (
        "system",
        """
        You are an AI financial advisor specializing in gold investments. 
        Your task is to provide a comprehensive recommendation to a user asking questions on gold investments.
        You have access to following tools and you must use these tools only directly on user query:

        {tools}

        You must think and reason step by step in cycles of:
        - Thought: You should always think about what to do, do not use any tool if it is not needed.
        - Action: A JSON block that calls a tool out of these {tool_names}.
        - Observation: You must stop token generation here. This will be the output of the tool. Do not create output of your own.

        You must only use this format:
        Thought: Describe your reasoning
        Action:
            ```json
            {{
                "action": "TOOL_NAME",
                "action_input": {{
                    "param1": "value1",
                    "param2": "value2"
                    // ... other parameters for the tool
                }}
            }}
            ```
        Observation: Result from tool. (You should always wait for tool result if a tool is called and the action is not Final Answer)
        ... (this Thought/Action/Observation cycle can repeat N times, untill you think you know the final answer.)

        To conclude:
        Thought: I know the final answer to user query.
        Action:
        ```json
            {{
                "action": "Final Answer",
                "action_input": "insert your final answer for the user"
            }}
        ```
        Tools available to you:
        - **query_decomposition_tool**: Analyze user's query and divides it into smaller steps.
          **Inputs**:
            - `query` (string): The user's input query. This MUST be a plain string, not a JSON object.
          **Example Action**:
            ```json
            {{
                "action": "query_decomposition_tool",
                "action_input": {{
                    "query": "Should I invest in gold through Sovereign Gold Bonds or buy physical gold this year?"
                }}
            }}
            ```

        Valid "action" values: "Final Answer" or {tool_names}

        Rules:
        1. Use only one tool per Action block.
        2. Always enclose Action in a JSON block inside triple backticks.
        3. Never invent tool outputs. Wait for the Observation.
        4. Only respond directly with "Final Answer" when you're fully confident and have used all necessary tools to format the final response.
        5. Never respond outside the above format.
        6. Do not include explanations after Final Answer.

        Start your reasoning now.
        """
    ),
    ("human", "{input}"),
    ("ai", "{agent_scratchpad}"),
]
prompt = ChatPromptTemplate.from_messages(messages)

# 3. Add manual truncation logic
# def cut_at_observation(text_or_msg):
#     text = text_or_msg.content if hasattr(text_or_msg, "content") else str(text_or_msg)
#     idx = text.find("\nObservation:")
#     return text if idx == -1 else text[:idx]

# truncate = RunnableLambda(cut_at_observation)

# 4. Wrap LLM with truncator
# llm_truncated = llm | truncate

agent = create_structured_chat_agent(
    llm = llm,
    tools = tools,
    prompt = prompt,
    stop_sequence=["\nObservation:", "\n\nObservation:"]
)

executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True
)

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
        # print(type(action), action)
        # print(f"\n---Thought: {action.log.strip().split('Action:')[0].replace('Thought:', '').strip()}")
        # print(f"---Action: {action.tool}")
        # try:
        #     # Attempt to parse action_input as JSON for cleaner display
        #     action_input_json = json.dumps(action.tool_input, indent=2)
        #     print(f"---Action Input:\n{action_input_json}\n ====action_input_type{type(action.tool_input)}")
        # except TypeError:
        #     print(f"---Action Input: {action.tool_input}")

    def on_tool_start(self, serialized: dict, input_str: str, inputs: dict[str, Any], run_id: UUID, **kwargs) -> None:
        """Run on tool start."""
        print("which toollll=====",{json.dumps(serialized)},"runID==========",{run_id}, "=======",{json.dumps(inputs)})
        print(f"--- 📊 Start input for tool:\n{input_str}") # Print raw output for clarity

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Run on tool end."""
        print(f"--- 📊 Observation:\n{str(output)}") # Print raw output for clarity

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Run on agent finish."""
        print("\n--- ✅ Agent Finished ---")
        print(f"--- 🏁 Final Answer:\n{finish.return_values['output']}")
        print("-------------------------\n")


my_callback_handler = CleanAgentStepCallbackHandler()

response = executor.invoke({
    "input": "Should I invest in gold through Sovereign Gold Bonds or buy physical gold this year?",
    "tool_names": ["query_decomposition_tool"]
},
config = { "callbacks": [my_callback_handler] }
)
print(f"\n🤖 Advisor: {response['output']}\n")
