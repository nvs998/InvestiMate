from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
import operator
from typing import Annotated, TypedDict
import uuid

from llm_models.llama3_wrapper import get_llama3_llm, get_mistral_llm
from langgraph.graph.message import add_messages

# Define the state class
# class State(TypedDict, total=False):
#     messages: Annotated[list, add_messages]


class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]
    summary: str


# Define the logic to call the model
def call_model(state: State):
    context = state["context"]
    question = state["question"]
    answer_template = f"""Answer the question: {question} \nusing this context: {context}"""
    print(answer_template)
    llm = get_mistral_llm()
    answer = llm.invoke(answer_template)
    print("answer=========",answer)
    return {"answer": answer}

# Define the search functions
def search_web(state: State):
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['question'])
    formatted_search_docs = "\n\n---\n\n".join(
        [f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>' for doc in search_docs]
    )
    return {"context": [formatted_search_docs]}

def query_decomposition_tool(state: State):
    # Access arguments directly as passed by Pydantic
    # Use f-strings for formatting and comma for thousands separator
    return {"context":
        f"Based on the question that user asked, "
        f"We should analyse factors like:\n"
        f"1. Duration for which user plan to hold the investment?\n2. User's convenience and safety.\n3. Taxation.\n4. Costs and Charges"
    }

def search_wikipedia(state: State):
    search_docs = WikipediaLoader(query=state['question'], load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>' for doc in search_docs]
    )
    return {"context": [formatted_search_docs]}

# Define the summarization function
def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = f"This is summary of the answer: {summary}\n\nExtend the summary by taking into account the new messages above:"
    else:
        summary_message = "Create a summary of the conversation above:"
    messages = [HumanMessage(content=state["question"])] + [SystemMessage(content=state["context"])] + [SystemMessage(content=state["answer"])]
    llm = get_mistral_llm()
    response = llm.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["context"][:-2]]
    return {"summary": response, "messages": delete_messages}

# Define the graph
builder = StateGraph(State)
# builder.add_node("search_web", search_web)
builder.add_node("query_decomposition_tool", query_decomposition_tool)
# builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", call_model)
builder.add_node("summarize_conversation", summarize_conversation)
builder.add_edge(START, "query_decomposition_tool")
# builder.add_edge(START, "search_web")
builder.add_edge("query_decomposition_tool", "generate_answer")
# builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", "summarize_conversation")
builder.add_edge("summarize_conversation", END)

# Compile the graph with memory
# memory = MemorySaver()
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    # Example question you want to test
    initial_state = {
        "question": "Should I invest in gold through Sovereign Gold Bonds or buy physical gold this year?",
        "answer": "",
        "context": [],
        "summary": "",
        "thread_id": str(uuid.uuid4())
    }
    result = graph.invoke(initial_state, config = {"configurable": {"thread_id": "1"}})

    # Print the output (final state)
    print("Final State:")
    print(result)
    print("\nGenerated Answer:")
    print(result.get("summary"))

    # # Run the graph with the initial state
    # for event in graph.stream({"messages": [{"role": "user", "content": "Should I invest in gold through Sovereign Gold Bonds or buy physical gold this year?"}]}):
    #     for value in event.values():
    #         print("Assistant:", value["messages"][-1].content)