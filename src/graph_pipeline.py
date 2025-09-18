from logger import logger

from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
import operator
from typing import Annotated, TypedDict
import uuid

from llm_models.llama3_wrapper import get_llama3_llm, get_mistral_llm
from langgraph.graph.message import add_messages

# RAG Setup: loaders, splitter, embeddings, vector store
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


import torch
import uuid
from typing import TypedDict, List

# LangChain & LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM

# Your Local LLM Wrapper (assuming it's in llm_models/mistral_wrapper.py)
from llm_models.mistral_wrapper import get_mistral_llm # Or your llama3 wrapper

# --- State for Graph ---
class RAGState(TypedDict):
    question: str
    pointers: str
    context: list[str]
    answer: str

class RAGSystem:
    def __init__(self, llm: LLM, pdf_paths: List[str]):
        """
        Initializes the RAG system by setting up the LLM, retriever, and compiling the graph.
        """
        self.llm = llm
        
        # 1. Build the Retriever once
        print("Building vector store from PDFs...")
        docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        splits = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        print("Vector store built successfully.")

        # 2. Compile the LangGraph
        self.graph = self._compile_graph()

    def query_decomposition_tool(self, state: RAGState) -> dict:
        """
        Dynamically generates pointers based on the question.
        NOTE: For a real system, this would be another LLM call.
              For this example, we'll keep the placeholder but show where the question would be used.
        """
        question = state["question"]
        print(f"--- Decomposing question: {question} ---")
        
        # In a real implementation, you would do:
        # prompt = f"Given the user question '{question}', what are the key points to consider for a detailed answer? List them."
        # pointers = self.llm.invoke(prompt)
        
        # Using placeholder pointers for now as in your original code.
        pointers = "1. Duration for which user plan to hold the investment?\n2. User's convenience and safety.\n3. Taxation.\n4. Costs and Charges"
        return {"pointers": pointers}

    # --- NODE DEFINITIONS ---
    # These are now methods of the class. They operate on the state.

def main():
    logger.info("Starting program")
    try:
        llm = get_mistral_llm()

        class State(TypedDict):
            question: str
            answer: str
            context: str
            summary: str
            pointers: str

        # Define the logic to call the model
        def call_model(state: State):
            context = state["context"]
            question = state["question"]
            pointers = state["pointers"]
            
            prompt = f"""Answer this question: {question} \nConsider these points for your analysis: {pointers} \nUse this context to answer: {context}.\nFinal Answer:"""
            
            # llm = get_mistral_llm()
            answer = llm.invoke(prompt)
            
            return { "answer" : answer }

        def final_answer(state: State):
            raw_output = state["answer"]
            answer = raw_output.lower().split('final answer:', 1)
            if len(answer) > 1:
                print("\n=============working===========\n")
                answer = answer[1].strip()
            else:
                answer = raw_output
                
            return { "answer" : answer }


        def query_decomposition_tool(state: State):
            return { "pointers" :
                f"1. Duration for which user plan to hold the investment?\n2. User's convenience and safety.\n3. Taxation.\n4. Costs and Charges"
            }

        def rag_search2(state: State):
            context = '''on behalf of the Government of India. It allows investors to invest in gold in a non-physical form, making it
            a safer and more eﬃcient alternative to buying physical gold. These bonds are denominated in grams of
            gold and oﬀer an annual interest rate of 2.50 percent over and above the potential capital gains linked to the
            market price of gold.
            One of the key beneﬁts of this scheme is that it eliminates the risks associated with storing physical gold,
            such as theft or damage. Additionally, it oﬀers tax beneﬁts—especially if the bonds are held till maturity,
            as the capital gains are tax-free.
            The upcoming SGB issue under the upcoming Sovereign Gold Bond Scheme for the ﬁnancial year 2024–
            25 presents a new opportunity for investors to buy these bonds at government-declared rates. It is an
            ideal option for those looking to diversify their portfolio while beneﬁting from gold’s price appreciation

            Tradability and liquidity: SGBs are tradable on stock exchanges, making them accessible for early
            exit if required.
            Small-ticket investment: You can buy as little as one gram of gold, making them suitable for all
            budgets.
            Tax beneﬁts: Capital gains are exempt if held till maturity.
            Collateral for loans: Investors can use SGBs as security when applying for loans.
            Stay updated on the sovereign gold bond next issue date and plan your investments wisely for the next
            SGB issue date.
            Understanding the upcoming sovereign gold bond issues
            Sovereign Gold Bonds (SGBs) are a smart alternative to buying physical gold, introduced by the
            Government of India to oﬀer safer, more rewarding investment options. The sovereign gold bond scheme
            Staying informed about the upcoming SGB dates ensures that you do not miss the opportunity to invest

            Tenure: Bonds have a maturity period of 8 years, with an option for premature redemption after 5
            years.
            Interest rate: Oﬀers a ﬁxed return of 2.50% per annum, paid semi-annually.
            Issue price: Based on the average closing price of gold during the preceding week.
            Eligibility: Open to resident individuals, Hindu Undivided Families (HUFs), trusts, universities, and
            charitable institutions.
            Tradability: Bonds are listed and tradable on recognised stock exchanges.
            Minimum investment: Just 1 gram of gold, making it accessible to small investors.
            Tax beneﬁts: Capital gains on redemption are exempt from tax if held until maturity.
            Loan collateral: Bonds can be used as collateral for secured loans.
            Stay informed about the upcoming sovereign gold bond scheme 2025-26 to make the most of these
            beneﬁts.
            Next sovereign gold bond issue date: Important dates to remember'''

            return { "context" : context }

        def rag_search(state: State):
            # 1) Load your PDFs
            pdf_paths = [
                "../query1_data/Gold ETF vs Physical Gold.pdf",
                "../query1_data/Sovereign Gold Bond Scheme 2025-26.pdf"
            ]

            docs = []
            for p in pdf_paths:
                loader = PyPDFLoader(p)
                # Each page comes with metadata; loader returns Document objects
                docs.extend(loader.load())

            # 2) Split into chunks (keep overlaps for context continuity)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=900, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
            )
            splits = splitter.split_documents(docs)

            # 3) Embeddings (no API key needed)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # 4) Build FAISS index
            vecstore = FAISS.from_documents(splits, embeddings)

            # 5) Create a retriever (top_k configurable inside the tool)
            retriever = vecstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

            # Define your query
            query = "What are the benefits of Sovereign Gold Bonds compared to physical gold?"

            # Search for relevant chunks
            retrieved_docs = vecstore.similarity_search(query, k=3)

            # Combine retrieved content
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            return {"context": ["context"]}

        # Define the summarization function
        def summarize_conversation(state: State):
            final_answer = state["answer"]
            summary = state.get("summary", "")
            if summary:
                summary_message = f"This is summary of the answer: {summary}"
            else:
                summary_prompt = f"Clean the below text properly without changing its meaning or adding additional information. Do not extend the answer and just the response as,\nSummary:\n\nText:{final_answer}"
                # llm = get_mistral_llm()
                summary_message = llm.invoke(summary_prompt)
            
            return { "summary" : summary_message }


        # Define the graph
        builder = StateGraph(State)
        # builder.add_node("search_web", search_web)
        builder.add_node("query_decomposition_tool", query_decomposition_tool)
        builder.add_node("rag_search", rag_search2)
        builder.add_node("generate_answer", call_model)
        builder.add_node("extract_answer", final_answer)
        builder.add_node("summarize_conversation", summarize_conversation)
        builder.add_edge(START, "query_decomposition_tool")
        # builder.add_edge(START, "search_web")
        builder.add_edge("query_decomposition_tool", "rag_search")
        # builder.add_edge("search_web", "generate_answer")
        # builder.add_edge("generate_answer", "rag_search")
        builder.add_edge("rag_search", "generate_answer")
        builder.add_edge("generate_answer", "extract_answer")
        # builder.add_edge("extract_answer", "summarize_conversation")
        builder.add_edge("extract_answer", END)

        # Compile the graph with memory
        # memory = MemorySaver()
        memory = InMemorySaver()
        graph = builder.compile(checkpointer=memory)

        # if __name__ == "__main__":
            # Example question you want to test
        initial_state = {
            "question": "Should I invest in gold through Sovereign Gold Bonds or buy physical gold this year?",
            "answer": "",
            "context": "",
            "summary": "",
            "pointers": "",
            "thread_id": str(uuid.uuid4())
        }
        result = graph.invoke(initial_state, config = {"configurable": {"thread_id": "1"}})

        # Print the output (final state)
        print("Final State:")
        print(result)
        print("\nGenerated Answer:")
        print(result.get("answer"))

        # llm2 = get_mistral_llm()
        normal_answer = llm.invoke('Should I invest in gold through Sovereign Gold Bonds or buy physical gold this year?')
        print("normal answer===========",normal_answer)
    except Exception as e:
        logger.exception("An error occurred")

if __name__ == "__main__":
    main()
