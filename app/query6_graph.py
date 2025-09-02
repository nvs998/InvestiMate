from logger import logger

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

# RAG Setup: loaders, splitter, embeddings, vector store
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

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
                f"1. Exact Down Payment Target.\n2. Current Savings.\n3. Monthly Savings Capacity.\n4. Inflation Expectations.\n5. Potential for Unexpected Expenses."
            }

        def rag_search2(state: State):
            context = '''Recommended Asset Allocation (General Rule of Thumb)
                        This is based on the age-old 100 minus age rule, adjusted for your goal rather than your age.
                        Why Equity Should Dominate
                        Pros:
                        • Historically, Indian equity returns average 12–15% p.a. over 10 years.
                        • Ideal for longer horizons: gives you time to ride out market volatility.
                        • Helps build a bigger corpus faster than debt instruments.
                        How to Invest:
                        • SIP in diversiﬁed equity mutual funds (e.g. large-cap, ﬂexi-cap, ELSS if tax saving).
                        • Consider index funds or ETFs if you want lower cost.
                        Why Debt Still Matters
                        Role:
                        • Adds stability to your portfolio.
                        • Helps protect capital as you near the goal.
                        • Rebalances volatility during market downturns.
                        How to Invest:
                        • Debt mutual funds (short-term, target maturity funds).
                        • PPF, EPF (if available), ﬁxed deposits for part of the amount.
                        Glide Path Strategy (Dynamic Rebalancing)

                        This ensures your house down payment is protected from late-stage market crashes.
                        Quick Illustration (₹10,000/month SIP)
                        *Assuming 12% CAGR for equity and 6% CAGR for debt.
                        Years Left to 
                        Goal Equity Debt
                        10–7 years 70% 30%
                        6–4 years 60% 40%
                        3 years left 40% 60%
                        1–2 years left 20–
                        30%
                        70–
                        80%
                        Asset Class Monthly 
                        Allocation
                        Expected 10Y 
                        Corpus
                        Equity 
                        (70%) ₹7,000/month ₹17–18 lakhs*
                        Debt (30%) ₹3,000/month ₹5–5.5 lakhs*
                        Total ₹10,000/month ₹22–23 lakhs

                        • PPF, EPF (if available), ﬁxed deposits for part of the amount.
                        Glide Path Strategy (Dynamic Rebalancing)
                        As the goal approaches, gradually reduce equity exposure:
                        Asset 
                        Class
                        Suggested 
                        Allocation Why
                        Equity 60–70% Higher growth potential over 10 years, can beat 
                        inﬂation
                        Debt 30–40% Reduces volatility, provides stability
                        '''
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
            query = "Should I allocate more to equity or debt in my portfolio if I’m planning to buy a house in 10 years?"

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
            "question": "Should I allocate more to equity or debt in my portfolio if I’m planning to buy a house in 10 years?",
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
        normal_answer = llm.invoke('Should I allocate more to equity or debt in my portfolio if I’m planning to buy a house in 10 years?')
        print("normal answer===========",normal_answer)
    except Exception as e:
        logger.exception("An error occurred")

if __name__ == "__main__":
    main()


