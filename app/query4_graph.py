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
                f"1. Market Indices Performance.\n2. Societal Impact - Impact on Quality of Life/Solving Global Challenges.\n3. Financial Performance - Revenue Growth Rate (Year-over-Year), Profit Margins (Gross, Operating, Net), Market Capitalization Growth, R&D Spending as a Percentage of Revenue, Return on Investment (ROI)."
            }

        def rag_search2(state: State):
            context = '''and regulatory ease, strengthening prospects in UK/an OECD markets The Economic 
                        Times+1The Economic Times+1.
                        • Dr. Reddy’s posted record quarterly revenue in Q1 FY26 (+11%) and proﬁt rose modestly 
                        ~2%, with strong momentum in European generics (up 142%) Wikipedia+15Reuters+15The 
                        Times of India+15.
                        Overall, pharma continues to demonstrate consistent earnings growth, resilience in macro 
                        downturns, and favorable policy tailwinds.
                        IT Sector: Under Pressure, Mixed Signals 
                        • The Nifty IT index is down ~14% year-to-date in 2025, making it the worst-performing 
                        sector so far. Large-cap IT names like TCS, Infosys, and HCL have dropped sharply—TCS 
                        down ~21%, trading ~30% below its 52-week high India Infoline+1The Economic Times+1.
                        • Earnings are being squeezed by slowing discretionary spending, macroeconomic headwinds,

                        Pharmaceutical Sector: Strong and Resilient 
                        • The Nifty Pharma index rose ~11.5% between July 2024 and June 2025, notably beating 
                        the Nifty 50 (~5.7%) and most other major sectors 
                        PwC+15indiamacroindicators.co.in+15The Times of India+15.
                        • Monthly growth in April 2025 was 7.8% YoY in pharma sales (₹19,711 crore), 
                        underscoring healthy demand and solid pricing dynamics The Times of India.
                        • The broader pharmaceutical market grew ~8.4% in FY24-25, driven by chronic therapy 
                        demand domestically and strong exports from generic makers like Sun Pharma, Cipla, and 
                        Dr Reddy’s Jainam Broking Ltd+14Business Standard+14Reuters+14.
                        • The India-UK free trade agreement is likely to boost pharma exports through tariff relief 
                        and regulatory ease, strengthening prospects in UK/an OECD markets The Economic 
                        Times+1The Economic Times+1.

                        Key Sector Stats
                        Metric Pharma IT
                        Index Performance 
                        (1 yr) 11.5% –14% (Nifty IT YTD)
                        Sales Growth FY25 ~8–8.4% ~3–5%
                        Earnings Trend Positive, steady Weak, margin pressure
                        Export Momentum Rising (US, UK futures) Mixed – some growth, some headwinds
                        Outlook Stable growth + 
                        tailwinds
                        Under stress; cautious medium-term 
                        outlook
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
            query = "Which sector is doing better in 2025 – pharma or IT?"

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
            "question": "Which sector is doing better in 2025 – pharma or IT?",
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
        normal_answer = llm.invoke('Which sector is doing better in 2025 – pharma or IT?')
        print("normal answer===========",normal_answer)
    except Exception as e:
        logger.exception("An error occurred")

if __name__ == "__main__":
    main()


