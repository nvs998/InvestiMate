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
                f"1. Current Dividend Yield.\n2. Dividend Payout Ratio.\n3. Company Financial Health and Stability."
            }

        def rag_search2(state: State):
            context = '''Sign InCompanies: 10,582 total market cap: $127.029 T
                        Market cap Revenue Earnings Dividend yield
                        Dividend yield history for Oil & Natural Gas
                        (ONGC.NS)
                        Oil & Natural Gas (stock symbol: ONGC.NS) dividend yield (TTM) as of July 26, 2025 : 5.66%
                        Average dividend yield, last 5 years: 5.51%
                        🇺🇸  English $ USD 
                        Company name, ticker...
                        Oil & Natural Gas 
                        O N G C .N S
                        🛢  Oil&Gas⚡  Energy
                        Categories
                        #647
                        Rank
                        $34.96 B
                        Marketcap
                        🇮🇳 India
                        Country
                        $2.78
                        Share price
                        -1.85%
                        Change (1 day)
                        -30.69%
                        Change (1 year)
                        Oil and Natural Gas Corporation or ONGC for short, is an Indian Multinational Crude Oil
                        and Gas Corporation. ONGC's operations include conventional exploration and production,
                        refining and progressive development of alternate energy sources like coal-bed methane
                        and shale gas.
                        More
                        26/07/2025, 21:33 Oil & Natural Gas (ONGC.NS) - Dividend Yield

                        and shale gas.
                        More
                        26/07/2025, 21:33 Oil & Natural Gas (ONGC.NS) - Dividend Yield

                        Home ETPrimeMarkets Market Data
                        NewsIndustrySMEPoliticsWealthMFTechAI CareersOpinionNRI Panache ••
                        Stocks Options IPOs/FPOs Expert Views Investment Ideas Commodities Forex Live Stream! AIF PMS Crypto Bonds More
                        Business News›Markets›Stocks›Earnings›Coal India Q4 Results: Profit rises 12% YoY to Rs 9,593 crore; Co declares Rs 5.15 per share as final dividend for FY25 
                        Coal India Q4 Results: Profit rises 12% YoY to Rs9,593 crore; Co declares Rs 5.15 per share as finaldividend for FY25
                        Vijay Shekhar Sharma’s resilience fuels the rise ofIndia’s merchant payments champion
                        SynopsisCoal India Q4 Results: State-owned Coal India on Wednesday reported 12 percent growth in itsconsolidated net profit at Rs 9,593 crore in the fourth quarter, compared with Rs 8,530crore in the last year quarter.
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
            query = "Which one offers better dividend yield – Coal India or ONGC?"

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
            "question": "Which one offers better dividend yield – Coal India or ONGC?",
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
        normal_answer = llm.invoke('Which one offers better dividend yield – Coal India or ONGC?')
        print("normal answer===========",normal_answer)
    except Exception as e:
        logger.exception("An error occurred")

if __name__ == "__main__":
    main()


