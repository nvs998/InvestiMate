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
                f"1. Profitability.\n2. Revenue Growth (Sales).\n3. Valuation Metrics.\n4. Debt and Liquidity."
            }

        def rag_search2(state: State):
            context = '''Sky+4website+4Value Research Online+4Reuters.
                        However, consider these before switching:
                        • Valuation: Asian Paints and Berger have traditionally traded at high P/E and P/B multiples; 
                        check trailing ratios and growth forecasts.
                        • Long-term brand strength: Asian Paints still has deeper pan-India reach and premium 
                        brand equity.
                        • Upcoming Q1 FY26 results (Asian Paints on July 29, Berger on August 5) will provide 
                        updated guidance and may shift the momentum pattern Reuters+11HDFC 
                        Sky+11BlinkX+11Reuters.
                        ✅ Bottom Line
                        If your decision is based on recent fundamentals and growth trajectory, Berger Paints 
                        currently appears the stronger choice. They posted consistent growth, delivered better margins, 
                        and beneﬁted from a robust industrial segment.
                        Asian Paints, while still a market leader, is grappling with weak demand, eroding share, and

                        Berger Paints is demonstrating stronger momentum right now:
                        • More resilient revenue and proﬁt growth amid broader industry softness.
                        • Industrial segment strength provides diversiﬁcation beyond consumer paints.
                        • Superior execution on cost and margins in a challenging macro environment.
                        Asian Paints is facing serious headwinds:
                        • Sharp earnings decline in Q4.
                        • Loss of market share to aggressive competitors like Birla Opus.
                        • Higher operating and marketing spend, leading to margin compression Wikipedia+15The 
                        Economic Times+15website+15HDFC Sky+3Screener+3The Economic Times+3Perplexity 
                        AI+2HDFC Sky+2Value Research Online+2Trendlyne.comyoutube.com+4Reuters+4HDFC 
                        Sky+4Perplexity AI+7Reuters+7HDFC Sky+7Reuters+1Reuters+1HDFC 
                        Sky+4website+4Value Research Online+4Reuters.
                        However, consider these before switching:

                        and beneﬁted from a robust industrial segment.
                        Asian Paints, while still a market leader, is grappling with weak demand, eroding share, and 
                        margin pressure—making it less compelling in the near term.
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
            query = "Should I switch from Asian Paints to Berger Paints based on recent quarterly results?"

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
            "question": "Should I switch from Asian Paints to Berger Paints based on recent quarterly results?",
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
        normal_answer = llm.invoke('Should I switch from Asian Paints to Berger Paints based on recent quarterly results?')
        print("normal answer===========",normal_answer)
    except Exception as e:
        logger.exception("An error occurred")

if __name__ == "__main__":
    main()


