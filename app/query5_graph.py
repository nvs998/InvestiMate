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
                f"1. Investment Goal.\n2. Risk Appetite.\n3. Liquidity Requirements.\n4. Expense Ratio (for ELSS) and Brokerage/Transaction Costs (for Direct Stocks)."
            }

        def rag_search2(state: State):
            context = '''Side-by-Side Comparison
                        When to Choose What?
                        Choose ELSS Funds if:
                        • You want tax savings under Section 80C.
                        • You’re not an expert in stock picking.
                        • You prefer diversiﬁed risk with long-term compounding.
                        • You’re starting with small amounts via SIP (e.g. ₹5,000/month).
                        • You want a set-it-and-forget-it approach for 3–5+ years.
                        Choose Blue-Chip Stocks if:
                        • You’re conﬁdent in evaluating stocks and sectors.
                        • You want full control over what you own.
                        • You’re building a custom long-term portfolio.
                        • You’re okay with market cycles and monitoring regularly.
                        • You don’t need Section 80C deductions or already maxed them out.
                        Feature ELSS Mutual Funds Blue-Chip Stocks (Direct Investment)
                        Tax Beneﬁts
                        ✅  Up to ₹1.5L under Section 80C
                        ❌  No tax beneﬁt
                        Lock-in Period 3 years (mandatory) No lock-in
                        Returns Potential 10–14% historically (managed 
                        portfolio)
                        8–15%+ (if picked & timed well)

                        stream.
                        • Liquidity: Blue-chip stocks are highly liquid, meaning they can be easily bought and sold in 
                        the market.
                        • Direct Control: You have full control over your investment decisions, allowing you to pick 
                        speciﬁc companies you believe in.
                        • Transparency: Financial information and performance of blue-chip companies are 
                        generally well-documented and easily accessible.
                        • No Expense Ratio: You don't pay an expense ratio like with mutual funds.
                        Drawbacks:
                        • No 80C Tax Beneﬁt: Investing in blue-chip stocks directly does not provide any tax 
                        deduction under Section 80C.
                        • Concentration Risk: Investing in a few blue-chip stocks directly exposes you to higher 
                        concentration risk compared to a diversiﬁed mutual fund. If one of your chosen 
                        stocks underperforms signiﬁcantly, it can have a larger impact on your portfolio.

                        🔧 Sample Strategy
                        For long-term wealth creation, many investors combine both:
                        • 60–70% in ELSS (or large-cap funds) to get tax beneﬁt + diversiﬁcation.
                        • 30–40% in handpicked blue-chip stocks to participate in speciﬁc sectoral stories (e.g. IT, 
                        banking, FMCG).
                        '''
            return { "context" : context }

        def rag_search(state: State):
            # 1) Load your PDFs
            pdf_paths = [
                "../query5_data/ELSS vs Blue chip.pdf",
                "../query5_data/ELSS vs Blue chip2.pdf"
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
            query = "Which is better for long-term wealth: investing in ELSS funds or in blue-chip stocks directly?"

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
            "question": "Which is better for long-term wealth: investing in ELSS funds or in blue-chip stocks directly?",
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
        normal_answer = llm.invoke('Which is better for long-term wealth: investing in ELSS funds or in blue-chip stocks directly?')
        print("normal answer===========",normal_answer)
    except Exception as e:
        logger.exception("An error occurred")

if __name__ == "__main__":
    main()


