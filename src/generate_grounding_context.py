from scripts import get_pdf_paths
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def generate_and_save_context():
    QUERIES = {
        # (Basic Investment)
        1: "Should I invest in gold through Sovereign Gold Bonds or buy physical gold this year?",
        2: "Is it better to invest in a Public Provident Fund (PPF) or a SIP in a debt mutual fund for long-term savings?",
        # (Equity Stock Comparison)
        3: "Which one offers better dividend yield – Coal India or ONGC?",
        # (Sector vs Sector – Investment Decision Focused)
        4: "Which sector is doing better in 2025 – pharma or IT?",
        # (Market-Oriented)
        5: "Which is better for long-term wealth: investing in ELSS funds or in blue-chip stocks directly?",
        # (Stock vs Stock – Performance Focused)
        6: "Should I switch from Asian Paints to Berger Paints based on recent quarterly results?",
        7: "Which stock is performing better this quarter – Infosys or TCS?",
        # (Asset Allocation and Diversification)
        8: "Should I allocate more to equity or debt in my portfolio if I am planning to buy a house in 10 years?",
        # (Commodities: Gold & Silver)
        9: "What are the tax implications of selling Sovereign Gold Bonds before maturity in India?",
        # (Risk & Goal-Based Planning)
        10:"Are tax-saving fixed deposits better than ELSS for someone in the 30% tax bracket?"
    }

    MODEL_ID = "intfloat/e5-base-v2"
    all_contexts = []
    for query_num, query in QUERIES.items():
        try:
            pdf_paths = get_pdf_paths(query_num)
            print(f"Processing query: {query_num} with '{len(pdf_paths)}' documents.")
            docs = []
            for p in pdf_paths:
                # Path to PDF file
                path = p
                loader = PyPDFLoader(path)
                # Each page comes with metadata; loader returns Document objects
                docs.extend(loader.load())

            # Split into chunks (keep overlaps for context continuity)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
            )
            splits = splitter.split_documents(docs)

            # Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name = MODEL_ID,
                encode_kwargs={"normalize_embeddings": True}
            )

            # Build FAISS index
            vecstore = FAISS.from_documents(splits, embeddings)

            # Create a retriever (top_5 configurable inside the tool)
            retriever = vecstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

            # Define your query
            query = query

            # Search for relevant chunks
            retrieved_docs = vecstore.similarity_search(query, k=5)

            # Get the text content of the retrieved documents
            retrieved_texts = [doc.page_content for doc in retrieved_docs]
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            all_contexts.append({
                "query_num": query_num,
                "query": query,
                "generated_context": context
            })
            print(f"  -> Successfully generated context for query {query_num}.")
        except Exception as e:
            print(f"  -> An error occurred while processing query {query_num}: {e}")
    
    # After processing all queries, create a DataFrame and save to CSV
    if not all_contexts:
        print("No contexts were generated. CSV file will not be created.")
        return

    df = pd.DataFrame(all_contexts)
    # Construct the full path to the desired folder
    output_dir = os.path.join("data", "generated")

    # Create the nested directories if they don't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full file path including the directory and filename
    filename = f"query_context.csv"
    output_csv_path = os.path.join(output_dir, filename)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\n✅ Process complete. Contexts saved to '{output_csv_path}'")

# To run the script:
if __name__ == "__main__":
    generate_and_save_context()