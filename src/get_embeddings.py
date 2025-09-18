import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # Example embedding model

from scripts import get_pdf_paths

import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

from scripts import visualize_embeddings, visualize_embedding_model_performance

# --- Queries to analyse ---
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

def get_average_cosine_scores(model_id: str) -> list[float]:
    querywise_avg_scores = []
    # Loop through each query and its documents
    for query_num, query in QUERIES.items():
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
            model_name = model_id,
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
        
        # Get the embeddings for both the query and the retrieved documents
        # This is the crucial step to fix the error.
        query_embedding = embeddings.embed_query(query)
        retrieved_embeddings = embeddings.embed_documents(retrieved_texts)
        print(type(retrieved_embeddings), type(query_embedding))
        # visualize_embeddings(query_embedding, retrieved_embeddings)

        # Convert embeddings to numpy arrays if they are not already
        query_embedding = np.array(query_embedding).reshape(1, -1)
        retrieved_embeddings = np.array(retrieved_embeddings)
        print(type(retrieved_embeddings), type(query_embedding))
        # Now, combine the query embedding with the document embeddings
        all_embeddings = np.vstack([query_embedding, retrieved_embeddings])

        # Perform dimensionality reduction using t-SNE
        tsne = TSNE(n_components=2, perplexity=1, random_state=42, init='pca', learning_rate='auto')
        reduced_vectors = tsne.fit_transform(all_embeddings)

        # Separate the reduced vectors for plotting
        query_vec_2d = reduced_vectors[0]
        doc_vecs_2d = reduced_vectors[1:]
        # print("2D Query Vector:", type(query_vec_2d), query_vec_2d)
        # print("2D Document Vectors:", type(doc_vecs_2d),doc_vecs_2d)
        visualize_embeddings(query_vec_2d, doc_vecs_2d, model_id, query_num)

        # To get the scores, you can use similarity_search_with_score
        # retrieved_docs_with_scores = vecstore.similarity_search_with_score(query, k=5)
        # print(retrieved_docs_with_scores)

        # for doc, score in retrieved_docs_with_scores:
        #     cosine_sim = faiss_score_to_cosine(score)
        #     print("Chunk:", doc.page_content[:80], "...")
        #     print("FAISS L2 distance:", score)
        #     print("Cosine similarity:", cosine_sim)
        #     print()

        # Reshape the query embedding to be a 1D vector for easier calculation
        query_vector = query_embedding[0]

        # Calculate dot product and norms
        dot_product = np.dot(retrieved_embeddings, query_vector)
        norm_docs = np.linalg.norm(retrieved_embeddings, axis=1)
        norm_query = np.linalg.norm(query_vector)

        # Calculate cosine similarity for each document
        cosine_scores = dot_product / (norm_docs * norm_query)

        # Now you have an array of cosine scores, one for each retrieved document.
        # You can pair these with the documents to sort them.
        for i, score in enumerate(cosine_scores):
            print(f"Document {i+1} Cosine Similarity Score: {score}")

        # Sort the documents based on their scores in descending order
        sorted_indices = np.argsort(cosine_scores)[::-1]
        sorted_docs = [retrieved_docs[i] for i in sorted_indices]
        sorted_scores = cosine_scores[sorted_indices]

        print("\nDocuments ranked by cosine similarity:")
        for i in range(len(sorted_docs)):
            print(f"Rank {i+1}: Score = {sorted_scores[i]:.4f}, Document = {sorted_docs[i].page_content[:50]}...")

        average_score = np.mean(sorted_scores)
        print(f"\nAverage Cosine Score: {average_score:.4f}")

        querywise_avg_scores.append(average_score)
    
    return querywise_avg_scores


def analyse_models():
    # Create a dictionary to store the average scores for each query
    average_scores = {}

    model_ids = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "intfloat/e5-base-v2",
        "BAAI/bge-small-en-v1.5",
        "intfloat/e5-large-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]

    for model_id in model_ids:
        print(f"\nAnalyzing model: {model_id}")
        avg_scores = get_average_cosine_scores(model_id)
        overall_avg = np.mean(avg_scores)
        print(f"Overall Average Cosine Score for {model_id}: {overall_avg:.4f}")
        average_scores[model_id] = overall_avg

    return average_scores

if __name__ == "__main__":
    print("-" * 50)
    print("Final Summary of Average Cosine Scores:")
    average_scores = analyse_models()
    visualize_embedding_model_performance(average_scores)
    for query, score in average_scores.items():
        print(f"  Model: '{query}'")
        print(f"  Average Score: {score:.4f}")

