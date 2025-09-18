import pandas as pd

# --- Main App ---
def main():
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
    
    # 2. Define the path to your knowledge base
    PDF_PATHS = [
        "../query1_data/Gold ETF vs Physical Gold.pdf",
        "../query1_data/Sovereign Gold Bond Scheme 2025-26.pdf"
    ]
    
    # 3. Initialize core components
    llm = get_mistral_llm() # Load your LLM once
    rag_system = RAGSystem(llm=llm, pdf_paths=PDF_PATHS)
    
    
    # --- PHASE 2: EXECUTION (Loop for Each Query) ---
    
    all_results = []
    
    print("\n--- Starting Dynamic Query Execution Loop ---")
    # Loop from 1 to 10 (or however many query folders you have)
    for query_num in range(1, 11):
        
        # Get the actual question text from our map
        query_text = QUERIES.get(query_num)
        if not query_text:
            print(f"No question text found for query {query_num}, skipping.")
            continue

        print(f"\n--- Processing Query {query_num}/{len(QUERIES)}: {query_text} ---")
        
        # 1. Dynamically fetch the PDF paths for the current query
        pdf_paths = get_pdf_paths(query_num)
        
        # 2. If no documents are found for this query, skip it
        if not pdf_paths:
            print(f"No PDFs found for query {query_num}, skipping.")
            continue
        
        print(f"Found {len(pdf_paths)} documents: {pdf_paths}")
        
        # 3. Initialize a fresh RAGSystem for this specific query and its documents
        #    This is the key change to ensure data isolation between queries.
        rag_system = RAGSystem(llm=llm, pdf_paths=pdf_paths)
        
        # 4. Run the RAG Graph for the query
        rag_result = rag_system.run_query(query_text)
        
        # 5. Generate a baseline "vanilla" answer for comparison
        vanilla_answer = llm.invoke(query_text)
        
        # 6. Store the results
        all_results.append({
            "question": query_text,
            "contexts": rag_result["context"],
            "answer_rag": rag_result["answer"],
            "answer_vanilla": vanilla_answer,
            "pointers": rag_result["pointers"]
        })

    # --- PREPARE FOR PHASE 3: EVALUATION ---
    if not all_results:
        print("No results were generated. Exiting.")
        return

    results_df = pd.DataFrame(all_results)
    print("\n\n--- All Queries Processed. Results: ---")
    print(results_df.to_string())
    
    results_df.to_csv("rag_evaluation_data.csv", index=False)
    print("\nResults saved to rag_evaluation_data.csv")

if __name__ == "__main__":
    main()