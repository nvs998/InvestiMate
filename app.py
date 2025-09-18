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

    all_results = []
    
    print("\n--- Starting Query Execution Loop ---")
    for i, query in enumerate(QUERIES):
        print(f"\n--- Processing Query {i+1}/{len(QUERIES)}: {query} ---")
        
        # Run the RAG Graph for the query
        rag_result = rag_system.run_query(query)
        
        # Generate a baseline "vanilla" answer for comparison
        vanilla_answer = llm.invoke(query)
        
        # Store the results
        all_results.append({
            "question": query,
            "contexts": rag_result["context"],
            "answer_rag": rag_result["answer"],
            "answer_vanilla": vanilla_answer,
            "pointers": rag_result["pointers"]
        })

    # --- PREPARE FOR PHASE 3: EVALUATION ---

    # Convert results to a pandas DataFrame for easy viewing and saving
    results_df = pd.DataFrame(all_results)
    
    print("\n\n--- All Queries Processed. Results: ---")
    print(results_df.to_string())
    
    # Save results to a CSV file, ready for the RAGAs evaluation script
    results_df.to_csv("rag_evaluation_data.csv", index=False)
    print("\nResults saved to rag_evaluation_data.csv")
    print("This file is now ready to be loaded for Phase 3: RAGAs Evaluation.")
    
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