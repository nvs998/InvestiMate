from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# This dictionary holds all the factors that are important to consider for answering each query.
QUERY_POINTERS = {
    1: """
    - How long do you plan to hold the investment?
    - Convenience and Safety.
    - Taxation.
    - Costs and Charges.
    """,

    2: """
    - Liquidity
    - Taxation
    - Returns on Investment
    """,

    3: """
    - Current Dividend Yield
    - Dividend Payout Ratio
    - Company Financial Health and Stability
    """,

    4: """
    - Market Indices Performance
    - Societal Impact - Impact on Quality of Life/Solving Global Challenges
    - Financial Performance - Revenue Growth Rate (Year-over-Year), Profit Margins (Gross, Operating, Net), Market Capitalization Growth, R&D Spending as a Percentage of Revenue, Return on Investment (ROI)
    """,

    5: """
    - Investment Goal
    - Risk Appetite
    - Liquidity Requirements
    - Expense Ratio (for ELSS) and Brokerage/Transaction Costs (for Direct Stocks)
    """,

    6: """
    - Analyzing Recent Quarterly Results
    - Profitability
    - Revenue Growth (Sales)
    - Valuation Metrics
    - Debt and Liquidity
    """,

    7: """
    - Revenue Growth
    - Stock Price Performance
    - Analyst Ratings and Reports
    """,

    8: """
    - Exact Down Payment Target
    - Current Savings
    - Monthly Savings Capacity
    - Inflation Expectations
    - Potential for Unexpected Expenses
    """,

    9: """
    - Distinction between Capital Gains and Interest Income
    - Tax Rate and Indexation
    - Holding Period
    - Manner of Sale/Withdrawal
    """,

    10: """
    - Inflation
    - Taxation of Returns
    - Potential for Returns
    """
}

def get_pdf_paths(query_number: int) -> list[str]:
    """ Finds all PDF file paths for a given query number. """
    base_data_dir = Path("data") / "raw"
    query_dir = base_data_dir / f"query{query_number}"
    if not query_dir.is_dir():
        print(f"Warning: Directory not found for query {query_number}: {query_dir}")
        return []
    pdf_path_strings = [str(p) for p in query_dir.glob("*.pdf")]
    return pdf_path_strings

def get_pointers_for_query(query_number: int) -> str:
    """ Retrieves a predefined string of pointers for a given query number. """
    return QUERY_POINTERS.get(query_number, "")


if __name__ == "__main__":
    print(get_pointers_for_query(1))