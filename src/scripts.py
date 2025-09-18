import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import plotly.graph_objects as go

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

def visualize_embeddings(query_vector: np.ndarray, doc_vector: np.ndarray, model_id: str = "", query_num: int = 1):
    print("2D Query Vector:", type(query_vector), query_vector)
    print("2D Document Vectors:", type(doc_vector),doc_vector)
    # Create a new figure and axes for the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the retrieved document chunks
    ax.scatter(doc_vector[:, 0], doc_vector[:, 1], c='blue', s=50, label='Retrieved Chunks')

    # Plot the query point
    ax.scatter(query_vector[0], query_vector[1], c='red', s=100, label='Query')

    # Add text labels for each point
    query_coord_text = f"Query ({query_vector[0]:.0f}, {query_vector[1]:.0f})"
    ax.text(query_vector[0] + 50, query_vector[1] + 50, query_coord_text,
            color='black', fontsize=11, ha='left', weight='bold') # Make query label bold

    for i, txt in enumerate(doc_vector):
        ax.text(txt[0] + 2, txt[1] + 2, f'Doc {i+1}', color='black', fontsize=9, ha='left')

    # Set plot titles and labels
    ax.set_title(f't-SNE Visualization of FAISS Search Results ({model_id}, Query {query_num})', fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)

    # Set the axis ticks to 100 for better visual spacing
    ax.xaxis.set_major_locator(plt.MultipleLocator(300))
    ax.yaxis.set_major_locator(plt.MultipleLocator(300))

    # Add a legend
    ax.legend(title='Vectors', loc='best')

    plt.tight_layout()
    plt.show()

    # Create a list of labels for the plot
    # labels = ["Query"] + [f"Doc {i+1}" for i in range(len(doc_vector))]

    # # Visualize the results using Plotly for an interactive plot
    # fig = go.Figure()

    # Add a trace for the query
    # fig.add_trace(go.Scatter(
    #     x=[query_vector[0]],
    #     y=[query_vector[1]],
    #     mode='markers+text',
    #     name='Query',
    #     text=['Query'],
    #     marker=dict(size=10, color='red'),
    #     textposition="top center"
    # ))

    # # Add a trace for the retrieved documents
    # fig.add_trace(go.Scatter(
    #     x=doc_vector[:, 0],
    #     y=doc_vector[:, 1],
    #     mode='markers+text',
    #     name='Retrieved Chunks',
    #     text=[f"Doc {i+1}" for i in range(len(doc_vector))],
    #     marker=dict(size=8, color='blue'),
    #     textposition="bottom center"
    # ))

    # # Update layout for a better visual
    # fig.update_layout(
    #     title=f't-SNE Visualization of FAISS Search Results ({model_id})',
    #     xaxis_title='t-SNE Dimension 1',
    #     yaxis_title='t-SNE Dimension 2',
    #     legend_title='Vectors',
    #     width=700,  # Set the canvas width to 600 pixels
    #     height=400  # Set the canvas height to 400 pixels   # Sets the y-axis tick step
    # )

    # fig.write_image(f"tsne_plot_{model_id.replace('/', '_')}.png")

    # # fig.show()

    # Or to save to a file:
    # Ensure the model ID is safe for a folder name by replacing slashes
    safe_model_id = model_id.replace('/', '_').replace('-', '_')

    # Construct the full path to the desired folder
    output_dir = os.path.join("data", "generated", safe_model_id)

    # Create the nested directories if they don't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full file path including the directory and filename
    filename = f"tsne_plot_query{query_num}.png"
    full_path = os.path.join(output_dir, filename)

    # Save the plot to the specified path
    plt.savefig(full_path)

    print(f"Image saved to: {full_path}")

import pandas as pd
import matplotlib.pyplot as plt

def visualize_embedding_model_performance(average_scores):
    """
    This function takes a dictionary of average scores and creates a bar chart.
    """
    # Convert the dictionary to a pandas DataFrame
    df_scores = pd.DataFrame(list(average_scores.items()), columns=['Model', 'Average Cosine Score'])

    # Sort the DataFrame by score in descending order
    df_scores = df_scores.sort_values(by='Average Cosine Score', ascending=False)

    # Create the bar chart
    plt.figure(figsize=(12, 8))
    plt.bar(df_scores['Model'], df_scores['Average Cosine Score'], color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Average Cosine Score')
    plt.title('Average Cosine Score of Different Sentence Transformer Models')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1) # Cosine scores are between -1 and 1, but for similarity, often 0 to 1
    plt.tight_layout()

    # Save the plot
    # Construct the full path to the desired folder
    output_dir = os.path.join("data", "generated")

    # Create the nested directories if they don't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full file path including the directory and filename
    filename = f"average_cosine_score.png"
    full_path = os.path.join(output_dir, filename)

    # Save the plot to the specified path
    plt.savefig(full_path)
    print("Bar chart saved as average_cosine_scores.png")

# Example usage with your function:
# average_scores = analyse_models()
# plot_average_scores(average_scores)

if __name__ == "__main__":
    print(get_pointers_for_query(1))