# First, install the library
# pip install ragas

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_recall,
    context_precision,
    answer_correctness,
)

# 1. Your data prepared in the Hugging Face Datasets format
#    (This is just a convenient way to structure the data)
data_samples = {
    'question': ['What is the function of the ribosome in a cell?'],
    'contexts': [[
        "The ribosome is a complex molecular machine...",
        "Ribosomes link amino acids together..."
    ]],
    'answer': ['The ribosome is a machine in the cell that builds proteins...'], # Your RAG answer
    'ground_truth': ['According to the documents, the ribosome is the site...']
}
dataset = Dataset.from_dict(data_samples)

# 2. Define the metrics you want to calculate
metrics_to_run = [
    faithfulness,
    answer_relevance,
    answer_correctness,
    context_precision,
    context_recall,
]

# 3. Run the evaluation
#    This will use an LLM (e.g., OpenAI's or a local one) to perform the scoring
result = evaluate(
    dataset=dataset,
    metrics=metrics_to_run,
)

# 4. The result is a clean dictionary or DataFrame
print(result)
# Expected output might look like:
# {'faithfulness': 1.0, 'answer_relevance': 0.95, 'answer_correctness': 0.92, ...}