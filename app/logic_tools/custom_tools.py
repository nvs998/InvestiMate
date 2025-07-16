def fetch_interest_rate():
    # Placeholder for external API fetch
    return 6.5  # %

def format_allocation(weights):
    return ", ".join([f"{k}: {v*100:.0f}%" for k, v in weights.items()])
