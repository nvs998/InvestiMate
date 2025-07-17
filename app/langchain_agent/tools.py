# tools.py

def analyze_user(surplus, risk):
    score = 0
    if surplus > 50000:
        score += 2
    if risk == 'high':
        score += 2
    elif risk == 'moderate':
        score += 1

    if score >= 3:
        profile = "Aggressive Growth Investor"
        weights = {"Equity": 0.7, "Gold": 0.2, "Bonds": 0.1}
    elif score == 2:
        profile = "Balanced Investor"
        weights = {"Equity": 0.5, "Gold": 0.3, "Bonds": 0.2}
    else:
        profile = "Conservative Saver"
        weights = {"Equity": 0.3, "Gold": 0.2, "Bonds": 0.5}

    return {"profile": profile, "weights": weights}

def fetch_interest_rate():
    return {"interest_rate": 6.5}

def format_allocation(weights):
    return {"formatted": ", ".join([f"{k}: {v*100:.0f}%" for k, v in weights.items()])}
