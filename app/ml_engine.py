# app/ml_engine.py

def analyze_user(user_data):
    """
    Dummy ML scoring function (you'll replace this with ML later)
    """
    score = 0
    if user_data['surplus'] > 50000:
        score += 2
    if user_data['risk'] == 'high':
        score += 2
    elif user_data['risk'] == 'moderate':
        score += 1

    # Dummy classification
    if score >= 3:
        profile = "Aggressive Growth Investor"
        weights = {"Equity": 0.7, "Gold": 0.2, "Bonds": 0.1}
    elif score == 2:
        profile = "Balanced Investor"
        weights = {"Equity": 0.5, "Gold": 0.3, "Bonds": 0.2}
    else:
        profile = "Conservative Saver"
        weights = {"Equity": 0.3, "Gold": 0.2, "Bonds": 0.5}

    return {
        "user_profile": profile,
        "recommended_weights": weights
    }
