def analyze_surplus(user_input):
    income = user_input.get("monthly_income", 0)
    expenses = user_input.get("monthly_expenses", 0)
    emis = user_input.get("monthly_emis", 0)
    
    surplus = income - expenses - emis
    surplus_status = "healthy" if surplus >= 20000 else "tight"
    print(f"Surplus Analysis: Income={income}, Expenses={expenses}, EMIs={emis}, Surplus={surplus}, Status={surplus_status}")
    return {
        "monthly_surplus": surplus,
        "status": surplus_status,
        "comment": f"Your monthly surplus is ₹{surplus}, which is considered {surplus_status}."
    }

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
