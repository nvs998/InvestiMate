# def recommend_investments(surplus, risk_tolerance, goal="wealth creation"):
#     """
#     Given surplus amount, risk profile, and financial goal,
#     return a list of investment recommendations.
#     """
#     risk = risk_tolerance.lower()
#     goal = goal.capitalize()
#     suggestions = []
#     notes = ""

#     if surplus < 5000:
#         suggestions = ["Liquid Mutual Funds", "High-Interest Savings Account"]
#         notes = "Small surplus — prioritize liquidity and emergency fund."
#     else:
#         if risk == "low":
#             suggestions = [
#                 "Public Provident Fund (PPF)",
#                 "Short-Term Debt Mutual Funds",
#                 "Bank Fixed Deposits"
#             ]
#         elif risk == "moderate":
#             suggestions = [
#                 "Balanced Mutual Funds",
#                 "Gold ETFs",
#                 "REITs for Passive Income"
#             ]
#         elif risk == "high":
#             suggestions = [
#                 "Index Mutual Funds (e.g. Nifty 50, S&P 500)",
#                 "Bluechip Stocks",
#                 "Thematic ETFs (Tech, EV, Green)"
#             ]
#         else:
#             suggestions = ["Unknown risk level"]
#             notes = "Please choose: low, moderate, or high"

#     return {
#         "surplus": surplus,
#         "goal": goal,
#         "risk": risk,
#         "suggestions": suggestions,
#         "notes": notes or f"Based on ₹{surplus} with {risk} risk profile and goal: {goal}."
#     }

# app/logic.py

def fetch_interest_rate():
    # Placeholder for external API fetch
    return 6.5  # %

def format_allocation(weights):
    return ", ".join([f"{k}: {v*100:.0f}%" for k, v in weights.items()])
