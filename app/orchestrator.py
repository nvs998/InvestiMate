# app/orchestrator.py

from ml_engine import analyze_user
from logic import fetch_interest_rate, format_allocation
from llm_explainer import explain_strategy

def run_ai_agent(user_data):
    print("✅ Running ML Analysis...")
    analysis = analyze_user(user_data)

    print("✅ Fetching External Logic Inputs...")
    interest_rate = fetch_interest_rate()

    print("✅ Sending to LLM for Explanation...")
    explanation = explain_strategy(user_data, analysis, interest_rate)

    # Combine results
    return {
        "profile": analysis["user_profile"],
        "weights": format_allocation(analysis["recommended_weights"]),
        "interest_rate": f"{interest_rate}%",
        "explanation": explanation
    }

if __name__ == "__main__":
    user_input = {
        "surplus": 60000,
        "risk": "moderate",
        "goal": "retirement"
    }

    result = run_ai_agent(user_input)
    print("\n📊 Final Output:")
    print("User Profile:", result["profile"])
    print("Suggested Allocation:", result["weights"])
    print("Interest Rate:", result["interest_rate"])
    print("\n🤖 Explanation from LLM:\n")
    print(result["explanation"])
