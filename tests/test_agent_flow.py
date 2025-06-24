import sys
import os

# Allow Python to find the 'app' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.orchestrator import run_agent

if __name__ == "__main__":
    # Sample user input for testing
    user_input = {
        "monthly_income": 400000,
        "monthly_expenses": 50000,
        "monthly_emis": 110000,
        "savings": 3000000,
        "investments": 1500000,
        "risk_preference": "",  # Leave blank to trigger LLM follow-up
        "goals": "I want to double my savings in 10 years"             # Leave blank to trigger LLM follow-up
    }

    print("🤖 Running InvestiMate Agent...\n")
    try:
        response = run_agent(user_input)
        print("💬 Final Agent Response:\n")
        print(response)
    except Exception as e:
        print(f"❌ Error: {e}")
