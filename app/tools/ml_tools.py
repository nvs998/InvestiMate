from langchain.tools import tool

from langchain.tools import tool
from pydantic import BaseModel, Field

class AnalyzeUserInput(BaseModel):
    monthly_income: int = Field(..., description="User's monthly income in INR")
    monthly_expenses: int = Field(..., description="User's monthly expenses in INR")
    risk_profile: str = Field(..., description="Risk appetite: low, moderate, high")

@tool(args_schema=AnalyzeUserInput)
def analyze_user_tool(monthly_income: int, monthly_expenses: int, risk_profile: str) -> dict:
    """
    Analyze user's financial situation and suggest asset allocation.
    """
    surplus = monthly_income - monthly_expenses

    if risk_profile.lower() == "high":
        allocation = {"equity": 70, "debt": 20, "gold": 10}
    elif risk_profile.lower() == "moderate":
        allocation = {"equity": 50, "debt": 40, "gold": 10}
    else:
        allocation = {"equity": 30, "debt": 60, "gold": 10}

    return {"surplus": surplus, "allocation": allocation}


# @tool
# def analyze_user_tool(input: dict) -> dict:
#     """Analyze user's income, expenses, and risk profile to suggest allocation.
#     Example input:
#     {
#         "monthly_income": 400000,
#         "monthly_expenses": 50000,
#         "risk_profile": "moderate"
#     }
#     """
#     print("========input",input,type(input))
#     # Dummy logic for now – replace with your actual ML analysis
#     surplus = input["monthly_income"] - input["monthly_expenses"]
#     if input["risk_profile"] == "high":
#         allocation = {"equity": 70, "debt": 20, "gold": 10}
#     elif input["risk_profile"] == "moderate":
#         allocation = {"equity": 50, "debt": 40, "gold": 10}
#     else:
#         allocation = {"equity": 30, "debt": 60, "gold": 10}

#     return {"surplus": surplus, "allocation": allocation}

@tool
def apply_rules_tool(input: dict) -> dict:
    """
    Apply business logic or compliance rules to allocation.
    Example input:
    {
        "surplus": 200000,
        "allocation": {"equity": 50, "debt": 40, "gold": 10}
    }
    """
    print("========input",input,type(input))
    # Example rule: If surplus < 1L, shift more to debt
    if input["surplus"] < 100000:
        input["allocation"]["equity"] -= 10
        input["allocation"]["debt"] += 10

    return input

@tool
def format_allocation_tool(input: dict) -> str:
    """
    Convert allocation output into a natural language explanation.
    Example input:
    {
        "surplus": 200000,
        "allocation": {"equity": 50, "debt": 40, "gold": 10}
    }
    """
    print("========input",input,type(input))
    surplus = input["surplus"]
    a = input["allocation"]
    return (
        f"Based on your monthly surplus of ₹{surplus:,}, "
        f"we recommend allocating {a['equity']}% to Equity, "
        f"{a['debt']}% to Debt, and {a['gold']}% to Gold investments."
    )