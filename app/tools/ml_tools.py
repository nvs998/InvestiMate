from langchain.tools import tool
from pydantic import BaseModel, Field
import json 

class AnalyzeUserInput(BaseModel):
    surplus: int = Field(..., description="User's surplus income after all expenses in INR")
    risk_profile: str = Field(..., description="Risk appetite: low, moderate, high")

@tool(args_schema=AnalyzeUserInput)
def analyze_user_tool(surplus: int, risk_profile: str) -> dict:
    """
    Analyze user's financial situation and suggest asset allocation.
    """

    if risk_profile.lower() == "high":
        allocation = {"equity": 70, "debt": 20, "gold": 10}
    elif risk_profile.lower() == "moderate":
        allocation = {"equity": 50, "debt": 40, "gold": 10}
    else:
        allocation = {"equity": 30, "debt": 60, "gold": 10}

    result_dict = {"surplus": surplus, "allocation": allocation}
    print(f"Returning this from custom AUT{result_dict},{type(json.dumps(result_dict, indent=2))}")
    return result_dict
    # return json.dumps(result_dict, indent=2)

class FormatAllocationInput(BaseModel):
    # This Pydantic model defines the *expected input* for the format_allocation_tool
    # It must match the structure of the output from analyze_user_tool
    surplus: int = Field(..., description="User's monthly surplus income after all expenses in INR, usually from analyze_user_tool output")
    allocation: dict = Field(..., description="Dictionary containing equity, debt, and gold allocation percentages, usually from analyze_user_tool output")

# --- NEW: format_allocation_tool ---
@tool(args_schema=FormatAllocationInput) # <-- Use the new args_schema
def format_allocation_tool(surplus: int, allocation: dict) -> str: # <-- Function arguments must match args_schema fields
    """
    Convert allocation output into a natural language explanation.
    Takes the full output dictionary from analyze_user_tool.
    Example input (as a dict, not a string):
    {
        "surplus": 200000,
        "allocation": {"equity": 50, "debt": 40, "gold": 10}
    }
    """
    print(f"Returning this from custom FAT{allocation},{type(allocation)}")
    
    # Access arguments directly as passed by Pydantic
    equity_pct = allocation.get('equity', 0)
    debt_pct = allocation.get('debt', 0)
    gold_pct = allocation.get('gold', 0)

    # Use f-strings for formatting and comma for thousands separator
    return (
        f"Based on your monthly surplus of ₹{surplus:,}, "
        f"we recommend allocating {equity_pct}% to Equity, "
        f"{debt_pct}% to Debt, and {gold_pct}% to Gold investments."
    )


# @tool
# def apply_rules_tool(input: dict) -> dict:
#     """
#     Apply business logic or compliance rules to allocation.
#     Example input:
#     {
#         "surplus": 200000,
#         "allocation": {"equity": 50, "debt": 40, "gold": 10}
#     }
#     """
#     print("========input",input,type(input))
#     # Example rule: If surplus < 1L, shift more to debt
#     if input["surplus"] < 100000:
#         input["allocation"]["equity"] -= 10
#         input["allocation"]["debt"] += 10

#     return input

# @tool
# def format_allocation_tool(input: dict) -> str:
#     """
#     Convert allocation output into a natural language explanation.
#     Example input:
#     {
#         "surplus": 200000,
#         "allocation": {"equity": 50, "debt": 40, "gold": 10}
#     }
#     """
#     print("========input",input,type(input))
#     surplus = input["surplus"]
#     a = input["allocation"]
#     return (
#         f"Based on your monthly surplus of ₹{surplus:,}, "
#         f"we recommend allocating {a['equity']}% to Equity, "
#         f"{a['debt']}% to Debt, and {a['gold']}% to Gold investments."
#     )