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
