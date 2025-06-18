from logic import recommend_investments

if __name__ == "__main__":
    surplus = float(input("How much extra money do you want to invest? ₹"))
    risk = input("Risk tolerance (low / moderate / high): ").lower()
    goal = input("Goal (retirement, house, etc) [optional]: ") or "wealth creation"

    result = recommend_investments(surplus, risk, goal)

    print("\n📊 Investment Recommendations:")
    for suggestion in result["suggestions"]:
        print(f" - {suggestion}")

    print("\nℹ️ Notes:", result["notes"])
