class AgentState:
    def __init__(self, user_input):
        self.user_input = user_input
        self.known_keys = set(k for k, v in user_input.items() if v)
        self.required_keys = {"goals", "risk_preference", "monthly_income", "monthly_expenses"}

    def is_complete(self, llm_response):
        # Very basic check for now — if required keys are present in input
        return self.required_keys.issubset(self.known_keys)

    def get_follow_up_question(self, llm_response):
        # Ask for any missing fields
        for key in self.required_keys:
            if key not in self.known_keys:
                return f"Can you tell me your {key.replace('_', ' ')}?"
        return None
