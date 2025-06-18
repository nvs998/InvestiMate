# app/llm_explainer.py

import requests

def explain_strategy(user_data, analysis, interest_rate):
    prompt = f"""
    A user has ₹{user_data['surplus']} extra, risk tolerance: {user_data['risk']}, goal: {user_data['goal']}.
    Our ML model classified them as "{analysis['user_profile']}".
    Suggested allocation: {analysis['recommended_weights']}.
    Current interest rate: {interest_rate}%.

    Write a personalized investment recommendation with reasoning in clear language.
    """

    response = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )

    return response.json()["response"]
