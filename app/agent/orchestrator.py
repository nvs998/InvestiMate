from app.agent.prompt_templates import SYSTEM_PROMPT, USER_PROMPT
from app.llm_tools.llm_explainer import call_llm
from app.agent.state_manager import AgentState
from app.ml_tools.ml_engine import analyze_surplus

def run_agent(user_input: dict):
    state = AgentState(user_input)
    conversation_history = []

    # 1. Use initial system prompt to define behavior
    conversation_history.append({"role": "system", "content": SYSTEM_PROMPT})

    # 2. Add user-provided inputs to context
    conversation_history.append({"role": "user", "content": USER_PROMPT.format(**user_input)})

    # 3. If all required inputs are present, analyze surplus and inject result
    # if not(state.is_complete(None)):
    #     print("All required inputs are present, proceeding with surplus analysis...")
    #     surplus_analysis = analyze_surplus(user_input)
    #     surplus_msg = f"Surplus Analysis Result: {surplus_analysis['comment']}"
    #     conversation_history.append({"role": "user", "content": surplus_msg})

    # 4. Main loop: LLM thinks, decides what else to ask or recommend
    for _ in range(3):  # limit to 3 rounds for now
        response = call_llm(conversation_history)
        conversation_history.append({"role": "assistant", "content": response})
        if "CALL analyze_surplus" in response:
            result = analyze_surplus(user_input)
            conversation_history.append({
                "role": "user",
                "content": f"RESULT analyze_surplus: {result['comment']}"
            })
            continue  # LLM continues reasoning with that result

        print(f"conversation_history\n: {conversation_history}")
        if state.is_complete(response):
            break  # all needed info gathered

        follow_up = state.get_follow_up_question(response)
        if follow_up:
            conversation_history.append({"role": "user", "content": follow_up})

    return response
