# Tool schema for LLM dynamic invocation
tool_schema = [
    {
        "type": "function",
        "function": {
            "name": "analyze_user",
            "description": "Analyze user investment profile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "surplus": {"type": "number"},
                    "risk": {"type": "string"}
                },
                "required": ["surplus", "risk"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_interest_rate",
            "description": "Fetch current interest rate.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_allocation",
            "description": "Format portfolio allocation into readable form.",
            "parameters": {
                "type": "object",
                "properties": {
                    "weights": {
                        "type": "object",
                        "additionalProperties": {"type": "number"}
                    }
                },
                "required": ["weights"]
            }
        }
    }
]
