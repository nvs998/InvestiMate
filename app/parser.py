import json
from typing import Optional, Dict

def get_json_before_first_observation(input_dict: Dict) -> Optional[Dict]:
    """
    Finds the first occurrence of 'Observation:' in the 'content' field of the input dictionary,
    and extracts the JSON blob (between '$JSON_BLOB' and the 'Observation:')
    that immediately precedes it.

    Args:
        input_dict (dict): A dictionary expected to have a 'content' key
                           whose value is a string containing the LLM's output.

    Returns:
        Optional[dict]: The parsed JSON dictionary if found, otherwise None.
    """
    content = input_dict.get('content')
    if not content:
        print("Error: Input dictionary has no 'content' key or content is empty.")
        return None

    # Step 1: Find the first occurrence of 'Observation:'
    first_observation_index = content.find("Observation:")
    if first_observation_index == -1:
        print("No 'Observation:' keyword found in the content.")
        return None

    # Step 2: Consider only the part of the string *before* the first 'Observation:'
    segment_before_obs = content[:first_observation_index]

    # Step 3: Find the LAST occurrence of '$JSON_BLOB' in this preceding segment
    # This is crucial because there might be multiple Thought/Action/Observation cycles
    json_blob_marker_index = segment_before_obs.rfind("$JSON_BLOB")
    if json_blob_marker_index == -1:
        print("No '$JSON_BLOB' marker found before the first 'Observation:'.")
        return None

    # Step 4: Extract the raw text segment that *should* contain the JSON
    # This segment starts right after '$JSON_BLOB' and goes to the end of segment_before_obs
    raw_json_text_segment = segment_before_obs[json_blob_marker_index + len("$JSON_BLOB"):]

    # Step 5: Isolate the actual JSON string within the raw_json_text_segment
    # We look for the first '{' and the last '}' to define the JSON boundaries.
    start_brace_idx = raw_json_text_segment.find('{')
    if start_brace_idx == -1:
        print(f"No opening brace '{{' found in the segment after '$JSON_BLOB'. Segment: '{raw_json_text_segment}'")
        return None

    end_brace_idx = raw_json_text_segment.rfind('}')
    # Ensure end_brace_idx is valid and comes after start_brace_idx
    if end_brace_idx == -1 or end_brace_idx < start_brace_idx:
        print(f"No closing brace '}}' found or it's malformed after '$JSON_BLOB'. Segment: '{raw_json_text_segment}'")
        return None

    # Extract the string that is most likely the JSON
    json_str = raw_json_text_segment[start_brace_idx : end_brace_idx + 1]

    # Step 6: Attempt to parse the extracted JSON string
    try:
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON found: {e}")
        print(f"Attempted to parse string: '{json_str}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        return None

import json
from typing import Optional, Dict

def extract_json_from_llm_code_block(input_dict: Dict) -> Optional[Dict]:
    """
    Extracts a JSON object from an LLM output string where the JSON is enclosed
    within a Markdown code block (```json...```).

    Args:
        llm_output_string (str): The full string output from the LLM.

    Returns:
        Optional[Dict]: The parsed JSON as a Python dictionary if found, otherwise None.
    """
    if not isinstance(input_dict, dict):
        print("Error: Input must be a string.")
        return None

    llm_output_string = input_dict.get('content')

    if llm_output_string is None:
        print("Warning: Input dictionary has no 'content' key or content is None.")
        return None
    if not isinstance(llm_output_string, str):
        print("Error: The 'content' field in the input dictionary must be a string.")
        return None

    # Define the markers for the JSON code block
    start_marker = "```json\n"
    end_marker = "\n```"

    # Find the starting index of the JSON block
    start_index = llm_output_string.find(start_marker)
    if start_index == -1:
        print("Warning: No '```json' starting marker found.")
        return None

    # Calculate the actual start of the JSON content
    json_content_start = start_index + len(start_marker)

    # Find the ending index of the JSON block, searching from after the start marker
    end_index = llm_output_string.find(end_marker, json_content_start)
    if end_index == -1:
        print("Warning: No closing '```' marker found after '```json'.")
        return None

    # Extract the raw JSON string
    json_str = llm_output_string[json_content_start:end_index].strip() # .strip() removes any extra whitespace/newlines

    # Attempt to parse the JSON string
    try:
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Attempted to parse this string:\n---\n{json_str}\n---")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during JSON extraction: {e}")
        return None

# --- Example Usage with your full input dictionary ---
full_llm_output_dict = {
  "role": "assistant",
  "content": "Thought: To determine the investment strategy, I need to analyze the user's surplus and risk tolerance.\n\n${\n  \"action\": \"analyze_user\",\n  \"action_input\": {\"surplus\": 100000, \"risk\": \"medium\"}\n}\n\nObservation:\n{\n  \"surplus\": 100000,\n  \"risk_tolerance\": \"medium\"\n}\n\nThought: Now that I have the user's surplus and risk tolerance, I need to get the current interest rate to determine the investment options.\n\n${\n  \"action\": \"fetch_interest_rate\",\n  \"action_input\": {}\n}\n\nObservation:\n{\n  \"interest_rate\": 4.5\n}\n\nThought: With the interest rate in mind, I can now determine the allocation of the investment.\n\nAction:\n\n${\n  \"action\": \"analyze_user\",\n  \"action_input\": {\"surplus\": 100000, \"risk\": \"medium\"}\n}\n\nObservation:\n{\n  \"allocation\": {\n    \"low_risk\": 0.4,\n    \"medium_risk\": 0.3,\n    \"high_risk\": 0.3\n  }\n}\n\nThought: Now that I have the allocation, I can format it into a readable string.\n\n${\n  \"action\": \"format_allocation"
}

# # Now, call the function directly with your dictionary:
# extracted_action = extract_json_from_llm_code_block(full_llm_output_dict)

# if extracted_action:
#     print("Successfully extracted the first action JSON:")
#     print(json.dumps(extracted_action, indent=2))
# else:
#     print("Failed to extract JSON.")

def remove_after_first_observation(input_object):
    """
    Removes everything from the 'content' string of the input object
    after the first occurrence of 'Observation:'.
    If 'Observation:' is not found in the content, the content remains unchanged.
    Returns the modified object, or None if input_object is not a dict or lacks 'content'.

    Args:
        input_object (Dict[str, Any]): The input dictionary object, expected to
                                       have a 'content' key whose value is a string.

    Returns:
        Optional[Dict[str, Any]]: The original object with its 'content' string modified,
                                  or None if the input is invalid.
    """
    if not isinstance(input_object, dict):
        print("Error: Input must be a dictionary.")
        return None

    # Get the content string from the object
    text_content = input_object.get('content')

    if text_content is None:
        # If 'content' key is missing or None, return the object as is or handle as error
        # For this function's purpose, if no content, nothing to remove, so return original.
        print("Warning: Input object has no 'content' key or its value is None. Returning original object.")
        return input_object
    
    if not isinstance(text_content, str):
        print("Error: The 'content' field must be a string. Returning original object.")
        return input_object


    keyword = "Observation:"
    first_occurrence_index = text_content.find(keyword)

    if first_occurrence_index != -1:
        # If 'Observation:' is found, slice the string to include it and everything before it
        modified_content = text_content[:first_occurrence_index + len(keyword)]
        # Update the 'content' field in the original object
        input_object['content'] = modified_content
    # else: If 'Observation:' is not found, the content remains unchanged, which is desired.

    return input_object

# example_object_1 = {
#         "role": "assistant",
#         "timestamp": "2025-07-17",
#         "content": (
#             "Thought: Some initial thoughts.\n"
#             "Action: {\"tool\": \"do_something\"}\n"
#             "Observation: This is the observation result. All text after this should be removed."
#             "\nMore text here that should disappear.\nAnother line to disappear."
#         )
#     }

# print("--- Example 1 (Content will be truncated) ---")
# print("Original Content:\n", example_object_1['content'])
# modified_object_1 = remove_after_first_observation(example_object_1)
# if modified_object_1:
#     print("\nModified Content:\n", modified_object_1['content'])
# else:
#     print("Function returned None.")

# # Example 2: Object with content that does NOT have 'Observation:'
# example_object_2 = {
#     "role": "user",
#     "id": "123",
#     "content": "Hello, how are you? This is a user message with no observation."
# }
# print("\n--- Example 2 (Content should remain unchanged) ---")
# print("Original Content:\n", example_object_2['content'])
# modified_object_2 = remove_after_first_observation(example_object_2)
# if modified_object_2:
#     print("\nModified Content:\n", modified_object_2['content'])
# else:
#     print("Function returned None.")

# # Example 3: Object with no 'content' key
# example_object_3 = {
#     "role": "system",
#     "instructions": "Be helpful."
# }
# print("\n--- Example 3 (No 'content' key) ---")
# print("Original Object:\n", example_object_3)
# modified_object_3 = remove_after_first_observation(example_object_3)
# if modified_object_3:
#     print("\nModified Object (should be original):\n", modified_object_3)
# else:
#     print("Function returned None.")

# # Example 4: Input is not a dictionary
# example_object_4 = "This is just a string, not a dict"
# print("\n--- Example 4 (Input is not a dict) ---")
# modified_object_4 = remove_after_first_observation(example_object_4)
# if modified_object_4 is None:
#     print("Correctly handled: Input was not a dictionary.")
# else:
#     print("Function returned an unexpected value:", modified_object_4)

import json
from typing import Optional, Dict

def extract_action_json(input_dict: Dict) -> Optional[Dict]:
    """
    Extracts the JSON object that directly follows 'Action:' and precedes the
    first 'Observation:' keyword within the 'content' field of the input dictionary.

    Args:
        input_dict (Dict): The input dictionary, expected to have a 'content' key
                           whose value is a string containing the LLM's output.

    Returns:
        Optional[Dict]: The parsed JSON as a Python dictionary if found, otherwise None.
    """
    if not isinstance(input_dict, dict):
        print("Error: Input to extract_action_json must be a dictionary.")
        return None

    llm_output_content = input_dict.get('content')

    if llm_output_content is None or not isinstance(llm_output_content, str):
        print("Warning: Input dictionary has no 'content' key, or content is not a string.")
        return None

    action_keyword = "Action:"
    observation_keyword = "Observation:"

    # Step 1: Find the start of the "Action:" line
    action_start_index = llm_output_content.find(action_keyword)
    if action_start_index == -1:
        # print("Warning: No 'Action:' keyword found.") # Uncomment for debugging warnings
        return None

    # Calculate the start of the potential JSON string (immediately after "Action:")
    json_potential_start = action_start_index + len(action_keyword)

    # Step 2: Find the start of the NEXT "Observation:"
    # We search from the position right after "Action:" to get the correct "Observation:"
    observation_start_index = llm_output_content.find(observation_keyword, json_potential_start)
    if observation_start_index == -1:
        # print("Warning: No 'Observation:' keyword found after 'Action:'.") # Uncomment for debugging warnings
        return None

    # Step 3: Extract the raw segment between "Action:" and "Observation:"
    raw_json_segment = llm_output_content[json_potential_start:observation_start_index]

    # Step 4: Isolate the actual JSON string (first '{' to last '}') within the segment
    # This handles any leading/trailing whitespace or newlines around the JSON
    json_start_brace_idx = raw_json_segment.find('{')
    if json_start_brace_idx == -1:
        print(f"Error: No opening brace '{{' found in the segment after 'Action:'. Segment: '{raw_json_segment.strip()}'")
        return None

    json_end_brace_idx = raw_json_segment.rfind('}')
    # Ensure the closing brace is found and is after the opening brace
    if json_end_brace_idx == -1 or json_end_brace_idx < json_start_brace_idx:
        print(f"Error: No closing brace '}}' found or malformed JSON after 'Action:'. Segment: '{raw_json_segment.strip()}'")
        return None

    # Extract the precise JSON string
    json_str = raw_json_segment[json_start_brace_idx : json_end_brace_idx + 1]

    # Step 5: Attempt to parse the JSON string
    try:
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Attempted to parse this string:\n---\n{json_str}\n---")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        return None

# --- Testing Script ---

# print("--- Test Case 1: Valid LLM Response with First Action JSON ---")
# llm_response_valid = {
#   "role": "assistant",
#   "content": "Thought: I should first get the current interest rate to determine the optimal investment strategy.\nAction: {\"action\": \"fetch_interest_rate\", \"action_input\": {}} \n\nObservation: The current interest rate is 5%.\n\nThought: I should analyze the user's surplus and risk tolerance to determine the optimal asset allocation.\nAction: {\"action\": \"analyze_user\", \"action_input\": {\"surplus\": 100000, \"risk\": \"medium\"}} \n\nObservation: The recommended asset allocation is 60% stocks, 20% bonds, and 20% liquid assets.\n\nThought: I should format the asset allocation into a readable string.\nAction: {\"action\": \"format_allocation\", \"action_input\": {\"weights\": {\"stocks\": 0.6, \"bonds\": 0.2, \"liquid_assets\": 0.2}}} \n\nObservation: The asset allocation is 60% stocks, 20% bonds, and 20% liquid assets.\n\nThought: I should now know the final answer.\nFinal Answer: The optimal investment strategy is to invest \u20b960,000 in stocks, \u20b920,000 in bonds, and \u20b920,000 in liquid assets."
# }

# extracted_json_valid = extract_action_json(llm_response_valid)
# if extracted_json_valid:
#     print("Extracted JSON (should be 'fetch_interest_rate'):")
#     print(json.dumps(extracted_json_valid, indent=2))
# else:
#     print("Could not extract JSON.")

# print("\n--- Test Case 2: No 'Action:' keyword ---")
# llm_response_no_action = {
#     "role": "assistant",
#     "content": "Thought: Just a thought, no action taken.\nObservation: Nothing happened."
# }
# extracted_json_no_action = extract_action_json(llm_response_no_action)
# if extracted_json_no_action:
#     print("Extracted JSON (should be None):", extracted_json_no_action)
# else:
#     print("Correctly handled: No 'Action:' keyword found.")

# print("\n--- Test Case 3: 'Action:' but no 'Observation:' after it ---")
# llm_response_no_obs_after_action = {
#     "role": "assistant",
#     "content": "Thought: Trying an action.\nAction: {\"action\": \"test_action\"} \nFinal Answer: Final thought."
# }
# extracted_json_no_obs = extract_action_json(llm_response_no_obs_after_action)
# if extracted_json_no_obs:
#     print("Extracted JSON (should be None):", extracted_json_no_obs)
# else:
#     print("Correctly handled: No 'Observation:' after 'Action:'.")

# print("\n--- Test Case 4: Malformed JSON after 'Action:' ---")
# llm_response_malformed_json = {
#     "role": "assistant",
#     "content": "Thought: This JSON is broken.\nAction: {\"action\": \"broken\", \"input\": \n\nObservation: Parse error."
# }
# extracted_json_malformed = extract_action_json(llm_response_malformed_json)
# if extracted_json_malformed:
#     print("Extracted JSON (should be None):", extracted_json_malformed)
# else:
#     print("Correctly handled: Malformed JSON.")

# print("\n--- Test Case 5: Empty 'content' string ---")
# llm_response_empty_content = {
#     "role": "assistant",
#     "content": ""
# }
# extracted_json_empty = extract_action_json(llm_response_empty_content)
# if extracted_json_empty:
#     print("Extracted JSON (should be None):", extracted_json_empty)
# else:
#     print("Correctly handled: Empty content.")