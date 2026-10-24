from tools import calculate_total_changes


# Simulates a structured tool request.
tool_request = {
    "name": "calculate_total_changes",
    "arguments": {
        "additions": 20,
        "deletions": 5
    }
}


# Reads the requested tool name.
tool_name = tool_request["name"]

# Reads the deletions argument from the tool request.
deletions = tool_request["arguments"]["deletions"]
additions = tool_request["arguments"]["additions"]

print(tool_name)
print(additions)
print(deletions)
# Checks whether this is the requested tool.
if tool_name == "calculate_total_changes":

    # Executes the requested Python function using the extracted arguments.
    result = calculate_total_changes(additions, deletions)

    print(result)
    assert result == 25 
# Describes the tool that will be available to the LLM.
tool_definition = {
    "name": "calculate_total_changes",
    "description": "Calculate the total number of changed lines in a Pull Request.",
    "parameters": {
        "additions": "integer",
        "deletions": "integer"
    }
}
print(tool_definition)
