import requests
import json


# Tool: calculates the total number of changed lines in a Pull Request.
def calculate_total_changes(additions, deletions):
    total = additions + deletions
    return total

# Tool: calculates the average number of changed lines per file.
def calculate_average_changes(total_changes, files_changed):
    average = total_changes / files_changed
    return average


# State: stores information needed while the current agent task is running.
state = {
    "user_request": "A Pull Request has 100 total changes across 5 files. Calculate the average number of changes per file.",
    "selected_tool": None,
    "tool_arguments": None,
    "tool_result": None
}


# Describes the Python tool in a structured format so the LLM can understand it.
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_total_changes",
            "description": "Calculate the total number of changed lines in a Pull Request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "additions": {
                        "type": "integer"
                    },
                    "deletions": {
                        "type": "integer"
                    }
                },
                "required": ["additions", "deletions"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "calculate_average_changes",
            "description": "Calculate the average number of changed lines per file in a Pull Request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "total_changes": {
                        "type": "integer"
                    },
                    "files_changed": {
                        "type": "integer"
                    }
                },
                "required": ["total_changes", "files_changed"]
            }
        }
    }
]

# Routes a tool request to the correct Python function.
def execute_tool(tool_name, arguments):
    if tool_name == "calculate_total_changes":
        return calculate_total_changes(
            arguments["additions"],
            arguments["deletions"]
        )

    if tool_name == "calculate_average_changes":
        return calculate_average_changes(
            arguments["total_changes"],
            arguments["files_changed"]
        )

        raise ValueError(f"Unknown tool: {tool_name}")


# FIRST QWEN CALL
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen2.5-coder:3b",

        # /api/chat receives conversation messages instead of one prompt.
        "messages": [
            {
                "role": "user",
                "content": "A Pull Request has 100 total changes across 5 files. Calculate the average number of changes per file."
            }
        ],

        # Gives Qwen the structured tools it is allowed to request.
        "tools": tools,

        "stream": False
    }
)

# Gets the complete Ollama response as a Python dictionary.
response_data = response.json()

# Gets Qwen's JSON tool request from message.content.
tool_request_text = response_data["message"]["content"]

# Converts the JSON text into a Python dictionary.
tool_request = json.loads(tool_request_text)

# Updates state with Qwen's tool decision.
state["selected_tool"] = tool_request["name"]
state["tool_arguments"] = tool_request["arguments"]

print(tool_request)

# Sends Qwen's selected tool and arguments to the tool dispatcher.
result = execute_tool(
    tool_request["name"],
    tool_request["arguments"]
)

# Updates state with the result returned by the selected tool.
state["tool_result"] = result

print(result)

# SECOND QWEN CALL
# Sends the tool result back to Qwen so it can create the final answer.
final_response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen2.5-coder:3b",
    "messages": [
    {
        "role": "user",
        "content": state["user_request"]
    },
    {
        "role": "assistant",
        "content": f'The {state["selected_tool"]} tool returned: {state["tool_result"]}'
    },
    {
        "role": "user",
        "content": "Using the tool result, give me the final answer in one short sentence."
    }
],
        "stream": False
    }
)

# Gets Qwen's final human-readable answer.
final_answer = final_response.json()["message"]["content"]

print(final_answer)
# Shows the final state after the agent task finishes.
print(state)