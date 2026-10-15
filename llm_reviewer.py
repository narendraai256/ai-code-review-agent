import requests
import json

from rag import retrieve_context


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def review_code(code):
    # Makes sure code was actually provided.
    if not code or not code.strip():
        print("Code to review is empty.")
        return None

    # Retrieves relevant knowledge using RAG.
    context = retrieve_context(code)

    # Stops the review if RAG retrieval failed.
    if context is None:
        print("Could not retrieve RAG context.")
        return None

    payload = {
        "model": "qwen2.5-coder:3b",
        "prompt": f"""Review the following code as a professional code reviewer.

Focus on the actual coding problem, not only the value itself.

Use the relevant coding guidelines below as context when reviewing the code.

Return ONLY valid JSON in this format:
{{
  "issue_type": "security | bug | code_quality",
  "severity": "low | medium | high",
  "problem": "short description of the actual code issue",
  "recommendation": "short recommendation"
}}

For example, if a password is directly written in source code, identify the problem as a hardcoded credential.

Relevant coding guidelines:
{context}

Code to review:
{code}""",
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

    except requests.RequestException as error:
        print("Ollama request failed:", error)
        return None

    except ValueError as error:
        print("Invalid response returned by Ollama:", error)
        return None

    # Makes sure Ollama returned the expected response field.
    if "response" not in result:
        print("Ollama response is missing the response field.")
        return None

    try:
        review_data = json.loads(
            result["response"]
        )

    except (json.JSONDecodeError, TypeError) as error:
        print("Invalid JSON returned by Ollama:", error)
        return None

    # Makes sure the required review fields exist.
    required_fields = [
        "issue_type",
        "severity",
        "problem",
        "recommendation"
    ]

    for field in required_fields:
        if field not in review_data:
            print(
                "LLM review is missing required field:",
                field
            )
            return None

    return review_data