import requests
import json


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def review_code(code):
    payload = {
        "model": "qwen2.5-coder:3b",
        "prompt": f"""Review the following code as a professional code reviewer.

Focus on the actual coding problem, not only the value itself.

Return ONLY valid JSON in this format:
{{
  "issue_type": "security | bug | code_quality",
  "severity": "low | medium | high",
  "problem": "short description of the actual code issue",
  "recommendation": "short recommendation"
}}

For example, if a password is directly written in source code, identify the problem as a hardcoded credential.

Code:
{code}""",
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()

    except requests.RequestException as error:
        print("Ollama request failed:", error)
        return None

    result = response.json()

    return result["response"]


code_to_review = 'password = "admin123"'

review = review_code(code_to_review)

print("RAW REVIEW:")
print(review)

try:  # try to convert the LLM JSON text into a Python dictionary
    review_data = json.loads(review)

except json.JSONDecodeError as error:  # catches invalid JSON returned by the LLM
    print("Invalid JSON returned by Ollama:", error)
    review_data = None


if review_data is not None:  # only access the fields when JSON parsing succeeded
    print(review_data)
    print(review_data["issue_type"])
    print(review_data["severity"])
    print(review_data["problem"])
    print(review_data["recommendation"])
