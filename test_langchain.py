import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from rag import retrieve_context


# =========================================================
# 1. LANGCHAIN TOOL
# =========================================================

@tool
def calculate_total_changes(additions: int, deletions: int) -> int:
    """Calculate the total number of changed lines in a Pull Request."""
    return additions + deletions


# =========================================================
# 2. LANGCHAIN MODEL
# =========================================================

model = ChatOllama(
    model="qwen2.5-coder:3b"
)


# =========================================================
# 3. PROMPT + CHAIN
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
    Review the following code:

    {code}

    Risk level: {risk_level}

    Explain the most important problem in one short sentence.
    """
)

chain = prompt | model

chain_response = chain.invoke(
    {
        "code": 'password = "admin123"',
        "risk_level": "high"
    }
)

print("\n=== BASIC LANGCHAIN CHAIN ===")
print(chain_response.content)


# =========================================================
# 4. LANGCHAIN TOOL
# =========================================================

model_with_tools = model.bind_tools(
    [calculate_total_changes]
)

tool_response = model_with_tools.invoke(
    "A Pull Request has 20 additions and 5 deletions. "
    "Calculate the total changes."
)

print("\n=== TOOL DECISION ===")
print(tool_response.content)


tool_result = calculate_total_changes.invoke(
    {
        "additions": 20,
        "deletions": 5
    }
)

print("\n=== DIRECT TOOL EXECUTION ===")
print(tool_result)


# =========================================================
# 5. LANGCHAIN + RAG
# =========================================================

code_to_review = 'api_key = "SECRET123"'

rag_context = retrieve_context(code_to_review)

rag_prompt = ChatPromptTemplate.from_template(
    """
    Review the following code:

    {code}

    Use these coding guidelines as context:

    {context}

    Explain the most important problem and recommendation
    in two short sentences.
    """
)

rag_chain = rag_prompt | model

rag_response = rag_chain.invoke(
    {
        "code": code_to_review,
        "context": rag_context
    }
)

print("\n=== RETRIEVED RAG CONTEXT ===")
print(rag_context)

print("\n=== LANGCHAIN RAG RESPONSE ===")
print(rag_response.content)


# =========================================================
# 6. SIMPLE AGENT LOOP
# =========================================================

agent_request = (
    "A Pull Request has 30 additions and 10 deletions. "
    "Calculate the total number of changes."
)

print("\n=== AGENT REQUEST ===")
print(agent_request)


# DECIDE:
# Qwen decides which available tool should handle the request.
agent_decision = model_with_tools.invoke(agent_request)

print("\n=== AGENT DECISION ===")
print(agent_decision.content)


# PARSE:
# Qwen sometimes wraps JSON inside Markdown ```json code fences.
clean_agent_decision = agent_decision.content.strip()

if clean_agent_decision.startswith("```json"):
    clean_agent_decision = clean_agent_decision[7:]

if clean_agent_decision.endswith("```"):
    clean_agent_decision = clean_agent_decision[:-3]

clean_agent_decision = clean_agent_decision.strip()

agent_tool_request = json.loads(clean_agent_decision)


# NORMALIZE:
# Qwen may return either:
#
# "additions": 30
#
# OR:
#
# "additions": {"type": "integer", "value": 30}

raw_arguments = agent_tool_request["arguments"]

additions = raw_arguments["additions"]
deletions = raw_arguments["deletions"]

if isinstance(additions, dict):
    additions = additions["value"]

if isinstance(deletions, dict):
    deletions = deletions["value"]

agent_arguments = {
    "additions": additions,
    "deletions": deletions
}

print("\n=== NORMALIZED ARGUMENTS ===")
print(agent_arguments)


# ACT:
# Executes the tool selected by Qwen.
agent_tool_result = calculate_total_changes.invoke(
    agent_arguments
)

print("\n=== AGENT TOOL RESULT ===")
print(agent_tool_result)


# OBSERVE + RESPOND:
# Gives the tool result back to Qwen for the final response.
agent_final_response = model.invoke(
    f"""
    Original request:
    {agent_request}

    Tool used:
    {agent_tool_request["name"]}

    Tool result:
    {agent_tool_result}

    Using the tool result, give the final answer in one short sentence.
    """
)

print("\n=== AGENT FINAL RESPONSE ===")
print(agent_final_response.content)