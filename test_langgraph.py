
"""
AI CODE REVIEW AGENT
PHASE 5 - COMPLETE LANGGRAPH WORKFLOW

Features:
- MCP calculations
- RAG retrieval
- ML risk prediction
- Qwen code review
- Low / Medium / High risk routing
- AST source-code security checks
- AI recommendation validation
- Error handling
- Final review aggregation

Uses existing project files:
    main.py
    rag.py
    mcp_tools.py
    security_validator.py
"""

import asyncio
import logging
import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

# Imports the shared ML risk predictor without depending on FastAPI.
from risk_predictor import predict_risk_with_ml
from rag import retrieve_context
from mcp_tools import calculate_changes_via_mcp

from security_validator import (
    analyze_source_security,
    contains_insecure_password_fallback,
    extract_python_code_blocks,
    contains_password_print,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)

model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b"),
    base_url=os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434"
    ),
    temperature=0,
)


# ============================================================
# 2. LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict):

    user_request: str
    code: str

    additions: int
    deletions: int
    files_changed: int

    total_changes: int | None
    average_changes_per_file: float | None
    addition_ratio: float | None
    deletion_ratio: float | None

    risk_level: str | None
    risk_score: float | None

    rag_context: str | None

    ast_findings: list[str]

    security_review: str | None
    quality_review: str | None
    performance_review: str | None
    llm_review: str | None

    final_response: str | None


# ============================================================
# 3. INPUT VALIDATION
# ============================================================

def validate_input(state: AgentState):

    code = state["code"]

    if not isinstance(code, str) or not code.strip():
        raise ValueError("Source code cannot be empty.")

    for name in (
        "additions",
        "deletions",
        "files_changed"
    ):

        value = state[name]

        if type(value) is not int:
            raise ValueError(
                f"{name} must be an integer."
            )

        if value < 0:
            raise ValueError(
                f"{name} cannot be negative."
            )

    if state["files_changed"] == 0:
        raise ValueError(
            "files_changed must be greater than zero."
        )


# ============================================================
# 4. SAFE QWEN CALL
# ============================================================

def invoke_qwen(prompt: str) -> str:

    try:

        response = model.invoke(prompt)

        content = response.content

        if not isinstance(content, str):
            raise ValueError(
                "Qwen returned non-text content."
            )

        if not content.strip():
            raise ValueError(
                "Qwen returned an empty response."
            )

        return content

    except Exception as error:

        logger.exception("Qwen invocation failed")

        raise RuntimeError(
            "Qwen review failed. "
            "Check whether Ollama is running."
        ) from error


# ============================================================
# 5. AST SOURCE-CODE SECURITY ANALYSIS
# ============================================================

def check_source_security(code: str) -> list[str]:

    try:

        findings = analyze_source_security(code)

        if not isinstance(findings, list):
            raise TypeError(
                "AST analyzer must return a list."
            )

        return findings

    except Exception as error:

        logger.exception("AST analysis failed")

        raise RuntimeError(
            "AST security analysis failed. "
            "Security status is unknown."
        ) from error


def format_ast_findings(findings: list[str]) -> str:

    if not findings:
        return "No matching AST security patterns detected."

    return "\n".join(
        f"- {finding}"
        for finding in findings
    )


# ============================================================
# 6. VALIDATE AI-GENERATED CODE EXAMPLES
# ============================================================

def validate_ai_review(review_text: str) -> str:

    code_blocks = extract_python_code_blocks(
        review_text
    )

    for code in code_blocks:

        if contains_insecure_password_fallback(code):

            print(
                "WARNING: Insecure AI fallback detected!"
            )

            return (
                "SECURITY VALIDATION FAILED:\n"
                "The AI response contains an insecure "
                "credential fallback in a Python example.\n"
                "The example has been withheld.\n"
                "Use a required environment variable "
                "or a secure secrets manager."
            )

        if contains_password_print(code):

            print(
                "WARNING: Sensitive-data print detected!"
            )

            return (
                "SECURITY VALIDATION FAILED:\n"
                "The AI response contains a Python "
                "example that may print credentials.\n"
                "The example has been withheld.\n"
                "Do not log passwords or tokens."
            )

    return review_text


def clean_ai_security_response(
    review: str,
    ast_findings: list[str]
) -> str:

    if not ast_findings:
        return review

    normalized = review.strip().lower()

    if normalized in (
        "no security issues identified.",
        "no security issues identified",
    ):
        return (
            "The AI did not identify additional "
            "security issues. See the AST findings."
        )

    # Remove a contradictory standalone conclusion
    # without discarding useful AI explanations.
    lines = review.splitlines()

    cleaned_lines = [
        line
        for line in lines
        if line.strip().lower() not in (
            "no security issues identified.",
            "no security issues identified",
        )
    ]

    cleaned = "\n".join(cleaned_lines).strip()

    return cleaned or (
        "See the independent AST security findings."
    )


# ============================================================
# 7. MCP CALCULATIONS
# ============================================================

async def calculate_changes_node(state: AgentState):

    print("Running calculate changes node...")

    additions = state["additions"]
    deletions = state["deletions"]
    files_changed = state["files_changed"]

    print("Calling MCP server from LangGraph...")

    total_changes = await calculate_changes_via_mcp(
        additions,
        deletions
    )

    total_changes = int(total_changes)

    expected_total = additions + deletions

    if total_changes != expected_total:
        raise ValueError(
            "MCP calculation returned an incorrect total."
        )

    print(
        "MCP result received by LangGraph:",
        total_changes
    )

    average_changes_per_file = (
        total_changes / files_changed
    )

    if total_changes > 0:

        addition_ratio = additions / total_changes
        deletion_ratio = deletions / total_changes

    else:

        addition_ratio = 0.0
        deletion_ratio = 0.0

    return {
        "total_changes": total_changes,
        "average_changes_per_file":
            average_changes_per_file,
        "addition_ratio": addition_ratio,
        "deletion_ratio": deletion_ratio,
    }


# ============================================================
# 8. RAG RETRIEVAL
# ============================================================

def retrieve_context_node(state: AgentState):

    print("Running RAG retrieval node...")

    context = retrieve_context(
        state["code"]
    )

    if context is None:
        raise RuntimeError(
            "RAG retrieval returned no context."
        )

    return {
        "rag_context": str(context)
    }


# ============================================================
# 9. ML RISK PREDICTION
# ============================================================

def check_risk_node(state: AgentState):

    print("Running ML risk prediction node...")

    features = {
        "total_changes":
            state["total_changes"],

        "files_changed":
            state["files_changed"],

        "average_changes_per_file":
            state["average_changes_per_file"],

        "addition_ratio":
            state["addition_ratio"],

        "deletion_ratio":
            state["deletion_ratio"],
    }

    risk_level, risk_score = predict_risk_with_ml(
        features
    )

    risk_level = str(risk_level).lower()
    risk_score = float(risk_score)

    if risk_level not in (
        "low",
        "medium",
        "high"
    ):
        raise ValueError(
            f"Unexpected ML risk: {risk_level}"
        )

    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(
            "ML risk score must be between 0 and 1."
        )

    print("Predicted risk:", risk_level)
    print("Risk score:", risk_score)

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
    }


# ============================================================
# 10. RISK ROUTING
# ============================================================

def route_by_risk(state: AgentState):

    risk = state["risk_level"]

    if risk in (
        "low",
        "medium",
        "high"
    ):
        return risk

    raise ValueError(
        f"Invalid risk level: {risk}"
    )


# ============================================================
# 11. LOW-RISK REVIEW AGENT
# ============================================================

def llm_review_node(state: AgentState):

    print("Running general LLM review...")

    source_code = state["code"]

    ast_findings = check_source_security(
        source_code
    )

    if ast_findings:

        print(
            "AST detected security findings "
            "in low-risk source code."
        )

    else:

        print(
            "AST found no matching security patterns."
        )

    ast_context = format_ast_findings(
        ast_findings
    )

    prompt = f"""
You are an AI Code Review Agent.

Review the following source code:

{source_code}

ML predicted risk:

{state["risk_level"]}

Retrieved coding guidelines:

{state["rag_context"]}

Independent AST security findings:

{ast_context}

Review the code for:

- Security vulnerabilities
- Correctness problems
- Code quality problems

Treat the AST findings as potential vulnerabilities.

Do not ignore a hardcoded credential.

Never recommend hardcoded passwords, API keys,
tokens, or fallback credentials.

Use a required environment variable or
a secure secrets manager.

Do not recommend unsafe logging of credentials.

Provide a short explanation and recommendation.
"""

    ai_review = invoke_qwen(prompt)

    print("Running AI output security validation...")

    validated_review = validate_ai_review(
        ai_review
    )

    validated_review = clean_ai_security_response(
        validated_review,
        ast_findings
    )

    if ast_findings:

        final_review = (
            "AST SECURITY FINDINGS:\n"
            + ast_context
            + "\n\nAI CODE REVIEW:\n"
            + validated_review
        )

    else:

        final_review = validated_review

    return {
        "ast_findings": ast_findings,
        "llm_review": validated_review,
        "final_response": final_review,
    }


# ============================================================
# 12. SECURITY REVIEW AGENT
# ============================================================

def security_review_node(state: AgentState):

    print("Running Security Review Agent...")

    source_code = state["code"]

    ast_findings = check_source_security(
        source_code
    )

    if ast_findings:

        print(
            "AST detected security findings "
            "in source code."
        )

    else:

        print(
            "AST found no matching security patterns."
        )

    ast_context = format_ast_findings(
        ast_findings
    )

    prompt = f"""
You are a Security Code Review Agent.

Review ONLY security-related problems.

Source code:

{source_code}

Retrieved coding guidelines:

{state["rag_context"]}

Independent AST security findings:

{ast_context}

Focus on:

- Hardcoded passwords and API keys
- Authentication and authorization
- Injection vulnerabilities
- Unsafe input handling
- Sensitive data exposure

A credential assigned a literal string
is a hardcoded secret.

Explain relevant AST findings.

Never recommend hardcoded fallback credentials.

Use required environment variables or
a secure secrets manager.

Do not recommend printing secrets.

If no security problem is identified, respond:

No security issues identified.

Otherwise provide a short explanation
and recommendation.
"""

    ai_review = invoke_qwen(prompt)

    ai_review = validate_ai_review(
        ai_review
    )

    ai_review = clean_ai_security_response(
        ai_review,
        ast_findings
    )

    if ast_findings:

        combined_security_review = (
            "AST SECURITY FINDINGS:\n"
            + ast_context
            + "\n\nAI SECURITY REVIEW:\n"
            + ai_review
        )

    else:

        combined_security_review = ai_review

    return {
        "ast_findings": ast_findings,
        "security_review": combined_security_review,
    }


# ============================================================
# 13. CODE QUALITY AGENT
# ============================================================

def quality_review_node(state: AgentState):

    print("Running Code Quality Review Agent...")

    security_finding = state.get(
        "security_review"
    )

    print(
        "Security finding received by Quality Agent:"
    )
    print(security_finding)

    prompt = f"""
You are a Code Quality Review Agent.

Review the following source code:

{state["code"]}

Retrieved coding guidelines:

{state["rag_context"]}

Security Agent findings:

{security_finding or "Not available."}

Security vulnerabilities are handled separately.

Do not repeat security findings.

Review only:

- Readability
- Maintainability
- Naming conventions
- Duplicate code
- Unnecessary complexity

Do not invent issues.

If there are no independent quality problems,
respond exactly:

No code quality issues identified.

Otherwise explain the most important
quality issue and recommendation.
"""

    ai_review = invoke_qwen(prompt)

    validated_review = validate_ai_review(
        ai_review
    )

    return {
        "quality_review": validated_review
    }


# ============================================================
# 14. PERFORMANCE AGENT
# ============================================================

def performance_review_node(state: AgentState):

    print("Running Performance Review Agent...")

    prompt = f"""
You are a Performance Code Review Agent.

Review ONLY performance-related problems.

Source code:

{state["code"]}

Retrieved coding guidelines:

{state["rag_context"]}

Focus on:

- Inefficient loops
- Repeated expensive operations
- Expensive database calls
- Excessive memory usage
- Inefficient algorithms

Do not report security or code quality problems.

Do not invent performance issues.

If no clear performance issue exists,
respond exactly:

No performance issues identified.

Otherwise explain the most important
performance issue and recommendation.
"""

    ai_review = invoke_qwen(prompt)

    validated_review = validate_ai_review(
        ai_review
    )

    return {
        "performance_review": validated_review
    }


# ============================================================
# 15. MEDIUM-RISK REVIEW AGGREGATOR
# ============================================================

def combine_medium_reviews_node(state: AgentState):

    print("Combining medium-risk reviews...")

    security_review = state.get(
        "security_review"
    )

    quality_review = state.get(
        "quality_review"
    )

    if not security_review or not quality_review:
        raise RuntimeError(
            "Medium-risk review is incomplete."
        )

    combined_review = (
        "SECURITY REVIEW:\n\n"
        + security_review
        + "\n\nCODE QUALITY REVIEW:\n\n"
        + quality_review
    )

    # Individual AI outputs have already been validated.
    # Preserve the independent AST security findings.
    return {
        "final_response": combined_review
    }


# ============================================================
# 16. HIGH-RISK REVIEW AGGREGATOR
# ============================================================

def combine_reviews_node(state: AgentState):

    print("Combining high-risk reviews...")

    security_review = state.get(
        "security_review"
    )

    quality_review = state.get(
        "quality_review"
    )

    performance_review = state.get(
        "performance_review"
    )

    if not all((
        security_review,
        quality_review,
        performance_review
    )):
        raise RuntimeError(
            "High-risk review is incomplete."
        )

    combined_review = (
        "SECURITY REVIEW:\n\n"
        + security_review
        + "\n\nCODE QUALITY REVIEW:\n\n"
        + quality_review
        + "\n\nPERFORMANCE REVIEW:\n\n"
        + performance_review
    )

    return {
        "final_response": combined_review
    }


# ============================================================
# 17. BUILD LANGGRAPH
# ============================================================

def build_graph():

    graph_builder = StateGraph(AgentState)

    # Common nodes.
    graph_builder.add_node(
        "calculate_changes",
        calculate_changes_node
    )

    graph_builder.add_node(
        "retrieve_context",
        retrieve_context_node
    )

    graph_builder.add_node(
        "check_risk",
        check_risk_node
    )

    # Low-risk node.
    graph_builder.add_node(
        "llm_review",
        llm_review_node
    )

    # Medium/high shared agents.
    graph_builder.add_node(
        "security_review",
        security_review_node
    )

    graph_builder.add_node(
        "quality_review",
        quality_review_node
    )

    # High-risk agent.
    graph_builder.add_node(
        "performance_review",
        performance_review_node
    )

    # Aggregators.
    graph_builder.add_node(
        "combine_medium_reviews",
        combine_medium_reviews_node
    )

    graph_builder.add_node(
        "combine_reviews",
        combine_reviews_node
    )

    # Common execution path.
    graph_builder.add_edge(
        START,
        "calculate_changes"
    )

    graph_builder.add_edge(
        "calculate_changes",
        "retrieve_context"
    )

    graph_builder.add_edge(
        "retrieve_context",
        "check_risk"
    )

    # Exactly one initial risk decision.
    graph_builder.add_conditional_edges(
        "check_risk",
        route_by_risk,
        {
            "low": "llm_review",
            "medium": "security_review",
            "high": "security_review",
        }
    )

    # Low-risk workflow.
    graph_builder.add_edge(
        "llm_review",
        END
    )

    # Shared medium/high review sequence.
    graph_builder.add_edge(
        "security_review",
        "quality_review"
    )

    # Choose the next step after Quality Agent.
    graph_builder.add_conditional_edges(
        "quality_review",
        route_by_risk,
        {
            "medium": "combine_medium_reviews",
            "high": "performance_review",
        }
    )

    # Medium-risk workflow ends here.
    graph_builder.add_edge(
        "combine_medium_reviews",
        END
    )

    # High-risk workflow continues through Performance.
    graph_builder.add_edge(
        "performance_review",
        "combine_reviews"
    )

    graph_builder.add_edge(
        "combine_reviews",
        END
    )

    return graph_builder.compile()


graph = build_graph()


# ============================================================
# 18. INITIAL TEST STATE
# ============================================================

initial_state: AgentState = {

    "user_request": "Review this Pull Request",

    "code": 'password = "admin123"',

    # Low-risk test.
    "additions": 400,
    "deletions": 100,
    "files_changed": 20,

    "total_changes": None,
    "average_changes_per_file": None,
    "addition_ratio": None,
    "deletion_ratio": None,

    "risk_level": None,
    "risk_score": None,

    "rag_context": None,

    "ast_findings": [],

    "security_review": None,
    "quality_review": None,
    "performance_review": None,
    "llm_review": None,

    "final_response": None,
}


# ============================================================
# 19. SAFE GRAPH EXECUTION
# ============================================================

async def run_review_safely(state: AgentState):

    try:

        validate_input(state)

        result = await graph.ainvoke(state)

        if not result.get("final_response"):
            raise RuntimeError(
                "Workflow completed without a final review."
            )

        return {
            "status": "success",
            "result": result,
        }

    except Exception:

        logger.exception(
            "LangGraph workflow failed"
        )

        return {
            "status": "failed",
            "message": (
                "Code review could not be completed. "
                "Check the terminal logs for details."
            ),
            "result": None,
        }


# ============================================================
# 20. MAIN FUNCTION
# ============================================================

async def main():

    workflow_result = await run_review_safely(
        initial_state
    )

    if workflow_result["status"] == "failed":

        print("\n========== REVIEW FAILED ==========")

        print(
            workflow_result["message"]
        )

        return

    result = workflow_result["result"]

    print("\n========== FINAL RESULTS ==========")

    print(
        "Risk level:",
        result["risk_level"]
    )

    print(
        "Risk score:",
        result["risk_score"]
    )

    print(
        "Total changes:",
        result["total_changes"]
    )

    print("\nSecurity review:")
    print(result["security_review"])

    print("\nQuality review:")
    print(result["quality_review"])

    print("\nPerformance review:")
    print(result["performance_review"])

    print("\n========== FINAL REVIEW ==========")

    print(result["final_response"])


if __name__ == "__main__":
    asyncio.run(main())
