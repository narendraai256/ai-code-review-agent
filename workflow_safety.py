
"""
Shared safety utilities for the AI Code Review Agent.

These helpers make external-service failures easier to
handle without silently claiming that code is safe.
"""

import logging

logger = logging.getLogger(__name__)


class ReviewServiceError(Exception):
    """Raised when a required review service fails."""


def safe_llm_invoke(model, prompt: str) -> str:
    """
    Invoke the configured LLM.

    Raises ReviewServiceError if the LLM is unavailable
    or returns an invalid response.
    """
    try:
        response = model.invoke(prompt)
        content = getattr(response, "content", None)

        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM returned an empty response.")

        return content

    except Exception as exc:
        logger.exception("LLM invocation failed")
        raise ReviewServiceError(
            "LLM review unavailable. Review is incomplete."
        ) from exc


def safe_ast_analysis(code: str, analyzer) -> list[str]:
    """
    Run source-code security analysis.

    An analyzer failure is not treated as a clean result.
    """
    try:
        findings = analyzer(code)

        if not isinstance(findings, list):
            raise TypeError(
                "Security analyzer must return a list."
            )

        return findings

    except Exception as exc:
        logger.exception("AST security analysis failed")
        raise ReviewServiceError(
            "AST security analysis failed. "
            "Security status is unknown."
        ) from exc


def validate_review_input(
    code: str,
    additions: int,
    deletions: int,
    files_changed: int
) -> None:
    """
    Validate the main inputs before starting analysis.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Source code cannot be empty.")

    for name, value in (
        ("additions", additions),
        ("deletions", deletions),
        ("files_changed", files_changed)
    ):
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer.")

    if additions < 0 or deletions < 0:
        raise ValueError(
            "Additions and deletions cannot be negative."
        )

    if files_changed <= 0:
        raise ValueError(
            "files_changed must be greater than zero."
        )


def build_failure_response(
    component: str,
    error: Exception
) -> dict:
    """
    Return a structured failure result.

    Do not include raw exception details in the
    public-facing review.
    """
    logger.error(
        "Review component %s failed: %s",
        component,
        error
    )

    return {
        "status": "failed",
        "component": component,
        "message": (
            "Review incomplete because a required "
            "component failed."
        ),
        "security_verified": False
    }

