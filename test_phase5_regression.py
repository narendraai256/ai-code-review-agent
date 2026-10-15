
import unittest
from unittest.mock import Mock

from security_validator import analyze_source_security

from workflow_safety import (
    ReviewServiceError,
    safe_llm_invoke,
    safe_ast_analysis,
    validate_review_input,
    build_failure_response
)


class TestPhase5Security(unittest.TestCase):

    def test_hardcoded_password(self):
        findings = analyze_source_security(
            'password = "admin123"'
        )

        self.assertTrue(
            any("HARDCODED SECRET" in f for f in findings)
        )

    def test_hardcoded_api_key(self):
        findings = analyze_source_security(
            'api_key = "example-key-value"'
        )

        self.assertTrue(
            any("HARDCODED SECRET" in f for f in findings)
        )

    def test_environment_variable_without_fallback(self):
        findings = analyze_source_security(
            'import os\npassword = os.environ["PASSWORD"]'
        )

        self.assertEqual(findings, [])

    def test_insecure_environment_fallback(self):
        findings = analyze_source_security(
            'import os\npassword = os.getenv("PASSWORD", "admin123")'
        )

        self.assertTrue(
            any("FALLBACK" in f.upper() for f in findings)
        )

    def test_safe_ast_wrapper(self):
        findings = safe_ast_analysis(
            'password = "admin123"',
            analyze_source_security
        )

        self.assertTrue(findings)

    def test_ast_failure_is_not_reported_as_safe(self):
        def broken_analyzer(code):
            raise RuntimeError("Analyzer unavailable")

        with self.assertRaises(ReviewServiceError):
            safe_ast_analysis(
                'password = "admin123"',
                broken_analyzer
            )


class TestPhase5LLM(unittest.TestCase):

    def test_successful_llm_response(self):
        model = Mock()
        model.invoke.return_value.content = (
            "Hardcoded password detected."
        )

        result = safe_llm_invoke(
            model,
            "Review this code"
        )

        self.assertIn("password", result)

    def test_llm_connection_failure(self):
        model = Mock()
        model.invoke.side_effect = ConnectionError(
            "Ollama unavailable"
        )

        with self.assertRaises(ReviewServiceError):
            safe_llm_invoke(
                model,
                "Review this code"
            )

    def test_empty_llm_response(self):
        model = Mock()
        model.invoke.return_value.content = ""

        with self.assertRaises(ReviewServiceError):
            safe_llm_invoke(
                model,
                "Review this code"
            )


class TestPhase5Inputs(unittest.TestCase):

    def test_valid_input(self):
        validate_review_input(
            "print('hello')",
            10,
            5,
            2
        )

    def test_empty_code(self):
        with self.assertRaises(ValueError):
            validate_review_input("", 10, 5, 2)

    def test_negative_additions(self):
        with self.assertRaises(ValueError):
            validate_review_input(
                "print('hello')",
                -1,
                5,
                2
            )

    def test_zero_files_changed(self):
        with self.assertRaises(ValueError):
            validate_review_input(
                "print('hello')",
                10,
                5,
                0
            )

    def test_boolean_is_not_valid_count(self):
        with self.assertRaises(ValueError):
            validate_review_input(
                "print('hello')",
                True,
                5,
                2
            )

    def test_failure_response(self):
        result = build_failure_response(
            "ollama",
            RuntimeError("Service unavailable")
        )

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["security_verified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
