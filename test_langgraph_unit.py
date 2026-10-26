
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import test_langgraph as workflow


class TestLangGraphUnit(unittest.TestCase):

    def setUp(self):
        # Give each test its own independent workflow state.
        self.state = workflow.initial_state.copy()
        self.state["ast_findings"] = []

        self.state.update({
            "code": 'print("hello")',
            "additions": 80,
            "deletions": 20,
            "files_changed": 5,
            "total_changes": 100,
            "average_changes_per_file": 20,
            "addition_ratio": 0.8,
            "deletion_ratio": 0.2,
            "risk_level": "low",
            "risk_score": 0.9,
            "rag_context": "Follow secure coding guidelines."
        })

    # ============================================================
    # 1. INPUT VALIDATION
    # ============================================================

    def test_valid_input(self):
        self.assertIsNone(workflow.validate_input(self.state))

    def test_empty_code(self):
        self.state["code"] = " "

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            workflow.validate_input(self.state)

    def test_invalid_numeric_type(self):
        self.state["additions"] = True

        with self.assertRaisesRegex(ValueError, "must be an integer"):
            workflow.validate_input(self.state)

    def test_negative_changes(self):
        self.state["deletions"] = -1

        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            workflow.validate_input(self.state)

    def test_zero_files(self):
        self.state["files_changed"] = 0

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            workflow.validate_input(self.state)

    # ============================================================
    # 2. QWEN RESPONSE VALIDATION
    # ============================================================

    def test_qwen_success(self):
        with patch.object(workflow, "model") as model:
            model.invoke.return_value.content = "Review completed"

            self.assertEqual(
                workflow.invoke_qwen("Review this code"),
                "Review completed"
            )

    def test_qwen_empty_response(self):
        with patch.object(workflow, "model") as model:
            model.invoke.return_value.content = " "

            with self.assertRaises(RuntimeError):
                workflow.invoke_qwen("Review this code")

    def test_qwen_non_text_response(self):
        with patch.object(workflow, "model") as model:
            model.invoke.return_value.content = ["unexpected"]

            with self.assertRaises(RuntimeError):
                workflow.invoke_qwen("Review this code")

    def test_qwen_connection_failure(self):
        with patch.object(workflow, "model") as model:
            model.invoke.side_effect = ConnectionError("Offline")

            with self.assertRaises(RuntimeError):
                workflow.invoke_qwen("Review this code")

    # ============================================================
    # 3. AST SECURITY
    # ============================================================

    def test_ast_findings(self):
        with patch.object(
            workflow,
            "analyze_source_security",
            return_value=["Hardcoded password"]
        ):
            result = workflow.check_source_security(
                "password = 'abc'"
            )

            self.assertEqual(result, ["Hardcoded password"])

    def test_ast_invalid_result(self):
        with patch.object(
            workflow,
            "analyze_source_security",
            return_value="incorrect format"
        ):
            with self.assertRaises(RuntimeError):
                workflow.check_source_security("x = 1")

    def test_ast_analyzer_failure(self):
        with patch.object(
            workflow,
            "analyze_source_security",
            side_effect=RuntimeError("Unavailable")
        ):
            with self.assertRaises(RuntimeError):
                workflow.check_source_security("x = 1")

    def test_format_ast_findings(self):
        self.assertIn(
            "Hardcoded password",
            workflow.format_ast_findings(
                ["Hardcoded password"]
            )
        )

        self.assertIn(
            "No matching",
            workflow.format_ast_findings([])
        )

    def test_clean_ai_security_response(self):
        self.assertEqual(
            workflow.clean_ai_security_response(
                "No security issues identified.",
                ["Hardcoded password"]
            ),
            "The AI did not identify additional security issues. "
            "See the AST findings."
        )

        self.assertEqual(
            workflow.clean_ai_security_response(
                "No security issues identified.",
                []
            ),
            "No security issues identified."
        )

        cleaned = workflow.clean_ai_security_response(
            "Potential issue.\nNo security issues identified.",
            ["Hardcoded password"]
        )

        self.assertEqual(cleaned, "Potential issue.")

    def test_validate_ai_review_safe(self):
        with patch.object(
            workflow,
            "extract_python_code_blocks",
            return_value=["print('hello')"]
        ):
            self.assertEqual(
                workflow.validate_ai_review("Safe review"),
                "Safe review"
            )

    def test_validate_ai_review_insecure_fallback(self):
        with patch.object(
            workflow,
            "extract_python_code_blocks",
            return_value=["example"]
        ), patch.object(
            workflow,
            "contains_insecure_password_fallback",
            return_value=True
        ):
            result = workflow.validate_ai_review(
                "Unsafe example"
            )

            self.assertIn(
                "SECURITY VALIDATION FAILED",
                result
            )

    def test_validate_ai_review_password_print(self):
        with patch.object(
            workflow,
            "extract_python_code_blocks",
            return_value=["example"]
        ), patch.object(
            workflow,
            "contains_insecure_password_fallback",
            return_value=False
        ), patch.object(
            workflow,
            "contains_password_print",
            return_value=True
        ):
            result = workflow.validate_ai_review(
                "Unsafe example"
            )

            self.assertIn(
                "Do not log passwords",
                result
            )

    # ============================================================
    # 4. MCP AND RAG
    # ============================================================

    def test_mcp_calculation(self):
        async def run():
            with patch.object(
                workflow,
                "calculate_changes_via_mcp",
                new_callable=AsyncMock
            ) as mock_mcp:

                mock_mcp.return_value = "100"

                result = await workflow.calculate_changes_node(
                    self.state
                )

                self.assertEqual(
                    result["total_changes"],
                    100
                )

                self.assertEqual(
                    result["average_changes_per_file"],
                    20
                )

                self.assertAlmostEqual(
                    result["addition_ratio"],
                    0.8
                )

                mock_mcp.assert_awaited_once_with(80, 20)

        asyncio.run(run())

    def test_mcp_incorrect_total(self):
        async def run():
            with patch.object(
                workflow,
                "calculate_changes_via_mcp",
                new_callable=AsyncMock,
                return_value=999
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "incorrect total"
                ):
                    await workflow.calculate_changes_node(
                        self.state
                    )

        asyncio.run(run())

    def test_mcp_zero_changes(self):
        self.state["additions"] = 0
        self.state["deletions"] = 0

        async def run():
            with patch.object(
                workflow,
                "calculate_changes_via_mcp",
                new_callable=AsyncMock,
                return_value=0
            ):
                result = await workflow.calculate_changes_node(
                    self.state
                )

                self.assertEqual(
                    result["addition_ratio"],
                    0
                )

                self.assertEqual(
                    result["deletion_ratio"],
                    0
                )

        asyncio.run(run())

    def test_rag_success(self):
        with patch.object(
            workflow,
            "retrieve_context",
            return_value="Coding guideline"
        ):
            result = workflow.retrieve_context_node(
                self.state
            )

            self.assertEqual(
                result["rag_context"],
                "Coding guideline"
            )

    def test_rag_failure(self):
        with patch.object(
            workflow,
            "retrieve_context",
            return_value=None
        ):
            with self.assertRaises(RuntimeError):
                workflow.retrieve_context_node(self.state)

    # ============================================================
    # 5. ML RISK AND ROUTING
    # ============================================================

    def test_ml_risk_prediction(self):
        with patch.object(
            workflow,
            "predict_risk_with_ml",
            return_value=("HIGH", 0.95)
        ) as predictor:

            result = workflow.check_risk_node(
                self.state
            )

            self.assertEqual(
                result["risk_level"],
                "high"
            )

            self.assertEqual(
                result["risk_score"],
                0.95
            )

            self.assertEqual(
                predictor.call_args.args[0]["total_changes"],
                100
            )

    def test_invalid_ml_risk(self):
        with patch.object(
            workflow,
            "predict_risk_with_ml",
            return_value=("unknown", 0.8)
        ):
            with self.assertRaises(ValueError):
                workflow.check_risk_node(self.state)

    def test_invalid_ml_probability(self):
        with patch.object(
            workflow,
            "predict_risk_with_ml",
            return_value=("low", 1.5)
        ):
            with self.assertRaises(ValueError):
                workflow.check_risk_node(self.state)

    def test_route_all_risk_levels(self):
        for risk in ("low", "medium", "high"):
            with self.subTest(risk=risk):

                self.state["risk_level"] = risk

                self.assertEqual(
                    workflow.route_by_risk(self.state),
                    risk
                )

    def test_route_invalid_risk(self):
        self.state["risk_level"] = "critical"

        with self.assertRaises(ValueError):
            workflow.route_by_risk(self.state)

    # ============================================================
    # 6. REVIEW AGENTS
    # ============================================================

    def test_low_risk_review_with_ast(self):
        with patch.object(
            workflow,
            "check_source_security",
            return_value=["Hardcoded password"]
        ), patch.object(
            workflow,
            "invoke_qwen",
            return_value="No security issues identified."
        ):
            result = workflow.llm_review_node(
                self.state
            )

            self.assertIn(
                "AST SECURITY FINDINGS",
                result["final_response"]
            )

            self.assertIn(
                "Hardcoded password",
                result["final_response"]
            )

            self.assertIn(
                "See the AST findings",
                result["llm_review"]
            )

    def test_low_risk_review_without_ast(self):
        with patch.object(
            workflow,
            "check_source_security",
            return_value=[]
        ), patch.object(
            workflow,
            "invoke_qwen",
            return_value="Code looks good."
        ):
            result = workflow.llm_review_node(
                self.state
            )

            self.assertEqual(
                result["final_response"],
                "Code looks good."
            )

    def test_security_agent_with_ast(self):
        with patch.object(
            workflow,
            "check_source_security",
            return_value=["Hardcoded password"]
        ), patch.object(
            workflow,
            "invoke_qwen",
            return_value="No security issues identified."
        ):
            result = workflow.security_review_node(
                self.state
            )

            self.assertIn(
                "Hardcoded password",
                result["security_review"]
            )

    def test_security_agent_without_ast(self):
        with patch.object(
            workflow,
            "check_source_security",
            return_value=[]
        ), patch.object(
            workflow,
            "invoke_qwen",
            return_value="No security issues identified."
        ):
            result = workflow.security_review_node(
                self.state
            )

            self.assertEqual(
                result["security_review"],
                "No security issues identified."
            )

    def test_quality_agent(self):
        with patch.object(
            workflow,
            "invoke_qwen",
            return_value="Improve variable naming."
        ):
            result = workflow.quality_review_node(
                self.state
            )

            self.assertEqual(
                result["quality_review"],
                "Improve variable naming."
            )

    def test_performance_agent(self):
        with patch.object(
            workflow,
            "invoke_qwen",
            return_value="Avoid repeated database queries."
        ):
            result = workflow.performance_review_node(
                self.state
            )

            self.assertIn(
                "database queries",
                result["performance_review"]
            )

    # ============================================================
    # 7. AGGREGATION
    # ============================================================

    def test_medium_review_combination(self):
        self.state["security_review"] = "Security checked"
        self.state["quality_review"] = "Quality checked"

        result = workflow.combine_medium_reviews_node(
            self.state
        )

        self.assertIn(
            "Security checked",
            result["final_response"]
        )

        self.assertIn(
            "Quality checked",
            result["final_response"]
        )

    def test_medium_review_incomplete(self):
        with self.assertRaises(RuntimeError):
            workflow.combine_medium_reviews_node(
                self.state
            )

    def test_high_review_combination(self):
        self.state["security_review"] = "Security checked"
        self.state["quality_review"] = "Quality checked"
        self.state["performance_review"] = "Performance checked"

        result = workflow.combine_reviews_node(
            self.state
        )

        self.assertIn(
            "Security checked",
            result["final_response"]
        )

        self.assertIn(
            "Quality checked",
            result["final_response"]
        )

        self.assertIn(
            "Performance checked",
            result["final_response"]
        )

    def test_high_review_incomplete(self):
        with self.assertRaises(RuntimeError):
            workflow.combine_reviews_node(
                self.state
            )

    # ============================================================
    # 8. SAFE WORKFLOW EXECUTION
    # ============================================================

    def test_safe_workflow_success(self):
        async def run():
            with patch.object(
                workflow,
                "graph"
            ) as mock_graph:

                mock_graph.ainvoke = AsyncMock(
                    return_value={
                        "final_response": "Review completed"
                    }
                )

                result = await workflow.run_review_safely(
                    self.state
                )

                self.assertEqual(
                    result["status"],
                    "success"
                )

                self.assertEqual(
                    result["result"]["final_response"],
                    "Review completed"
                )

        asyncio.run(run())

    def test_safe_workflow_missing_response(self):
        async def run():
            with patch.object(
                workflow,
                "graph"
            ) as mock_graph:

                mock_graph.ainvoke = AsyncMock(
                    return_value={"final_response": ""}
                )

                result = await workflow.run_review_safely(
                    self.state
                )

                self.assertEqual(
                    result["status"],
                    "failed"
                )

                self.assertIsNone(
                    result["result"]
                )

        asyncio.run(run())

    def test_safe_workflow_invalid_input(self):
        self.state["code"] = ""

        async def run():
            with patch.object(
                workflow,
                "graph"
            ) as mock_graph:

                mock_graph.ainvoke = AsyncMock()

                result = await workflow.run_review_safely(
                    self.state
                )

                self.assertEqual(
                    result["status"],
                    "failed"
                )

                mock_graph.ainvoke.assert_not_awaited()

        asyncio.run(run())

    # ============================================================
    # 9. COMPLETE LANGGRAPH RISK ROUTES
    # ============================================================

    def test_complete_graph_all_risk_paths(self):
        async def run_one_path(risk):
            state = self.state.copy()

            state["risk_level"] = None
            state["risk_score"] = None
            state["final_response"] = None

            visited = []

            def fake_qwen(prompt):
                if "Security Code Review Agent" in prompt:
                    visited.append("security")
                    return "Security review completed"

                if "Code Quality Review Agent" in prompt:
                    visited.append("quality")
                    return "Quality review completed"

                if "Performance Code Review Agent" in prompt:
                    visited.append("performance")
                    return "Performance review completed"

                visited.append("general")
                return "General review completed"

            with patch.object(
                workflow,
                "calculate_changes_via_mcp",
                new_callable=AsyncMock,
                return_value=100
            ), patch.object(
                workflow,
                "retrieve_context",
                return_value="Secure coding guidelines"
            ), patch.object(
                workflow,
                "predict_risk_with_ml",
                return_value=(risk, 0.95)
            ), patch.object(
                workflow,
                "check_source_security",
                return_value=[]
            ), patch.object(
                workflow,
                "invoke_qwen",
                side_effect=fake_qwen
            ):
                result = await workflow.run_review_safely(
                    state
                )

            self.assertEqual(
                result["status"],
                "success"
            )

            self.assertEqual(
                result["result"]["risk_level"],
                risk
            )

            self.assertTrue(
                result["result"]["final_response"]
            )

            return visited

        async def run():
            expected = {
                "low": ["general"],
                "medium": ["security", "quality"],
                "high": [
                    "security",
                    "quality",
                    "performance"
                ]
            }

            for risk, expected_nodes in expected.items():
                with self.subTest(risk=risk):
                    visited = await run_one_path(risk)

                    self.assertEqual(
                        visited,
                        expected_nodes
                    )

        asyncio.run(run())

    # ============================================================
    # 10. MAIN FUNCTION OUTPUT
    # ============================================================

    def test_main_success(self):
        async def run():
            fake_result = {
                "risk_level": "high",
                "risk_score": 0.95,
                "total_changes": 100,
                "security_review": "Security checked",
                "quality_review": "Quality checked",
                "performance_review": "Performance checked",
                "final_response": "Review completed"
            }

            with patch.object(
                workflow,
                "run_review_safely",
                new_callable=AsyncMock
            ) as mock_run, patch(
                "builtins.print"
            ) as mock_print:

                mock_run.return_value = {
                    "status": "success",
                    "result": fake_result
                }

                await workflow.main()

                mock_run.assert_awaited_once_with(
                    workflow.initial_state
                )

                printed = " ".join(
                    str(argument)
                    for call in mock_print.call_args_list
                    for argument in call.args
                )

                self.assertIn(
                    "Review completed",
                    printed
                )

                self.assertIn(
                    "high",
                    printed
                )

        asyncio.run(run())

    def test_main_failure(self):
        async def run():
            with patch.object(
                workflow,
                "run_review_safely",
                new_callable=AsyncMock
            ) as mock_run, patch(
                "builtins.print"
            ) as mock_print:

                mock_run.return_value = {
                    "status": "failed",
                    "message": "Workflow unavailable",
                    "result": None
                }

                await workflow.main()

                mock_run.assert_awaited_once_with(
                    workflow.initial_state
                )

                printed = " ".join(
                    str(argument)
                    for call in mock_print.call_args_list
                    for argument in call.args
                )

                self.assertIn(
                    "REVIEW FAILED",
                    printed
                )

                self.assertIn(
                    "Workflow unavailable",
                    printed
                )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
