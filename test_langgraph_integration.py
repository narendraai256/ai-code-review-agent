import asyncio
from copy import deepcopy

from test_langgraph import initial_state, run_review_safely


# Defines real input scenarios for all three ML risk categories.
SCENARIOS = [
    ("low", 5, 2, 1),
    ("medium", 80, 20, 5),
    ("high", 400, 100, 20),
]


# Runs the actual LangGraph workflow for each scenario.
async def main():

    for expected_risk, additions, deletions, files_changed in SCENARIOS:

        print(f"\n========== TESTING {expected_risk.upper()} RISK ==========")

        # Creates an independent copy of the original workflow state.
        state = deepcopy(initial_state)

        state["additions"] = additions
        state["deletions"] = deletions
        state["files_changed"] = files_changed

        # Uses harmless sample code for low and medium risk.
        if expected_risk != "high":
            state["code"] = "result = 10 + 20"

        workflow_result = await run_review_safely(state)

        if workflow_result["status"] != "success":
            raise RuntimeError(
                f"{expected_risk} workflow failed: "
                f"{workflow_result['message']}"
            )

        result = workflow_result["result"]

        print("Expected risk:", expected_risk)
        print("Actual risk:", result["risk_level"])
        print("Risk score:", result["risk_score"])
        print("Total changes:", result["total_changes"])

        # Confirms LangGraph selected the correct risk path.
        assert result["risk_level"] == expected_risk, (
            f"Expected {expected_risk}, got {result['risk_level']}"
        )

        assert result["final_response"], (
            "Final review is missing."
        )

        if expected_risk == "low":
            assert result["llm_review"], (
                "Low-risk general review is missing."
            )

        if expected_risk in ("medium", "high"):
            assert result["security_review"], (
                "Security review is missing."
            )
            assert result["quality_review"], (
                "Quality review is missing."
            )

        if expected_risk == "high":
            assert result["performance_review"], (
                "Performance review is missing."
            )

        print(f"{expected_risk.upper()} RISK TEST PASSED")

    print("\nALL THREE LIVE INTEGRATION TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
