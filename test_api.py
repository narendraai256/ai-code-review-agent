
import os
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from fastapi.testclient import TestClient

import main


class TestFastAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(main.app)

        self.valid_pr = {
            "repository": "test-repository",
            "pr_number": 101,
            "additions": 80,
            "deletions": 20,
            "files_changed": 5,
            "code": 'password = "admin123"'
        }

        # Sample database row containing 14 columns.
        self.sample_row = (
            1, "test-repository", 101, 100, 5,
            "high", 0.95, 20.0, 0.2, 0.8,
            "security", "high",
            "Hardcoded password", "Use environment variables"
        )

    # Creates a simulated PostgreSQL connection and cursor.
    def make_mock_database(self):
        connection = MagicMock()
        cursor = MagicMock()
        connection.cursor.return_value = cursor
        return connection, cursor

    # TEST 1: Health check.
    def test_health_check(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["message"],
            "AI Code Review Agent is running"
        )

    # TEST 2: Database health endpoint.
    def test_database_connection(self):
        with patch("main.get_db_connection") as mock_db:
            connection = MagicMock()
            mock_db.return_value = connection

            response = self.client.get("/db-test")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["message"],
                "Database connection successful"
            )
            connection.close.assert_called_once()

    # TEST 3: Get all reviews endpoint.
    def test_get_all_reviews_endpoint(self):
        reviews = [{"id": 1, "repository": "test-repository"}]

        with patch(
            "main.get_all_reviews",
            return_value=reviews
        ):
            response = self.client.get("/reviews")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), reviews)

    # TEST 4: Get one review endpoint.
    def test_get_review_by_id_endpoint(self):
        review = {"id": 1, "repository": "test-repository"}

        with patch(
            "main.get_review_by_id",
            return_value=review
        ):
            response = self.client.get("/reviews/1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), review)

    # TEST 5: Review not found.
    def test_review_not_found(self):
        with patch(
            "main.get_review_by_id",
            return_value=None
        ):
            response = self.client.get("/reviews/999")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["detail"],
            "Review not found"
        )

    # TEST 6: Invalid PR input.
    def test_invalid_pull_request(self):
        invalid_pr = self.valid_pr.copy()
        invalid_pr["pr_number"] = -1

        response = self.client.post(
            "/analyze",
            json=invalid_pr
        )

        self.assertEqual(response.status_code, 422)

    # TEST 7: Successful AI workflow.
    def test_analyze_success(self):
        result = {
            "risk_level": "high",
            "risk_score": 0.95,
            "final_response": "Hardcoded password detected"
        }

        with patch(
            "main.run_review_safely",
            new_callable=AsyncMock
        ) as mock_workflow, patch(
            "main.save_agent_review"
        ) as mock_save:

            mock_workflow.return_value = {
                "status": "success",
                "result": result
            }

            response = self.client.post(
                "/analyze",
                json=self.valid_pr
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), result)

            mock_workflow.assert_awaited_once()
            mock_save.assert_called_once_with(
                repository="test-repository",
                pr_number=101,
                result=result
            )

    # TEST 8: Failed workflow must not save.
    def test_analyze_workflow_failure(self):
        with patch(
            "main.run_review_safely",
            new_callable=AsyncMock
        ) as mock_workflow, patch(
            "main.save_agent_review"
        ) as mock_save:

            mock_workflow.return_value = {
                "status": "error",
                "message": "AI review service unavailable"
            }

            response = self.client.post(
                "/analyze",
                json=self.valid_pr
            )

            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json()["detail"],
                "AI review service unavailable"
            )
            mock_save.assert_not_called()

    # TEST 9: Normal feature engineering.
    def test_extract_pr_features(self):
        data = main.PullRequestData(**self.valid_pr)

        features = main.extract_pr_features(data)

        self.assertEqual(features["total_changes"], 100)
        self.assertEqual(features["files_changed"], 5)
        self.assertEqual(
            features["average_changes_per_file"], 20
        )
        self.assertAlmostEqual(features["addition_ratio"], 0.8)
        self.assertAlmostEqual(features["deletion_ratio"], 0.2)

    # TEST 10: No changed files.
    def test_extract_pr_features_zero_files(self):
        pr = self.valid_pr.copy()
        pr["files_changed"] = 0

        data = main.PullRequestData(**pr)
        features = main.extract_pr_features(data)

        self.assertEqual(
            features["average_changes_per_file"], 0
        )
        self.assertEqual(features["total_changes"], 100)

    # TEST 11: No additions or deletions.
    def test_extract_pr_features_zero_changes(self):
        pr = self.valid_pr.copy()
        pr["additions"] = 0
        pr["deletions"] = 0

        data = main.PullRequestData(**pr)
        features = main.extract_pr_features(data)

        self.assertEqual(features["total_changes"], 0)
        self.assertEqual(features["addition_ratio"], 0)
        self.assertEqual(features["deletion_ratio"], 0)

    # TEST 12: Real connection function with mocked psycopg.
    def test_get_db_connection_function(self):
        with patch.dict(
            os.environ,
            {"DB_PASSWORD": "test-password"}
        ), patch("main.psycopg.connect") as mock_connect:

            expected_connection = MagicMock()
            mock_connect.return_value = expected_connection

            connection = main.get_db_connection()

            self.assertIs(connection, expected_connection)

            mock_connect.assert_called_once_with(
                host="localhost",
                port=5432,
                dbname="code_review",
                user="postgres",
                password="test-password"
            )

    # TEST 13: Save a basic PR review.
    def test_save_review_function(self):
        connection, cursor = self.make_mock_database()

        with patch(
            "main.get_db_connection",
            return_value=connection
        ):
            main.save_review(
                repository="test-repository",
                pr_number=101,
                total_changes=100,
                files_changed=5,
                average_changes_per_file=20,
                deletion_ratio=0.2,
                addition_ratio=0.8,
                risk_level="high",
                risk_score=0.95,
                issue_type="security",
                severity="high",
                problem="Hardcoded password",
                recommendation="Use environment variables"
            )

        cursor.execute.assert_called_once()

        sql, values = cursor.execute.call_args.args

        self.assertIn("INSERT INTO reviews", sql)
        self.assertEqual(len(values), 13)
        self.assertEqual(values[0], "test-repository")
        self.assertEqual(values[1], 101)
        self.assertEqual(values[7], "high")

        connection.commit.assert_called_once()
        cursor.close.assert_called_once()
        connection.close.assert_called_once()

    # TEST 14: Save LangGraph agent results.
    def test_save_agent_review_function(self):
        connection, cursor = self.make_mock_database()

        result = {
            "total_changes": 100,
            "files_changed": 5,
            "average_changes_per_file": 20,
            "deletion_ratio": 0.2,
            "addition_ratio": 0.8,
            "risk_level": "high",
            "risk_score": 0.95,
            "ast_findings": ["Hardcoded password"],
            "security_review": "Security issue detected",
            "quality_review": "Improve code quality",
            "performance_review": "No performance issue",
            "llm_review": "AI review completed",
            "final_response": "Fix the hardcoded password"
        }

        with patch(
            "main.get_db_connection",
            return_value=connection
        ):
            main.save_agent_review(
                repository="test-repository",
                pr_number=101,
                result=result
            )

        cursor.execute.assert_called_once()

        sql, values = cursor.execute.call_args.args

        self.assertIn("INSERT INTO reviews", sql)
        self.assertEqual(len(values), 15)
        self.assertEqual(values[0], "test-repository")
        self.assertEqual(values[1], 101)
        self.assertEqual(values[7], "high")
        self.assertEqual(values[8], 0.95)

        # Verify PostgreSQL JSONB conversion.
        self.assertIsInstance(values[9], main.Jsonb)
        self.assertEqual(
            values[9].obj,
            ["Hardcoded password"]
        )

        connection.commit.assert_called_once()
        cursor.close.assert_called_once()
        connection.close.assert_called_once()

    # TEST 15: Read populated database results.
    def test_get_all_reviews_function(self):
        connection, cursor = self.make_mock_database()
        cursor.fetchall.return_value = [self.sample_row]

        with patch(
            "main.get_db_connection",
            return_value=connection
        ):
            reviews = main.get_all_reviews()

        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["id"], 1)
        self.assertEqual(
            reviews[0]["repository"],
            "test-repository"
        )
        self.assertEqual(reviews[0]["risk_level"], "high")
        self.assertEqual(reviews[0]["risk_score"], 0.95)
        self.assertEqual(
            reviews[0]["average_changes_per_file"], 20.0
        )
        self.assertEqual(reviews[0]["deletion_ratio"], 0.2)
        self.assertEqual(reviews[0]["addition_ratio"], 0.8)

        cursor.execute.assert_called_once()
        cursor.fetchall.assert_called_once()
        cursor.close.assert_called_once()
        connection.close.assert_called_once()

    # TEST 16: Read empty database results.
    def test_get_all_reviews_empty(self):
        connection, cursor = self.make_mock_database()
        cursor.fetchall.return_value = []

        with patch(
            "main.get_db_connection",
            return_value=connection
        ):
            reviews = main.get_all_reviews()

        self.assertEqual(reviews, [])
        cursor.close.assert_called_once()
        connection.close.assert_called_once()

    # TEST 17: Read rows with nullable numeric columns.
    def test_get_all_reviews_null_values(self):
        connection, cursor = self.make_mock_database()

        row = list(self.sample_row)
        row[7] = None
        row[8] = None
        row[9] = None

        cursor.fetchall.return_value = [tuple(row)]

        with patch(
            "main.get_db_connection",
            return_value=connection
        ):
            reviews = main.get_all_reviews()

        self.assertIsNone(
            reviews[0]["average_changes_per_file"]
        )
        self.assertIsNone(reviews[0]["deletion_ratio"])
        self.assertIsNone(reviews[0]["addition_ratio"])

    # TEST 18: Read one existing review.
    def test_get_review_by_id_function(self):
        connection, cursor = self.make_mock_database()
        cursor.fetchone.return_value = self.sample_row

        with patch(
            "main.get_db_connection",
            return_value=connection
        ):
            review = main.get_review_by_id(1)

        self.assertEqual(review["id"], 1)
        self.assertEqual(review["pr_number"], 101)
        self.assertEqual(review["risk_level"], "high")
        self.assertEqual(review["risk_score"], 0.95)
        self.assertEqual(review["deletion_ratio"], 0.2)

        sql, parameters = cursor.execute.call_args.args

        self.assertIn("WHERE id = %s", sql)
        self.assertEqual(parameters, (1,))

        cursor.close.assert_called_once()
        connection.close.assert_called_once()

    # TEST 19: Missing review in database.
    def test_get_review_by_id_missing(self):
        connection, cursor = self.make_mock_database()
        cursor.fetchone.return_value = None

        with patch(
            "main.get_db_connection",
            return_value=connection
        ):
            review = main.get_review_by_id(999)

        self.assertIsNone(review)
        cursor.close.assert_called_once()
        connection.close.assert_called_once()

    # TEST 20: Nullable fields in one review.
    def test_get_review_by_id_null_values(self):
        connection, cursor = self.make_mock_database()

        row = list(self.sample_row)
        row[7] = None
        row[8] = None
        row[9] = None

        cursor.fetchone.return_value = tuple(row)

        with patch(
            "main.get_db_connection",
            return_value=connection
        ):
            review = main.get_review_by_id(1)

        self.assertIsNone(
            review["average_changes_per_file"]
        )
        self.assertIsNone(review["deletion_ratio"])
        self.assertIsNone(review["addition_ratio"])

    # TEST 21: ML risk prediction.
    def test_predict_risk_with_ml(self):
        features = {
            "total_changes": 100,
            "files_changed": 5,
            "average_changes_per_file": 20,
            "addition_ratio": 0.8,
            "deletion_ratio": 0.2
        }

        with patch("main.ml_model") as mock_model:
            mock_model.classes_ = ["low", "medium", "high"]
            mock_model.predict.return_value = ["high"]
            mock_model.predict_proba.return_value = [
                [0.02, 0.03, 0.95]
            ]

            risk_level, risk_score = (
                main.predict_risk_with_ml(features)
            )

            self.assertEqual(risk_level, "high")
            self.assertAlmostEqual(risk_score, 0.95)

            expected_input = [[100, 5, 20, 0.8, 0.2]]

            mock_model.predict.assert_called_once_with(
                expected_input
            )
            mock_model.predict_proba.assert_called_once_with(
                expected_input
            )


if __name__ == "__main__":
    unittest.main()
