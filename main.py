from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from llm_reviewer import review_code
import psycopg
import joblib  # Loads our already-trained machine-learning model from disk.
# Imports our Phase 5 LangGraph workflow execution function.
from test_langgraph import run_review_safely
import os
# Converts Python lists and dictionaries into PostgreSQL JSONB values.
from psycopg.types.json import Jsonb

# Loads the trained Decision Tree once when the FastAPI application starts.
ml_model = joblib.load("risk_model.joblib")


# Defines the exact feature order expected by the trained ML model.
MODEL_FEATURES = [
    "total_changes",
    "files_changed",
    "average_changes_per_file",
    "addition_ratio",
    "deletion_ratio"
]


# Opens a connection from Python to PostgreSQL.
def get_db_connection():
    return psycopg.connect(
        host="localhost",
        port=5432,
        dbname="code_review",
        user="postgres",
        password=os.environ["DB_PASSWORD"]
    )


# Saves one completed PR analysis into PostgreSQL.
def save_review(
    repository,
    pr_number,
    total_changes,
    files_changed,
    average_changes_per_file,
    deletion_ratio,
    addition_ratio,
    risk_level,
    risk_score,
    issue_type,
    severity,
    problem,
    recommendation
):
    # Opens a connection to PostgreSQL.
    connection = get_db_connection()

    # Creates a cursor so Python can execute SQL commands.
    cursor = connection.cursor()

    # Inserts the PR features, ML prediction, and LLM review
    # into one row in the reviews table.
    cursor.execute(
        """
        INSERT INTO reviews (
            repository,
            pr_number,
            total_changes,
            files_changed,
            average_changes_per_file,
            deletion_ratio,
            addition_ratio,
            risk_level,
            risk_score,
            issue_type,
            severity,
            problem,
            recommendation
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            repository,
            pr_number,
            total_changes,
            files_changed,
            average_changes_per_file,
            deletion_ratio,
            addition_ratio,
            risk_level,
            risk_score,
            issue_type,
            severity,
            problem,
            recommendation
        )
    )

    # Permanently saves the INSERT operation in PostgreSQL.
    connection.commit()

    # Closes the cursor after the SQL work is finished.
    cursor.close()

    # Closes the PostgreSQL connection.
    connection.close()



# Saves a completed LangGraph code review into PostgreSQL.
# Saves a completed LangGraph code review into PostgreSQL.
def save_agent_review(
    repository,
    pr_number,
    result
):
    # Opens a connection to PostgreSQL.
    connection = get_db_connection()

    # Creates a cursor to execute SQL commands.
    cursor = connection.cursor()

    # Defines the SQL statement for saving LangGraph results.
    insert_sql = """
        INSERT INTO reviews (
            repository,
            pr_number,
            total_changes,
            files_changed,
            average_changes_per_file,
            deletion_ratio,
            addition_ratio,
            risk_level,
            risk_score,
            ast_findings,
            security_review,
            quality_review,
            performance_review,
            llm_review,
            final_response
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
    """

    # Prepares the 15 values in the same order as the SQL columns.
    review_values = (
        repository,
        pr_number,
        result["total_changes"],
        result["files_changed"],
        result["average_changes_per_file"],
        result["deletion_ratio"],
        result["addition_ratio"],
        result["risk_level"],
        float(result["risk_score"]),
        Jsonb(result.get("ast_findings", [])),
        result.get("security_review"),
        result.get("quality_review"),
        result.get("performance_review"),
        result.get("llm_review"),
        result.get("final_response")
    )

        # Executes the INSERT statement using our 15 prepared values.
    cursor.execute(insert_sql, review_values)

    # Permanently saves the new review record in PostgreSQL.
    connection.commit()

    # Closes the cursor.
    cursor.close()

    # Closes the database connection.
    connection.close()


# Gets every saved review from PostgreSQL.
def get_all_reviews():
    # Opens the database connection.
    connection = get_db_connection()

    # Creates a cursor so we can execute SQL.
    cursor = connection.cursor()

    # Selects all currently stored PR/ML information.
    cursor.execute(
    """
    SELECT
        id,
        repository,
        pr_number,
        total_changes,
        files_changed,
        risk_level,
        risk_score,
        average_changes_per_file,
        deletion_ratio,
        addition_ratio,
        issue_type,
        severity,
        problem,
        recommendation
    FROM reviews
    ORDER BY id;
    """
)

    # Gets all matching rows returned by PostgreSQL.
    rows = cursor.fetchall()

    # Creates an empty list where all review dictionaries will be stored.
    reviews = []

    # Goes through every PostgreSQL row one at a time.
    for row in rows:

        # Converts one PostgreSQL row into a Python dictionary.
        review = {
            "id": row[0],
            "repository": row[1],
            "pr_number": row[2],
            "total_changes": row[3],
            "files_changed": row[4],
            "risk_level": row[5],
            "risk_score": float(row[6]),

            "average_changes_per_file": (
                float(row[7]) if row[7] is not None else None
            ),

            "deletion_ratio": (
                float(row[8]) if row[8] is not None else None
            ),

            "addition_ratio": (
                float(row[9]) if row[9] is not None else None
            ),

            "issue_type": row[10],
            "severity": row[11],
            "problem": row[12],
            "recommendation": row[13]
            
        }

        # Adds this review dictionary to the reviews list.
        reviews.append(review)

    # Closes the cursor.
    cursor.close()

    # Closes the PostgreSQL connection.
    connection.close()

    # Returns the completed list of reviews.
    return reviews





# Gets one specific review from PostgreSQL using its ID.
def get_review_by_id(review_id):

    # Opens the database connection.
    connection = get_db_connection()

    # Creates a cursor so we can execute SQL.
    cursor = connection.cursor()

    # Searches for the review whose id matches review_id.
    cursor.execute(
        """
SELECT
    id,
    repository,
    pr_number,
    total_changes,
    files_changed,
    risk_level,
    risk_score,
    average_changes_per_file,
    deletion_ratio,
    addition_ratio,
    issue_type,
    severity,
    problem,
    recommendation
FROM reviews
WHERE id = %s;
        """,
        (review_id,)
    )

    # Gets one matching row from PostgreSQL.
    row = cursor.fetchone()

    # Closes database resources.
    cursor.close()
    connection.close()

    # If PostgreSQL found nothing, return None.
    if row is None:
        return None

    # Converts the PostgreSQL row into a Python dictionary.
    review = {
        "id": row[0],
        "repository": row[1],
        "pr_number": row[2],
        "total_changes": row[3],
        "files_changed": row[4],
        "risk_level": row[5],
        "risk_score": float(row[6]),

        "average_changes_per_file": (
            float(row[7]) if row[7] is not None else None
        ),

        "deletion_ratio": (
            float(row[8]) if row[8] is not None else None
        ),

        "addition_ratio": (
            float(row[9]) if row[9] is not None else None
        ),

        # Adds the saved LLM review information to the single-review response.
       "issue_type": row[10],
       "severity": row[11],
       "problem": row[12],
       "recommendation": row[13]
    }

    # Sends the completed dictionary back.
    return review


# Defines and validates the JSON expected by POST /analyze.
class PullRequestData(BaseModel):

    # Requires the repository name to contain at least one character.
    repository: str = Field(min_length=1)

    # gt=0 means the PR number must be greater than 0.
    pr_number: int = Field(gt=0)

    # ge=0 means these values must be greater than or equal to 0.
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    files_changed: int = Field(ge=0)

    # Requires actual source code for the LLM to review.
    code: str = Field(min_length=1)


# FEATURE ENGINEERING: converts raw PR data into measurable features.
def extract_pr_features(data: PullRequestData):

    # Total number of changed lines in the PR.
    total_changes = data.additions + data.deletions

    # Average number of changed lines per file.
    if data.files_changed > 0:
        average_changes_per_file = total_changes / data.files_changed
    else:
        average_changes_per_file = 0

    # Calculates what fraction of the changes are additions vs deletions.
    if total_changes > 0:
        addition_ratio = data.additions / total_changes
        deletion_ratio = data.deletions / total_changes
    else:
        addition_ratio = 0
        deletion_ratio = 0

    # Returns all engineered features.
    return {
        "total_changes": total_changes,
        "files_changed": data.files_changed,
        "average_changes_per_file": average_changes_per_file,
        "addition_ratio": addition_ratio,
        "deletion_ratio": deletion_ratio
    }


# Uses the trained machine-learning model to predict PR risk.
def predict_risk_with_ml(features):

    # Builds the model input using the shared feature order.
    # Puts the PR features in the exact order used during model training.
    model_input = [[
        features[feature_name]
        for feature_name in MODEL_FEATURES
    ]]

    # Predicts the risk category: low, medium, or high.
    prediction = ml_model.predict(model_input)

    # Gets the probability for every possible risk category.
    probabilities = ml_model.predict_proba(model_input)

    # Gets the predicted risk category from the prediction array.
    risk_level = prediction[0]

    # Finds the position of the predicted class inside model.classes_.
    class_index = list(ml_model.classes_).index(risk_level)

    # Gets the probability belonging to the predicted risk category.
    risk_probability = probabilities[0][class_index]

    # Returns TWO values: predicted class and its probability.
    return risk_level, risk_probability



# Converts incoming FastAPI pull request data into LangGraph state.
def create_agent_state(data: PullRequestData):

    return {
        "user_request": f"Review pull request #{data.pr_number} from {data.repository}",
        "code": data.code,

        "additions": data.additions,
        "deletions": data.deletions,
        "files_changed": data.files_changed,

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

        "final_response": None
    }


# Creates the FastAPI application.
app = FastAPI()


# Health-check endpoint.
@app.get("/")
def read_root():
    return {
        "message": "AI Code Review Agent is running"
    }


# Tests whether FastAPI can connect to PostgreSQL.
@app.get("/db-test")
def test_database_connection():

    # Opens a PostgreSQL connection.
    connection = get_db_connection()

    # Closes it immediately because this endpoint only tests the connection.
    connection.close()

    # Returns a success message if no database error occurred.
    return {
        "message": "Database connection successful"
    }


# Returns all saved reviews.
@app.get("/reviews")
def read_reviews():

    # Gets all reviews from PostgreSQL.
    reviews = get_all_reviews()

    # Returns the reviews as the API response.
    return reviews


# Returns one review using its ID.
@app.get(
    "/reviews/{review_id}",
    responses={
        404: {"description": "Review not found"}
    }
)
def read_review(review_id: int):

    # Searches PostgreSQL for the requested review.
    review = get_review_by_id(review_id)

    # If the review does not exist, return HTTP 404.
    if review is None:
        raise HTTPException(
            status_code=404,
            detail="Review not found"
        )

    # Returns the matching review.
    return review


# Main PR-analysis endpoint.
@app.post("/analyze")
async def analyze_pull_request(data: PullRequestData):

        # Converts the incoming FastAPI request into LangGraph state.
    state = create_agent_state(data)

    # Runs the Phase 5 LangGraph workflow asynchronously.
    workflow_result = await run_review_safely(state)

    # Stops the request if LangGraph fails.
    if workflow_result["status"] != "success":
        raise HTTPException(
            status_code=503,
            detail=workflow_result["message"]
        )

    # Gets the completed state returned by LangGraph.
    result = workflow_result["result"]

    
    # Saves the completed LangGraph review into PostgreSQL.
    save_agent_review(
        repository=data.repository,
        pr_number=data.pr_number,
        result=result
    )

    # Returns the completed AI review to the API caller.
    return result
