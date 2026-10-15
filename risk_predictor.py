
from pathlib import Path
import joblib

# Locates the trained model relative to this Python file.
MODEL_PATH = Path(__file__).resolve().parent / "risk_model.joblib"

# Loads our already-trained Decision Tree model.
ml_model = joblib.load(MODEL_PATH)

# Keeps the five input features in their original training order.
MODEL_FEATURES = [
    "total_changes",
    "files_changed",
    "average_changes_per_file",
    "addition_ratio",
    "deletion_ratio"
]


# Predicts the risk level of a pull request using our ML model.
def predict_risk_with_ml(features):

    # Arranges input values in the order expected by the model.
    model_input = [[
        features[feature_name]
        for feature_name in MODEL_FEATURES
    ]]

    # Predicts low, medium, or high risk.
    prediction = ml_model.predict(model_input)

    # Calculates the probability for each risk category.
    probabilities = ml_model.predict_proba(model_input)

    # Gets the predicted risk category.
    risk_level = prediction[0]

    # Finds the index of the predicted category.
    class_index = list(ml_model.classes_).index(risk_level)

    # Gets the probability of that prediction.
    risk_probability = probabilities[0][class_index]

    # Returns the predicted risk and probability.
    return risk_level, risk_probability
