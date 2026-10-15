import joblib  # Library used to save and load trained machine-learning models.


# load() reads our previously saved trained model from risk_model.joblib.
# loaded_model is now an object containing the trained Decision Tree.
loaded_model = joblib.load("risk_model.joblib")


# Creates one new PR for the trained model to analyze.
# Feature order:
# [total_changes, files_changed, average_changes_per_file, addition_ratio, deletion_ratio]
new_pr = [
    [100, 5, 20, 0.80, 0.20]
]


# predict() uses the already-trained model to predict the risk level of the new PR.
# It does NOT train the model again.
prediction = loaded_model.predict(new_pr)


# Displays the predicted risk level: low, medium, or high.
print("Loaded model prediction:", prediction)