from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import joblib  # Library used to save and load our trained machine-learning model.


# Creates a Decision Tree classifier object.
model = DecisionTreeClassifier(
    max_depth=3,      # Limits the tree to 3 levels to help prevent overfitting.
    random_state=42   # Makes the model's randomized behavior reproducible.
)


# FEATURES:
# These are the input values the model uses to learn.
# Order:
# [total_changes, files_changed, average_changes_per_file, addition_ratio, deletion_ratio]
X = [

    # Low-risk examples
    [15, 1, 15, 0.80, 0.20],
    [25, 2, 12.5, 0.90, 0.10],
    [30, 2, 15, 0.70, 0.30],
    [35, 3, 11.67, 0.85, 0.15],
    [40, 2, 20, 0.75, 0.25],
    [45, 3, 15, 0.80, 0.20],

    # Medium-risk examples
    [70, 4, 17.5, 0.75, 0.25],
    [80, 5, 16, 0.80, 0.20],
    [100, 5, 20, 0.70, 0.30],
    [120, 6, 20, 0.85, 0.15],
    [140, 7, 20, 0.65, 0.35],
    [160, 8, 20, 0.90, 0.10],

    # High-risk examples
    [200, 10, 20, 0.80, 0.20],
    [220, 11, 20, 0.75, 0.25],
    [250, 12, 20.83, 0.90, 0.10],
    [280, 14, 20, 0.70, 0.30],
    [300, 15, 20, 0.85, 0.15],
    [400, 20, 20, 0.80, 0.20]
]


# LABELS:
# These are the correct answers the model learns from.
# Every label matches the feature row at the same position in X.
y = [
    "low", "low", "low", "low", "low", "low",
    "medium", "medium", "medium", "medium", "medium", "medium",
    "high", "high", "high", "high", "high", "high"
]


# Splits the complete dataset into training data and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,   # Uses about 33% of the data for testing.
    random_state=42,  # Makes the same split happen each time we run the program.
    stratify=y        # Tries to preserve the LOW/MEDIUM/HIGH class proportions.
)


# fit() TRAINs the model using training features and their correct labels.
model.fit(X_train, y_train)


# dump() SAVES the trained model into a file.
# This lets us use the trained model later without training it again.
joblib.dump(
    model,
    "risk_model.joblib"
)


# predict() asks the trained model to predict the test examples.
predictions = model.predict(X_test)


# Compares the predictions with the real answers and calculates overall accuracy.
accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)


# NEW PR:
# This represents one new pull request that we want the trained model to classify.
# Feature order:
# [total_changes, files_changed, average_changes_per_file, addition_ratio, deletion_ratio]
new_pr = [
    [100, 5, 20, 0.80, 0.20]
]


# predict() gives the final predicted class: LOW, MEDIUM, or HIGH.
prediction = model.predict(new_pr)

print("Prediction:", prediction)


# predict_proba() returns the predicted probability for every possible class.
probabilities = model.predict_proba(new_pr)

print("Prediction probabilities:", probabilities)


# zip() pairs each class name with its corresponding probability.
# The for loop goes through those pairs one at a time.
for class_name, probability in zip(
    model.classes_,     # Names of the classes learned by the model.
    probabilities[0]    # Probabilities for our first new PR.
):
    print(class_name, ":", probability)


# classes_ is an ATTRIBUTE that shows the exact class order used by predict_proba().
print("Classes:", model.classes_)


# export_text() converts the trained Decision Tree into readable rules.
tree_rules = export_text(
    model,
    feature_names=[
        "total_changes",
        "files_changed",
        "average_changes_per_file",
        "addition_ratio",
        "deletion_ratio"
    ]
)

print(tree_rules)


# Displays what the model predicted for the test data.
print("Predictions:", predictions)

# Displays the actual correct answers so we can compare them.
print("Actual:", y_test)


# feature_importances_ shows how much the Decision Tree used each feature.
print(
    "Feature importance:",
    model.feature_importances_
)


# confusion_matrix() shows which classes were predicted correctly
# and which classes were confused with another class.
matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[
        "low",
        "medium",
        "high"
    ]
)

print("Confusion Matrix:")
print(matrix)


# classification_report() calculates:
# precision, recall, F1-score, support, and overall accuracy.
print("Classification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[
            "low",
            "medium",
            "high"
        ]
    )
)


# Creates a Stratified K-Fold object for cross-validation.
# n_splits=3 means we evaluate the model using 3 different folds.
stratified_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,     # Shuffles the data before creating the folds.
    random_state=42   # Makes the shuffled folds reproducible.
)


# cross_val_score() trains/tests the model multiple times
# using the different folds created above.
scores = cross_val_score(
    model,
    X,
    y,
    cv=stratified_cv
)


# Shows the accuracy from each cross-validation fold.
print(
    "Cross-validation scores:",
    scores
)


# mean() calculates the average of all cross-validation accuracy scores.
print(
    "Average cross-validation accuracy:",
    scores.mean()
)