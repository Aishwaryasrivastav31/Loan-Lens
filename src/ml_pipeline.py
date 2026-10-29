import os
import joblib
import pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)

TRAIN_FILE = "data/loan-train.csv"
MODEL_FILE = (
    "models/loan_model.joblib"  # new name, so an old saved model is never picked up
)

CAT_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
]
NUM_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

# 1. load and clean
df = pd.read_csv(TRAIN_FILE)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
df["Credit_History"] = df["Credit_History"].map(
    {1: "1", 0: "0"}
)  # category, not a number

fill = {}  # values used to fill missing data
for col in CAT_COLS:
    df[col] = df[col].astype("string")
    fill[col] = df[col].mode()[0]
for col in NUM_COLS:
    fill[col] = float(df[col].median())
df[CAT_COLS] = df[CAT_COLS].fillna(fill).astype(str)
df[NUM_COLS] = df[NUM_COLS].fillna(fill)

print("Approved / Rejected:", df["Loan_Status"].value_counts().to_dict())

X = df[CAT_COLS + NUM_COLS]
y = df["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. models - preprocessing is INSIDE the pipeline, so the saved model takes raw applicant data
prep = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ]
)
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    ),
}

# 3. train, score, track in MLflow
mlflow.set_experiment("loan-approval")
results = []
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("prep", prep), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        scores = {
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "rejected_caught": recall_score(y_test, pred, pos_label=0),
            "pct_approved": float(pred.mean()),
        }
        mlflow.log_param("model", name)
        mlflow.log_metrics(scores)
        results.append({"model": name, **scores})
        print(name, {k: round(v, 3) for k, v in scores.items()})

# 4. best model = highest balanced accuracy -> retrain on all data and save
table = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)
print(table.round(3).to_string(index=False))
best_name = table.iloc[0]["model"]
print("BEST:", best_name)

best = Pipeline([("prep", prep), ("model", models[best_name])]).fit(X, y)
os.makedirs("models", exist_ok=True)
joblib.dump(
    {
        "model": best,
        "name": best_name,
        "fill": fill,
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "train_data": df[CAT_COLS + NUM_COLS + ["Loan_Status"]],
    },
    MODEL_FILE,
)
print("saved ->", MODEL_FILE)
