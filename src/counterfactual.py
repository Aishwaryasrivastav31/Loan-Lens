import os
import joblib
import pandas as pd
import dice_ml

# path from this file's folder (src/ -> project root), so it works from any folder (streamlit too)
MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "loan_model.joblib"
)
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"{os.path.abspath(MODEL_FILE)} not found -> first run:  python src/ml_pipeline.py"
    )
saved = joblib.load(MODEL_FILE)
model, fill = saved["model"], saved["fill"]
cat_cols, num_cols = saved["cat_cols"], saved["num_cols"]
train = saved["train_data"]

# DiCE setup (once)
data = dice_ml.Data(
    dataframe=train, continuous_features=num_cols, outcome_name="Loan_Status"
)
dice = dice_ml.Dice(
    data, dice_ml.Model(model=model, backend="sklearn"), method="random"
)


def check_applicant(applicant: dict):
    """Approved -> probability only.  Rejected -> up to 3 changes that would get approval."""
    row = pd.DataFrame([applicant])[cat_cols + num_cols]
    if "Credit_History" in applicant and pd.notna(applicant["Credit_History"]):
        row["Credit_History"] = str(int(float(applicant["Credit_History"])))
    row = row.fillna(fill)
    row[cat_cols] = row[cat_cols].astype(str)
    row[num_cols] = row[num_cols].astype(float)

    prob = float(model.predict_proba(row)[0][1])
    if model.predict(row)[0] == 1:
        return {"status": "approved", "approval_probability": round(prob, 3)}

    # only things an applicant can change, and only in a sensible direction
    income, co_income, loan = (
        row.iloc[0]["ApplicantIncome"],
        row.iloc[0]["CoapplicantIncome"],
        row.iloc[0]["LoanAmount"],
    )
    try:
        cf = dice.generate_counterfactuals(
            row,
            total_CFs=3,
            desired_class=1,
            features_to_vary=["ApplicantIncome", "CoapplicantIncome", "LoanAmount"],
            permitted_range={
                "ApplicantIncome": [
                    income,
                    max(income, train["ApplicantIncome"].max()),
                ],
                "CoapplicantIncome": [
                    co_income,
                    max(co_income, train["CoapplicantIncome"].max()),
                ],
                "LoanAmount": [min(loan, train["LoanAmount"].min()), loan],
            },
        )
        cf_df = cf.cf_examples_list[0].final_cfs_df
    except Exception as e:
        return {
            "status": "rejected",
            "approval_probability": round(prob, 3),
            "counterfactuals": [],
            "note": f"no change in income / loan amount gets approval ({e})",
        }

    changes = []
    for _, r in cf_df.iterrows():
        diff = {
            c: {"from": round(float(row.iloc[0][c])), "to": round(float(r[c]))}
            for c in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
            if float(r[c]) != float(row.iloc[0][c])
        }
        if diff:
            changes.append(diff)
    return {
        "status": "rejected",
        "approval_probability": round(prob, 3),
        "counterfactuals": changes,
    }


if __name__ == "__main__":
    # demo: one approved applicant, and one rejected applicant with a good credit history
    preds = model.predict(train[cat_cols + num_cols])
    approved = train[preds == 1]
    rejected = train[(preds == 0) & (train["Credit_History"] == "1")]
    if rejected.empty:
        rejected = train[preds == 0]
    for applicant in (approved.iloc[0], rejected.iloc[0]):
        applicant = applicant[cat_cols + num_cols].to_dict()
        print("\nApplicant:", applicant)
        print("Result:", check_applicant(applicant))
