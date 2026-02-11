#!/usr/bin/env python3
"""Admissions classifier: logistic regression. Train on CSV or review applications interactively."""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = ["GPA", "SAT_Score", "Extracurricular_Activities"]


class SimpleScaler:
    """Minimal scaler with fit/transform (no sklearn)."""

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_


def train(csv_path: str) -> None:
    """Train logistic regression on CSV and save model.joblib + scaler.joblib."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df["Admit"] = (df["Admission_Status"] == "Accepted").astype(int)
    X = df[["GPA", "SAT_Score", "Extracurricular_Activities"]].values.astype(np.float64)
    y = df["Admit"].values

    # Stratified train/test split
    np.random.seed(42)
    idx_1 = np.where(y == 1)[0]
    idx_0 = np.where(y == 0)[0]
    np.random.shuffle(idx_1)
    np.random.shuffle(idx_0)
    n_test_1 = max(1, int(len(idx_1) * 0.2))
    n_test_0 = max(1, int(len(idx_0) * 0.2))
    test_idx = np.concatenate([idx_1[:n_test_1], idx_0[:n_test_0]])
    train_idx = np.concatenate([idx_1[n_test_1:], idx_0[n_test_0:]])
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = SimpleScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Training logistic regression...")
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    model.fit(X_train_s, y_train)

    acc = (model.predict(X_test_s) == y_test).mean()
    print(f"Test accuracy: {acc:.4f}")

    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("Saved model.joblib and scaler.joblib")


def review() -> None:
    """Interactive review: prompt for GPA, SAT, extracurriculars; give verdict; ask for another."""
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")

    while True:
        print("\n--- New application ---")
        values = []
        for name in FEATURE_NAMES:
            val = input(f"  {name}: ").strip()
            values.append(float(val))
        X = scaler.transform([values])
        pred = model.predict(X)[0]

        verdict = "ACCEPTED" if pred == 1 else "REJECTED"
        print(f"\n  Verdict: {verdict}\n")

        again = input("Review another application? (y/n): ").strip().lower()
        if again != "y":
            print("Done.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Admissions classifier (logistic regression): train on CSV or review applications"
    )
    parser.add_argument(
        "--train",
        nargs="?",
        const="student_admission_dataset.csv",
        default=None,
        metavar="CSV",
        help="Train on CSV (default: student_admission_dataset.csv)",
    )
    args = parser.parse_args()

    if args.train:
        train(args.train)
    else:
        review()


if __name__ == "__main__":
    main()
