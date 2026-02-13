#!/usr/bin/env python3
"""Admissions classifier: logistic regression for undergraduate applications. Train on CSV or review applications interactively."""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = ["GPA", "SAT_Score", "Extracurricular_Activities"]

# Input validation bounds (per README roadmap)
VALIDATION = {
    "GPA": (2.0, 4.0),
    "SAT_Score": (400, 1600),
    "Extracurricular_Activities": (0, 20),
}


def prompt_feature(name: str) -> float:
    """Prompt for a feature value until valid. Returns float in range."""
    lo, hi = VALIDATION[name]
    while True:
        raw = input(f"  {name}: ").strip()
        try:
            val = float(raw)
        except ValueError:
            print(f"  Invalid: enter a number (e.g. {lo}–{hi})")
            continue
        if lo <= val <= hi:
            return val
        print(f"  Out of range: {name} must be {lo}–{hi}")


class SimpleScaler:
    """Z-score scaler: fit learns mean/std from training data; transform standardizes inputs."""

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_


def train(csv_path: str) -> None:
    """Train logistic regression on CSV. Saves model.joblib and scaler.joblib."""
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
    """Interactive review: prompt for GPA, SAT, extracurriculars; print verdict; offer to review another."""
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")

    while True:
        print("\n--- New application ---")
        values = [prompt_feature(name) for name in FEATURE_NAMES]
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
        description="Admissions classifier (logistic regression) for undergraduate applications: train on CSV or review interactively"
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
