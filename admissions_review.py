#!/usr/bin/env python3
"""Interactive script to review applications using a trained model."""
import pickle

from tensorflow.keras.models import load_model

FEATURE_NAMES = ["GPA", "SAT_Score", "Extracurricular_Activities"]


def prompt_application(feature_names):
    """Prompt for each feature, return list of values."""
    values = []
    for name in feature_names:
        val = input(f"  {name}: ").strip()
        values.append(float(val))
    return values


def main():
    model_path = input("Model path [model.keras]: ").strip() or "model.keras"
    scaler_path = input("Scaler path [scaler.pkl]: ").strip() or "scaler.pkl"

    model = load_model(model_path)
    scaler = pickle.load(open(scaler_path, "rb"))

    while True:
        print("\n--- New application ---")
        values = prompt_application(FEATURE_NAMES)
        X = scaler.transform([values])
        pred = (model.predict(X, verbose=0) > 0.5).astype(int)[0][0]
        result = "ACCEPTED" if pred == 1 else "REJECTED"
        print(f"\n  Prediction: {result}\n")

        again = input("Another application to review? (y/n): ").strip().lower()
        if again != "y":
            print("Done.")
            break


if __name__ == "__main__":
    main()
