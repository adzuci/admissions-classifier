#!/usr/bin/env python3
"""Interactive script to review applications using a trained model."""
import joblib
import numpy as np

FEATURE_NAMES = ["GPA", "SAT_Score", "Extracurricular_Activities"]


def prompt_application(feature_names):
    """Prompt for each feature, return list of values."""
    values = []
    for name in feature_names:
        val = input(f"  {name}: ").strip()
        values.append(float(val))
    return values


def predict_numpy(model, X_scaled):
    """Predict using NumPy logistic regression model."""
    X_b = np.hstack([np.ones((len(X_scaled), 1)), X_scaled])
    w = model["weights"]
    logits = X_b @ w
    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
    return (probs > 0.5).astype(int).flatten()


def main():
    model_path = input("Model path [model.keras]: ").strip() or "model.keras"
    scaler_path = input("Scaler path [scaler.joblib]: ").strip() or "scaler.joblib"

    scaler = joblib.load(scaler_path)

    if model_path.endswith("model_numpy.joblib"):
        model = joblib.load(model_path)
        use_numpy = True
    elif model_path.endswith(".joblib"):
        model = joblib.load(model_path)
        use_numpy = model.get("type") == "numpy_lr"
    else:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        use_numpy = False

    while True:
        print("\n--- New application ---")
        values = prompt_application(FEATURE_NAMES)
        X = scaler.transform([values])

        if use_numpy:
            pred = predict_numpy(model, X)[0]
        else:
            pred = (model.predict(X, verbose=0) > 0.5).astype(int)[0][0]

        result = "ACCEPTED" if pred == 1 else "REJECTED"
        print(f"\n  Prediction: {result}\n")

        again = input("Another application to review? (y/n): ").strip().lower()
        if again != "y":
            print("Done.")
            break


if __name__ == "__main__":
    main()
