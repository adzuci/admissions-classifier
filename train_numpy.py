"""Train with pure NumPy (no TensorFlow). Use when TensorFlow hangs on Mac."""
import joblib
import numpy as np
import pandas as pd

from scaler import SimpleScaler

print("Loading data...")
df = pd.read_csv("student_admission_dataset.csv")
df["Admit"] = (df["Admission_Status"] == "Accepted").astype(int)
X = df[["GPA", "SAT_Score", "Extracurricular_Activities"]].values.astype(np.float64)
y = df["Admit"].values.reshape(-1, 1)

np.random.seed(42)
idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

print("Scaling features...")
scaler = SimpleScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Add bias column
X_train_s = np.hstack([np.ones((len(X_train_s), 1)), X_train_s])
X_test_s = np.hstack([np.ones((len(X_test_s), 1)), X_test_s])

print("Training logistic regression (NumPy, 500 steps)...")
n_samples, n_features = X_train_s.shape
w = np.zeros((n_features, 1))
lr = 0.1

for step in range(500):
    logits = X_train_s @ w
    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
    grad = X_train_s.T @ (probs - y_train) / n_samples
    w -= lr * grad
    if step % 100 == 0:
        preds = (probs > 0.5).astype(float)
        acc = (preds == y_train).mean()
        print(f"  Step {step}: train acc {acc:.4f}")

probs_test = 1 / (1 + np.exp(-np.clip(X_test_s @ w, -500, 500)))
preds_test = (probs_test > 0.5).astype(float)
acc_test = (preds_test == y_test).mean()
print(f"Test accuracy: {acc_test:.4f}")

model = {"weights": w, "type": "numpy_lr"}
joblib.dump(model, "model_numpy.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Saved model_numpy.joblib and scaler.joblib")
print("Run: python admissions_review.py  (use model_numpy.joblib when prompted)")
