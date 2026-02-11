"""Train admissions classifier and save model + scaler."""
import os

# Force CPU - Metal/GPU on Mac can hang; must set before any TF import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
import pandas as pd

print("Importing TensorFlow (CPU mode, may take 10-30s on first run)...")
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from tensorflow import keras
from tensorflow.keras import layers

from scaler import SimpleScaler

print("Loading data...")
df = pd.read_csv("student_admission_dataset.csv")
df["Admit"] = (df["Admission_Status"] == "Accepted").astype(int)
X = df[["GPA", "SAT_Score", "Extracurricular_Activities"]].values.astype(np.float32)
y = df["Admit"].values

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

print("Building model...")
model = keras.Sequential([
    keras.Input(shape=(3,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("Training (10 epochs, ~200 samples)...")
model.fit(X_train_s, y_train, epochs=10, batch_size=32, verbose=1)

loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

print("Saving model and scaler...")
model.save("model.keras")
joblib.dump(scaler, "scaler.joblib")
print("Saved model.keras and scaler.joblib")
