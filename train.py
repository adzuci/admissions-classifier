"""Train admissions classifier and save model + scaler."""
import pickle

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from scaler import SimpleScaler

df = pd.read_csv("student_admission_dataset.csv")
df["Admit"] = (df["Admission_Status"] == "Accepted").astype(int)
X = df[["GPA", "SAT_Score", "Extracurricular_Activities"]].values.astype(np.float32)
y = df["Admit"].values

np.random.seed(42)
idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

scaler = SimpleScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = keras.Sequential([
    keras.Input(shape=(3,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train_s, y_train, epochs=50, batch_size=32, verbose=0)

loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

model.save("model.keras")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Saved model.keras and scaler.pkl")
