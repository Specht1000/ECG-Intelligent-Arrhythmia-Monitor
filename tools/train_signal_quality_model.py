from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_FILE = Path("data/processed/signal_quality_dataset.csv")
MODEL_FILE = Path("models/trained/signal_quality_model.pkl")

df = pd.read_csv(DATA_FILE)

features = [
    "std",
    "range",
    "diff_std",
    "energy",
    "zero_or_clip",
]

X = df[features]
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_FILE)

print(f"Modelo salvo em: {MODEL_FILE}")