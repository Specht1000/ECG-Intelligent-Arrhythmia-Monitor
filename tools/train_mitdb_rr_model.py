from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "mitdb_feature_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "trained"

MODEL_FILE = MODEL_DIR / "random_forest_mitdb.pkl"
SCALER_FILE = MODEL_DIR / "scaler_mitdb.pkl"
ENCODER_FILE = MODEL_DIR / "label_encoder_mitdb.pkl"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    df = df[df["label"].isin(["normal", "supraventricular", "ventricular"])].copy()

    target_col = "label"
    groups = df["record_name"].astype(str)

    drop_cols = [
        "label",
        "record_name",
        "signal_column",
        "start_sec",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = df[target_col]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y_enc[train_idx]
    y_test = y_enc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, target_names=le.classes_))
    print(confusion_matrix(y_test, pred))

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, ENCODER_FILE)

    print(f"Modelo salvo em: {MODEL_FILE}")
    print(f"Scaler salvo em: {SCALER_FILE}")
    print(f"Encoder salvo em: {ENCODER_FILE}")


if __name__ == "__main__":
    main()