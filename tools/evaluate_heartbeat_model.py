from pathlib import Path

import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_FILE = PROJECT_ROOT / "data" / "processed" / "heartbeat_dataset" / "X_beats.npy"
Y_FILE = PROJECT_ROOT / "data" / "processed" / "heartbeat_dataset" / "y_beats.npy"

MODEL_FILE = PROJECT_ROOT / "models" / "trained" / "heartbeat_cnn.keras"
ENCODER_FILE = PROJECT_ROOT / "models" / "trained" / "heartbeat_label_encoder.pkl"

OUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Carregando dataset...")
X = np.load(X_FILE)
y = np.load(Y_FILE, allow_pickle=True)

print("Carregando modelo...")
model = load_model(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)

print("Normalizando entrada...")
X = X.astype(np.float32)

means = np.mean(X, axis=1, keepdims=True)
stds = np.std(X, axis=1, keepdims=True)

X = X - means
X = X / np.where(stds > 1e-6, stds, 1.0)

X = X.reshape(X.shape[0], X.shape[1], 1)

print("Realizando predição...")
pred_prob = model.predict(X, batch_size=256, verbose=1)

y_pred_idx = np.argmax(pred_prob, axis=1)
y_pred = encoder.inverse_transform(y_pred_idx)

acc = accuracy_score(y, y_pred)

print("\n==============================")
print("AVALIAÇÃO FINAL CNN")
print("==============================")
print(f"Accuracy: {acc:.4f}")

print("\nClassification report:")
print(classification_report(y, y_pred))

labels = list(encoder.classes_)

cm = confusion_matrix(y, y_pred, labels=labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)

plt.title("Matriz de confusão - CNN MIT-BIH")
plt.tight_layout()

out_file = OUT_DIR / "confusion_matrix_heartbeat_cnn.png"
plt.savefig(out_file, dpi=300)

print(f"\nMatriz de confusão salva em: {out_file}")

plt.show()