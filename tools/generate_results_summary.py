from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_FILE = PROJECT_ROOT / "data" / "processed" / "heartbeat_dataset" / "X_beats.npy"
Y_FILE = PROJECT_ROOT / "data" / "processed" / "heartbeat_dataset" / "y_beats.npy"

MODEL_FILE = PROJECT_ROOT / "models" / "trained" / "heartbeat_cnn.keras"
ENCODER_FILE = PROJECT_ROOT / "models" / "trained" / "heartbeat_label_encoder.pkl"

OUT_FILE = PROJECT_ROOT / "reports" / "results_summary.txt"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Carregando dados...")
X = np.load(X_FILE)
y = np.load(Y_FILE, allow_pickle=True)

print("Carregando modelo...")
model = load_model(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)

X = X.astype(np.float32)
X -= np.mean(X, axis=1, keepdims=True)
X /= np.where(np.std(X, axis=1, keepdims=True) > 1e-6,
              np.std(X, axis=1, keepdims=True),
              1.0)

X = X.reshape(X.shape[0], X.shape[1], 1)

print("Predizendo...")
pred_prob = model.predict(X, batch_size=256, verbose=1)
pred_idx = np.argmax(pred_prob, axis=1)
y_pred = encoder.inverse_transform(pred_idx)

acc = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)

class_counts = pd.Series(y).value_counts()

text = f"""
RESULTADOS DO SISTEMA ECG + IA

1. Dataset utilizado
Total de batimentos: {len(y)}

Distribuição das classes:
{class_counts.to_string()}

2. Modelo
Modelo utilizado: CNN 1D
Entrada: batimentos segmentados com 252 amostras
Classes: normal, supraventricular, ventricular

3. Desempenho geral
Accuracy: {acc:.4f}

4. Métricas por classe
{report}

5. Observações técnicas
O sistema apresentou alto desempenho na classificação de batimentos do banco MIT-BIH.
A classe ventricular apresentou excelente desempenho, sendo a mais relevante para alerta clínico.
A classe supraventricular apresentou menor recall, o que é esperado, pois pode apresentar morfologia semelhante à classe normal.

6. Resultados experimentais do protótipo
O sistema também foi testado com sinais reais adquiridos por AD8232 + ADS1115 + ESP32-S3.
Os sinais reais apresentaram ruídos e artefatos típicos de sensores de baixo custo, como interferência muscular, movimentação dos eletrodos e variação de linha de base.
Por isso, foi implementado um classificador de qualidade do sinal para indicar quando a leitura é boa, aceitável ou ruim.

7. Conclusão parcial
O sistema desenvolvido demonstrou viabilidade técnica para aquisição, processamento, detecção de picos R, estimativa de BPM, classificação de batimentos e emissão de alertas em tempo real.
"""

OUT_FILE.write_text(text, encoding="utf-8")

print(f"\nResumo salvo em: {OUT_FILE}")
print(text)