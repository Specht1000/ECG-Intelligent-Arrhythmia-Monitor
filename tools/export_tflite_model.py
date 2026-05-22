from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "heartbeat_cnn.keras"
OUT_PATH = PROJECT_ROOT / "models" / "exported" / "heartbeat_cnn.tflite"

model = tf.keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_PATH, "wb") as f:
    f.write(tflite_model)

print("Modelo exportado:")
print(OUT_PATH)