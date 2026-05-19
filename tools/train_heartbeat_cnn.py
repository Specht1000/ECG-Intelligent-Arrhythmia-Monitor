from pathlib import Path
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten
)
from tensorflow.keras.utils import to_categorical


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "heartbeat_dataset"

X_FILE = DATASET_DIR / "X_beats.npy"
Y_FILE = DATASET_DIR / "y_beats.npy"

MODEL_DIR = PROJECT_ROOT / "models" / "trained"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "heartbeat_cnn.keras"
ENCODER_FILE = MODEL_DIR / "heartbeat_label_encoder.pkl"


def build_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv1D(
        filters=32,
        kernel_size=5,
        activation="relu",
        input_shape=input_shape
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    print("Carregando dataset...")

    X = np.load(X_FILE)
    y = np.load(Y_FILE)

    print("Formato X:", X.shape)
    print("Formato y:", y.shape)

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    num_classes = len(encoder.classes_)

    print("Classes:", encoder.classes_)

    y_cat = to_categorical(y_enc)

    X = X[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_cat,
        test_size=0.2,
        random_state=42,
        stratify=y_enc
    )

    print("Treino:", X_train.shape)
    print("Teste:", X_test.shape)

    model = build_model(
        input_shape=X_train.shape[1:],
        num_classes=num_classes
    )

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=256
    )

    print("\nAvaliando modelo...")

    loss, acc = model.evaluate(X_test, y_test)

    print(f"Test accuracy: {acc:.4f}")

    pred = model.predict(X_test)

    pred_classes = np.argmax(pred, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    print("\nClassification report:")
    print(classification_report(
        true_classes,
        pred_classes,
        target_names=encoder.classes_
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(true_classes, pred_classes))

    model.save(MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    print(f"\nModelo salvo em: {MODEL_FILE}")
    print(f"Encoder salvo em: {ENCODER_FILE}")


if __name__ == "__main__":
    main()