from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
)

from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent.parent

X_PATH = BASE_DIR / "data" / "processed" / "cnn_X.npy"
Y_PATH = BASE_DIR / "data" / "processed" / "cnn_y.csv"

MODEL_PATH = BASE_DIR / "data" / "processed" / "cnn_arrhythmia_model_final.keras"

OUTPUT_DIR = BASE_DIR / "report_figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def loss_fn(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


def find_label_column(df):
    for col in ["arrhythmia_label", "label", "class", "target", "y"]:
        if col in df.columns:
            return col
    return df.columns[-1]


def normalize_input_shape(X):
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 2:
        X = X[..., np.newaxis]

    if X.ndim != 3:
        raise ValueError(f"Invalid CNN input shape: {X.shape}")

    return X


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_filename(value):
    return str(value).replace(" ", "_").replace("/", "_")


def build_manual_latex_table(metrics_df, class_names, accuracy):
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{CNN classification performance on the test set.}")
    lines.append("\\label{tab:cnn_performance}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Class & Precision & Recall & F1-score & Support \\\\")
    lines.append("\\midrule")

    for cls in class_names:
        row = metrics_df.loc[str(cls)]
        lines.append(
            f"{cls} & "
            f"{row['precision']:.3f} & "
            f"{row['recall']:.3f} & "
            f"{row['f1-score']:.3f} & "
            f"{int(row['support'])} \\\\"
        )

    lines.append("\\midrule")
    lines.append(f"Accuracy & \\multicolumn{{3}}{{c}}{{{accuracy:.4f}}} & -- \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def infer_best_label_mapping(y_true_encoded, y_pred_raw, class_names):
    n = len(class_names)
    best_acc = -1.0
    best_perm = None

    for perm in itertools.permutations(range(n)):
        mapped_pred = np.array([perm[p] for p in y_pred_raw])
        acc = accuracy_score(y_true_encoded, mapped_pred)

        if acc > best_acc:
            best_acc = acc
            best_perm = perm

    mapped_pred = np.array([best_perm[p] for p in y_pred_raw])

    mapping = {
        f"model_output_{i}": class_names[best_perm[i]]
        for i in range(n)
    }

    return mapped_pred, best_acc, mapping


def main():
    print("=" * 60)
    print("REAL CNN ANALYSIS")
    print("=" * 60)

    print("[INFO] Loading dataset...")

    X = np.load(X_PATH)
    y_df = pd.read_csv(Y_PATH)

    label_col = find_label_column(y_df)
    y = y_df[label_col].values.astype(str)

    print(f"[INFO] X shape before reshape: {X.shape}")
    print(f"[INFO] y shape: {y.shape}")
    print(f"[INFO] Label column: {label_col}")

    X = normalize_input_shape(X)

    print(f"[INFO] CNN input shape: {X.shape}")

    print("[INFO] Loading CNN model...")

    model = load_model(
        MODEL_PATH,
        custom_objects={"loss_fn": loss_fn},
        compile=False
    )

    print("[INFO] Model loaded successfully.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    class_names = sorted(np.unique(y))
    numeric_class_labels = list(range(len(class_names)))

    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    y_test_encoded = np.array([label_to_index[label] for label in y_test])

    print(f"[INFO] X_test shape: {X_test.shape}")
    print(f"[INFO] y_test shape: {y_test.shape}")
    print(f"[INFO] Classes: {class_names}")

    print("[INFO] Running predictions...")

    pred_probs = model.predict(
        X_test,
        batch_size=128,
        verbose=1
    )

    y_pred_raw = np.argmax(pred_probs, axis=1)

    direct_accuracy = accuracy_score(y_test_encoded, y_pred_raw)

    print(f"[INFO] Direct accuracy: {direct_accuracy:.4f}")

    y_pred_mapped, mapped_accuracy, inferred_mapping = infer_best_label_mapping(
        y_test_encoded,
        y_pred_raw,
        class_names
    )

    print(f"[INFO] Inferred-mapping accuracy: {mapped_accuracy:.4f}")
    print("[INFO] Inferred output mapping:")
    for k, v in inferred_mapping.items():
        print(f"       {k} -> {v}")

    if mapped_accuracy > direct_accuracy:
        print("[WARN] Model output order does not match sorted class order.")
        print("[WARN] Using inferred mapping for report figures.")
        y_pred = y_pred_mapped
        accuracy = mapped_accuracy
        mapping_used = "inferred"
    else:
        y_pred = y_pred_raw
        accuracy = direct_accuracy
        mapping_used = "direct"

    print(f"[RESULT] Final accuracy used: {accuracy:.4f}")

    save_text(
        OUTPUT_DIR / "cnn_accuracy.txt",
        f"Direct accuracy: {direct_accuracy:.6f}\n"
        f"Inferred-mapping accuracy: {mapped_accuracy:.6f}\n"
        f"Final accuracy used: {accuracy:.6f}\n"
        f"Mapping used: {mapping_used}\n"
    )

    with open(OUTPUT_DIR / "cnn_inferred_mapping.json", "w", encoding="utf-8") as f:
        json.dump(inferred_mapping, f, indent=4)

    report_txt = classification_report(
        y_test_encoded,
        y_pred,
        labels=numeric_class_labels,
        target_names=[str(c) for c in class_names],
        zero_division=0
    )

    report_dict = classification_report(
        y_test_encoded,
        y_pred,
        labels=numeric_class_labels,
        target_names=[str(c) for c in class_names],
        output_dict=True,
        zero_division=0
    )

    save_text(OUTPUT_DIR / "cnn_classification_report.txt", report_txt)

    with open(OUTPUT_DIR / "cnn_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(OUTPUT_DIR / "cnn_classification_report.csv")

    cm = confusion_matrix(
        y_test_encoded,
        y_pred,
        labels=numeric_class_labels
    )

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        OUTPUT_DIR / "cnn_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("CNN Confusion Matrix")
    plt.xticks(rotation=30)
    plt.tight_layout()

    cm_path = OUTPUT_DIR / "cnn_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"[OK] Saved: {cm_path}")

    cm_norm = confusion_matrix(
        y_test_encoded,
        y_pred,
        labels=numeric_class_labels,
        normalize="true"
    )

    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(
        OUTPUT_DIR / "cnn_confusion_matrix_normalized.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=class_names
    )

    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title("CNN Normalized Confusion Matrix")
    plt.xticks(rotation=30)
    plt.tight_layout()

    cm_norm_path = OUTPUT_DIR / "cnn_confusion_matrix_normalized.png"
    plt.savefig(cm_norm_path, dpi=300)
    plt.close()

    print(f"[OK] Saved: {cm_norm_path}")

    metrics = report_df.loc[
        [str(c) for c in class_names],
        ["precision", "recall", "f1-score"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    metrics.plot(kind="bar", ax=ax)

    ax.set_title("CNN Performance Metrics by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=30)
    plt.tight_layout()

    metrics_path = OUTPUT_DIR / "cnn_metrics_by_class.png"
    plt.savefig(metrics_path, dpi=300)
    plt.close()

    print(f"[OK] Saved: {metrics_path}")

    confidence = np.max(pred_probs, axis=1)
    correct = y_pred == y_test_encoded

    confidence_df = pd.DataFrame(
        {
            "confidence": confidence,
            "correct": correct,
            "y_true_index": y_test_encoded,
            "y_pred_index": y_pred,
            "y_true_label": [index_to_label[int(i)] for i in y_test_encoded],
            "y_pred_label": [index_to_label[int(i)] for i in y_pred],
        }
    )

    confidence_df.to_csv(
        OUTPUT_DIR / "cnn_prediction_confidence.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(confidence[correct], bins=30, alpha=0.7, label="Correct predictions")
    ax.hist(confidence[~correct], bins=30, alpha=0.7, label="Incorrect predictions")

    ax.set_title("CNN Prediction Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Number of samples")
    ax.legend()

    plt.tight_layout()

    conf_path = OUTPUT_DIR / "cnn_confidence_histogram.png"
    plt.savefig(conf_path, dpi=300)
    plt.close()

    print(f"[OK] Saved: {conf_path}")

    distribution = pd.Series(y_test_encoded).value_counts().sort_index()
    distribution.index = [index_to_label[int(i)] for i in distribution.index]

    distribution.to_csv(OUTPUT_DIR / "cnn_test_class_distribution.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    distribution.plot(kind="bar", ax=ax)

    ax.set_title("Class Distribution in the CNN Test Set")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=30)
    plt.tight_layout()

    dist_path = OUTPUT_DIR / "cnn_test_class_distribution.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()

    print(f"[OK] Saved: {dist_path}")

    print("[INFO] Saving ECG examples...")

    for cls in class_names:
        cls_idx = label_to_index[cls]

        idxs = np.where(
            (y_test_encoded == cls_idx) &
            (y_pred == cls_idx)
        )[0]

        if len(idxs) == 0:
            print(f"[WARN] No correct examples found for class: {cls}")
            continue

        idx = idxs[0]
        signal = X_test[idx].squeeze()

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(signal)

        ax.set_title(
            f"ECG Window Example - True={cls}, Predicted={index_to_label[int(y_pred[idx])]}"
        )

        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        fig_path = OUTPUT_DIR / f"cnn_ecg_example_class_{safe_filename(cls)}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f"[OK] Saved: {fig_path}")

    latex_table = metrics.copy()
    latex_table["support"] = report_df.loc[
        [str(c) for c in class_names],
        "support"
    ]

    latex_code = build_manual_latex_table(
        latex_table,
        class_names,
        accuracy
    )

    save_text(
        OUTPUT_DIR / "cnn_metrics_table_latex.txt",
        latex_code
    )

    summary = f"""
REAL CNN ANALYSIS SUMMARY

Dataset:
- X file: {X_PATH}
- y file: {Y_PATH}
- Label column: {label_col}
- Total samples: {len(X)}
- Test samples: {len(X_test)}
- Classes: {class_names}

Model:
- File: {MODEL_PATH}

Results:
- Direct accuracy: {direct_accuracy:.6f}
- Inferred-mapping accuracy: {mapped_accuracy:.6f}
- Final accuracy used: {accuracy:.6f}
- Mapping used: {mapping_used}

Inferred mapping:
{json.dumps(inferred_mapping, indent=4)}

Important:
If mapping_used is 'inferred', the class order saved in the model does not match alphabetical class order.
This must be explained in the report.
"""

    save_text(
        OUTPUT_DIR / "cnn_analysis_summary.txt",
        summary
    )

    print("=" * 60)
    print("[SUCCESS] REAL CNN ANALYSIS GENERATED")
    print(f"[INFO] Output folder: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()