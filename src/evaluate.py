""""" Info: Evaluate dosyası, model performansını değerlendirmek için kullanılır."""

import json
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    classification_report,
)

from src.config import FIGURES_DIR, METRICS_DIR
from src.utils import ensure_dir


def evaluate_classification_model(model, X_test, y_test, model_name="model"):
    """
    Sınıflandırma modelini değerlendirir ve metrikleri döndürür.
    """
    y_prob = model.predict(X_test)

    # Binary: (N, 1) olasılık çıktısı. Multiclass: (N, C) softmax çıktısı.
    if len(y_prob.shape) == 1 or (len(y_prob.shape) == 2 and y_prob.shape[1] == 1):
        y_pred = (y_prob.reshape(-1) > 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        }
    else:
        y_pred = y_prob.argmax(axis=1)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

    print(f"\n--- {model_name} Classification Report ---")
    print(classification_report(y_test, y_pred))

    return metrics, y_pred


def save_metrics(metrics: dict, filename: str):
    """
    Metrikleri JSON dosyası olarak kaydeder.
    """
    ensure_dir(METRICS_DIR)

    output_path = METRICS_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def plot_confusion_matrix(y_test, y_pred, model_name="model"):
    """
    Confusion matrix grafiğini oluşturur ve kaydeder.
    """
    ensure_dir(FIGURES_DIR)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(FIGURES_DIR / f"{model_name}_confusion_matrix.png", bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_test, model_name="model"):
    """
    ROC curve grafiğini oluşturur ve kaydeder.
    """

    ensure_dir(FIGURES_DIR)

    fpr, tpr, _ = roc_curve(y_test)

    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.savefig(FIGURES_DIR / f"{model_name}_roc_curve.png", bbox_inches="tight")
    plt.close()