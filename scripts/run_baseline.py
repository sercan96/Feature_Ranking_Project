from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.models import build_baseline_model
from src.evaluate import (
    evaluate_classification_model,
    save_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.train import save_sklearn_model
from src.config import BASELINE_MODEL_DIR, METRICS_DIR, BASE_DIR
from src.utils import ensure_dir, save_json

def main():
    # 1. Veriyi yükle
    df = load_data()

    # 2. Preprocessing yap
    processed = preprocess_data(df)

    X_train = processed["X_train_scaled"]
    X_test = processed["X_test_scaled"]
    y_train = processed["y_train"]
    y_test = processed["y_test"]

    # 3. Modeli oluştur
    model = build_baseline_model()

    # 4. Modeli eğit
    model.fit(X_train, y_train)

    # 5. Değerlendir
    metrics, y_pred, y_prob = evaluate_classification_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name="baseline_logistic_regression"
    )

    # 6. Metrikleri kaydet
    save_metrics(metrics, "baseline_logistic_regression_metrics.json")

    # 7. Grafikleri kaydet
    plot_confusion_matrix(
        y_test=y_test,
        y_pred=y_pred,
        model_name="baseline_logistic_regression"
    )

    plot_roc_curve(
        y_test=y_test,
        y_prob=y_prob,
        model_name="baseline_logistic_regression"
    )

    print("\n--- Baseline Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nBaseline model başarıyla eğitildi ve değerlendirildi.")

    ensure_dir(BASELINE_MODEL_DIR)

    # Modeli kaydet
    save_sklearn_model(
        model,
        BASELINE_MODEL_DIR / "baseline_logistic_regression.pkl"
    )

    # Katsayıları kaydet
    coef_data = {
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.tolist()
    }
    save_json(coef_data, BASELINE_MODEL_DIR / "baseline_logistic_regression_weights.json")
    
if __name__ == "__main__":
    main()