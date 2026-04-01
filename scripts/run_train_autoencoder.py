import sys
import os

# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.models import build_autoencoder
from src.evaluate import (
    evaluate_classification_model,
    save_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.config import AUTOENCODER_MODEL_DIR
from src.utils import ensure_dir, save_json

def main():
    # 1. Veriyi yükle
    df = load_data()

    # 2. Preprocessing
    processed = preprocess_data(df)

    X_train = processed["X_train_scaled"]
    X_test = processed["X_test_scaled"]
    y_train = processed["y_train"]
    y_test = processed["y_test"]

    # 3. Klasör oluştur
    ensure_dir(AUTOENCODER_MODEL_DIR)

    # 4. Autoencoder ve encoder oluştur
    autoencoder, encoder = build_autoencoder(
        input_dim=X_train.shape[1],
        encoding_dim=8
    )

    print("\n--- Autoencoder Summary ---")
    autoencoder.summary()

    print("\n--- Encoder Summary ---")
    encoder.summary()

    # 5. Callback'ler
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath=str(AUTOENCODER_MODEL_DIR / "autoencoder_best.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # 6. Autoencoder eğitimi
    history = autoencoder.fit(
        X_train,
        X_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # 7. Eğitim geçmişi kaydet
    save_json(history.history, AUTOENCODER_MODEL_DIR / "autoencoder_history.json")

    # 8. Modelleri kaydet
    autoencoder.save(AUTOENCODER_MODEL_DIR / "autoencoder_full.keras")
    encoder.save(AUTOENCODER_MODEL_DIR / "encoder_full.keras")

    # 9. Encoder çıktıları üret
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    print("\n--- Encoded Train Shape ---")
    print(X_train_encoded.shape)

    print("\n--- Encoded Test Shape ---")
    print(X_test_encoded.shape)

    # 10. Encoded verilerle classifier eğit
    classifier = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    classifier.fit(X_train_encoded, y_train)

    # 11. Değerlendirme
    metrics, y_pred, y_prob = evaluate_classification_model(
        model=classifier,
        X_test=X_test_encoded,
        y_test=y_test,
        model_name="autoencoder_logistic_regression"
    )

    # 12. Sonuçları kaydet
    save_metrics(metrics, "autoencoder_logistic_regression_metrics.json")
    plot_confusion_matrix(y_test, y_pred, model_name="autoencoder_logistic_regression")
    plot_roc_curve(y_test, y_prob, model_name="autoencoder_logistic_regression")

    print("\n--- Autoencoder + Logistic Regression Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nAutoencoder eğitildi, encoder çıkarıldı ve classifier değerlendirildi.")
    #print("\n Feature Sıralaması : " + )

if __name__ == "__main__":
    main()