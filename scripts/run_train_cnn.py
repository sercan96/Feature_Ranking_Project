import os
import sys
import numpy as np
import pandas as pd

# Proje kökünü path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.evaluate import evaluate_classification_model
from src.config import CNN_MODEL_DIR
from src.utils import ensure_dir, save_json
from src.models import build_cnn


def extract_feature_scores_from_first_conv(model, layer_name="feature_conv"):
    """
    İlk Conv1D katmanının genel weight özetini çıkarır.
    """
    feature_layer = model.get_layer(layer_name)
    weights, bias = feature_layer.get_weights()

    abs_weights = np.abs(weights)

    weight_max_score = float(np.max(abs_weights))
    weight_avg_score = float(np.mean(abs_weights))

    return {
        "raw_weights_shape": list(weights.shape),
        "global_weight_max": weight_max_score,
        "global_weight_avg": weight_avg_score
    }


def save_first_conv_kernels(model, output_dir, layer_name="feature_conv"):
    """
    İlk Conv1D katmanının kernel ağırlıklarını tablo halinde kaydeder.
    Ayrıca feature başına MAX ve AVG weight skorlarını üretir.
    """
    feature_layer = model.get_layer(layer_name)
    weights, bias = feature_layer.get_weights()

    print("\n--- First Conv Raw Kernel Shape ---")
    print(weights.shape)

    # Beklenen şekil: (n_features, 1, n_filters)
    squeezed_weights = np.squeeze(weights)

    print("\n--- First Conv Squeezed Kernel Shape ---")
    print(squeezed_weights.shape)

    # Tek filtre varsa güvenli hale getir
    if squeezed_weights.ndim == 1:
        squeezed_weights = squeezed_weights.reshape(-1, 1)

    # Kernel tablosu: satır = feature, sütun = filter
    df_kernels = pd.DataFrame(
        squeezed_weights,
        index=[f"F{i+1}" for i in range(squeezed_weights.shape[0])],
        columns=[f"Filter{j+1}" for j in range(squeezed_weights.shape[1])]
    )

    kernel_csv_path = CNN_MODEL_DIR / "cnn_first_conv_kernels.csv"
    df_kernels.to_csv(kernel_csv_path)

    print("\n--- First Conv Kernel Weights (ilk 10 feature) ---")
    print(df_kernels.head(10).to_string())

    # Sadece kernel weight'lerine göre feature skorları
    abs_kernel_weights = np.abs(squeezed_weights)

    # skorları hesapla
    feature_max_from_weights = np.max(abs_kernel_weights, axis=1)
    feature_avg_from_weights = np.mean(abs_kernel_weights, axis=1)

    # ortak feature listesi
    features = [f"F{i+1}" for i in range(len(feature_avg_from_weights))]

    # MAX dataframe
    df_max_scores = pd.DataFrame({
        "feature": features,
        "max_weight_score": feature_max_from_weights
    }).sort_values(by="max_weight_score", ascending=False).reset_index(drop=True)

    # AVG dataframe
    df_avg_scores = pd.DataFrame({
        "feature": features,
        "avg_weight_score": feature_avg_from_weights
    }).sort_values(by="avg_weight_score", ascending=False).reset_index(drop=True)

    # CSV kaydet
    df_max_scores.to_csv(output_dir / "cnn_feature_weight_ranking_max.csv", index=False)
    df_avg_scores.to_csv(output_dir / "cnn_feature_weight_ranking_avg.csv", index=False)
    print("\n--- Feature Weight Scores (AVG'ye göre sıralı, ilk 10) ---")
    print(df_avg_scores.head(10).to_string(index=False))

    return df_kernels, df_max_scores, df_avg_scores


def main():
    # 1. Veri yükleme
    df = load_data("breast_cancer_data.csv")

    # 2. Preprocessing
    processed = preprocess_data(df)

    X_train = processed["X_train_cnn"]
    X_test = processed["X_test_cnn"]
    y_train = processed["y_train"]
    y_test = processed["y_test"]

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("Train sample sayısı:", X_train.shape[0])
    print("Test sample sayısı :", X_test.shape[0])
    print("Feature sayısı     :", X_train.shape[1])
    print("Channel sayısı     :", X_train.shape[2])

    # 3. Klasörü garanti et
    ensure_dir(CNN_MODEL_DIR)

    # 4. Modeli oluştur
    model = build_cnn(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    print("\n--- CNN Model Summary ---")
    model.summary()

    # 5. Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # 6. Callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # 7. Eğit
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1
    )

    # 8. Eğitim geçmişini kaydet
    save_json(history.history, CNN_MODEL_DIR / "cnn_history.json")

    best_val_loss = min(history.history["val_loss"])
    best_epoch = history.history["val_loss"].index(best_val_loss) + 1

    best_info = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }
    save_json(best_info, CNN_MODEL_DIR / "cnn_best_info.json")

    # 9. Tam modeli kaydet
    model.save(CNN_MODEL_DIR / "cnn_model_full.keras")

    # 10. Test değerlendirmesi
    metrics, y_pred = evaluate_classification_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name="cnn_feature_ranking_model"
    )

    print("\n--- CNN Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    save_json(metrics, CNN_MODEL_DIR / "cnn_test_metrics.json")

    # 11. İlk katman genel weight özeti
    weight_summary = extract_feature_scores_from_first_conv(
        model,
        layer_name="feature_conv"
    )
    save_json(weight_summary, CNN_MODEL_DIR / "cnn_first_conv_weight_summary.json")

    # 12. İlk conv kernel ağırlıklarını kaydet ve weight-based ranking çıkar
    kernel_df, weight_max_df, weight_avg_df = save_first_conv_kernels(
        model,
        CNN_MODEL_DIR,
        layer_name="feature_conv"
    )
    print("Top 10 Kernel Weights (ilk 10 feature):")
    print(kernel_df.head(10).to_string())

    print("Top 10 Kernel MAX (ilk 10 feature):")
    print(weight_max_df.head(10).to_string(index=False))

    print("Top 10 Kernel AVG (ilk 10 feature):")
    print(weight_avg_df.head(10).to_string(index=False))

    # 13. Sıralamaları string olarak yazdır
    feature_order_max = " > ".join(weight_max_df["feature"].tolist())
    feature_order_avg = " > ".join(weight_avg_df["feature"].tolist())

    print("\n--- Top 10 Features (Kernel MAX Weight) ---")
    print(weight_max_df.head(10).to_string(index=False))

    print("\n--- Top 10 Features (Kernel AVG Weight) ---")
    print(weight_avg_df.head(10).to_string(index=False))

    print("\nFeature Max Weight Ranking:")
    print(feature_order_max)

    print("\nFeature Avg Weight Ranking:")
    print(feature_order_avg)


    print("\nCNN modeli eğitildi ve kernel-weight tabanlı feature ranking dosyaları oluşturuldu.")


if __name__ == "__main__":
    main()