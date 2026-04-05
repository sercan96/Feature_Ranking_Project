import os
import sys
import numpy as np
import pandas as pd
import random
import tensorflow as tf

# Proje kökünü path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.callbacks import EarlyStopping

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.config import AUTOENCODER_MODEL_DIR, RANDOM_STATE
from src.utils import ensure_dir, save_json
from src.models import build_autoencoder


def extract_feature_scores_from_first_encoder(autoencoder, layer_name="encoder_dense_1"):
    """
    İlk encoder dense katmanının genel weight özetini çıkarır.
    """
    encoder_layer = autoencoder.get_layer(layer_name)
    weights, bias = encoder_layer.get_weights()

    abs_weights = np.abs(weights)

    weight_max_score = float(np.max(abs_weights))
    weight_avg_score = float(np.mean(abs_weights))

    return {
        "raw_weights_shape": list(weights.shape),
        "global_weight_max": weight_max_score,
        "global_weight_avg": weight_avg_score
    }


def save_first_encoder_weights(autoencoder, output_dir, layer_name="encoder_dense_1"):
    """
    İlk encoder katmanının ağırlıklarını tablo halinde kaydeder.
    Ayrıca feature başına MAX ve AVG weight skorlarını üretir.
    """
    encoder_layer = autoencoder.get_layer(layer_name)
    weights, bias = encoder_layer.get_weights()

    print("\n--- First Encoder Raw Weight Shape ---")
    print(weights.shape)

    # Beklenen şekil: (n_features, n_hidden)
    df_weights = pd.DataFrame(
        weights,
        index=[f"F{i+1}" for i in range(weights.shape[0])],
        columns=[f"Neuron{j+1}" for j in range(weights.shape[1])]
    )

    weights_csv_path = output_dir / "autoencoder_first_encoder_weights.csv"
    df_weights.to_csv(weights_csv_path)

    print("\n--- First Encoder Weights (ilk 10 feature) ---")
    print(df_weights.head(10).to_string())

    abs_weights = np.abs(weights)

    feature_max_from_weights = np.max(abs_weights, axis=1)
    feature_avg_from_weights = np.mean(abs_weights, axis=1)

    features = [f"F{i+1}" for i in range(len(feature_avg_from_weights))]

    df_max_scores = pd.DataFrame({
        "feature": features,
        "max_weight_score": feature_max_from_weights
    }).sort_values(by="max_weight_score", ascending=False).reset_index(drop=True)

    df_avg_scores = pd.DataFrame({
        "feature": features,
        "avg_weight_score": feature_avg_from_weights
    }).sort_values(by="avg_weight_score", ascending=False).reset_index(drop=True)

    df_max_scores.to_csv(output_dir / "autoencoder_feature_weight_ranking_max.csv", index=False)
    df_avg_scores.to_csv(output_dir / "autoencoder_feature_weight_ranking_avg.csv", index=False)

    print("\n--- Feature Weight Scores (MAX'e göre sıralı, ilk 10) ---")
    print(df_max_scores.head(10).to_string(index=False))

    print("\n--- Feature Weight Scores (AVG'ye göre sıralı, ilk 10) ---")
    print(df_avg_scores.head(10).to_string(index=False))

    return df_weights, df_max_scores, df_avg_scores


def main():
    # Random seed set et - Reproducible results
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    
    # 1. Veri yükleme
    df = load_data("breast_cancer_data.csv")

    # 2. Preprocessing
    processed = preprocess_data(df)

    # Autoencoder için düz (2D) veri lazım
    X_train = processed["X_train"]
    X_test = processed["X_test"]

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("Train sample sayısı:", X_train.shape[0])
    print("Test sample sayısı :", X_test.shape[0])
    print("Feature sayısı     :", X_train.shape[1])

    # 3. Klasörü garanti et
    ensure_dir(AUTOENCODER_MODEL_DIR)

    # 4. Modeli oluştur
    autoencoder, encoder = build_autoencoder(
        input_dim=X_train.shape[1],
        encoding_dim=8
    )

    print("\n--- Autoencoder Model Summary ---")
    autoencoder.summary()

    # 5. Callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # 6. Eğit
    history = autoencoder.fit(
        X_train, #input ve target aynı çünkü autoencoder
        X_train,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1
    )

    # 7. Eğitim geçmişini kaydet
    save_json(history.history, AUTOENCODER_MODEL_DIR / "autoencoder_history.json")

    best_val_loss = min(history.history["val_loss"])
    best_epoch = history.history["val_loss"].index(best_val_loss) + 1

    best_info = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }
    save_json(best_info, AUTOENCODER_MODEL_DIR / "autoencoder_best_info.json")

    # 8. Modeli kaydet
    autoencoder.save(AUTOENCODER_MODEL_DIR / "autoencoder_model_full.keras")
    encoder.save(AUTOENCODER_MODEL_DIR / "encoder_model_full.keras")

    # 9. İlk encoder katman genel weight özeti
    weight_summary = extract_feature_scores_from_first_encoder(
        autoencoder,
        layer_name="encoder_dense_1"
    )
    save_json(weight_summary, AUTOENCODER_MODEL_DIR / "autoencoder_first_encoder_weight_summary.json")

    # 10. İlk encoder ağırlıklarını kaydet ve ranking çıkar
    weight_df, weight_max_df, weight_avg_df = save_first_encoder_weights(
        autoencoder,
        AUTOENCODER_MODEL_DIR,
        layer_name="encoder_dense_1"
    )

    print("\n--- Top 10 Features (Encoder MAX Weight) ---")
    print(weight_max_df.head(10).to_string(index=False))

    print("\n--- Top 10 Features (Encoder AVG Weight) ---")
    print(weight_avg_df.head(10).to_string(index=False))

    feature_order_max = " > ".join(weight_max_df["feature"].tolist())
    feature_order_avg = " > ".join(weight_avg_df["feature"].tolist())

    print("\nFeature Max Weight Ranking:")
    print(feature_order_max)

    print("\nFeature Avg Weight Ranking:")
    print(feature_order_avg)

    print("\nAutoencoder modeli eğitildi ve weight tabanlı feature ranking dosyaları oluşturuldu.")


if __name__ == "__main__":
    main()