import os
import sys
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import argparse
from pathlib import Path

# Proje kökünü path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.callbacks import EarlyStopping

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.config import AUTOENCODER_MODEL_DIR, get_data, get_model_output_dir, RANDOM_STATE
from src.utils import ensure_dir, save_json
from src.models import build_autoencoder
from src.feature_selector import FeatureSelector, FeatureSelectionConfig


def validate_feature_percent(feature_percent: float) -> float:
    if feature_percent <= 0 or feature_percent > 100:
        raise ValueError("feature-percent 0 ile 100 arasında olmalı (100 dahil).")
    return feature_percent


def calculate_top_k_from_percent(total_features: int, feature_percent: float) -> int:
    selection_ratio = feature_percent / 100.0
    return max(int(total_features * selection_ratio), 1)


def create_filtered_feature_datasets(feature_percent: float) -> int:
    base_dir = Path(__file__).resolve().parent.parent
    ratio = feature_percent / 100.0

    config = FeatureSelectionConfig(
        dataset_path=str(get_data()),
        ranking_max_path=str(AUTOENCODER_MODEL_DIR / "autoencoder_feature_weight_ranking_max.csv"),
        ranking_avg_path=str(AUTOENCODER_MODEL_DIR / "autoencoder_feature_weight_ranking_avg.csv"),
        output_dir=str(base_dir / "data" / "filtered_datasets" / "autoencoder" / "breast_cancer_data" / "reports"),
        label_column="diagnosis",
        excluded_columns=["id", "ID"],
        selection_ratio=ratio,
        min_features=1,
    )

    selector = FeatureSelector(config)
    selector.load_files()
    selector.create_both_datasets()

    total_features = len(selector._get_feature_columns())
    top_k = calculate_top_k_from_percent(total_features, feature_percent)

    print("\n--- Filtered Dataset Uretimi (Autoencoder) ---")
    print(f"Secilen oran: %{feature_percent}")
    print(f"Toplam feature: {total_features}")
    print(f"Secilen feature sayisi (top_k): {top_k}")

    return top_k


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


def save_first_encoder_weights(
    autoencoder,
    feature_names,
    output_dir,
    reports_output_dir=None,
    layer_name="encoder_dense_1"
):
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

    if reports_output_dir is not None:
        reports_output_dir.mkdir(parents=True, exist_ok=True)
        df_weights.to_csv(reports_output_dir / "first_encoder_weights.csv")

    print("\n--- First Encoder Weights (ilk 10 feature) ---")
    print(df_weights.head(10).to_string())

    abs_weights = np.abs(weights)

    feature_max_from_weights = np.max(abs_weights, axis=1)
    feature_avg_from_weights = np.mean(abs_weights, axis=1)

    if len(feature_names) != len(feature_avg_from_weights):
        raise ValueError(
            f"Feature name sayısı ({len(feature_names)}) ile weight sayısı ({len(feature_avg_from_weights)}) eşleşmiyor."
        )

    features = [f"F{i+1}" for i in range(len(feature_avg_from_weights))]

    df_max_scores = pd.DataFrame({
        "feature": features,
        "feature_name": feature_names,
        "max_weight_score": feature_max_from_weights
    }).sort_values(by="max_weight_score", ascending=False).reset_index(drop=True)

    df_avg_scores = pd.DataFrame({
        "feature": features,
        "feature_name": feature_names,
        "avg_weight_score": feature_avg_from_weights
    }).sort_values(by="avg_weight_score", ascending=False).reset_index(drop=True)

    df_max_scores.to_csv(output_dir / "autoencoder_feature_weight_ranking_max.csv", index=False)
    df_avg_scores.to_csv(output_dir / "autoencoder_feature_weight_ranking_avg.csv", index=False)

    if reports_output_dir is not None:
        df_max_scores.to_csv(reports_output_dir / "feature_weight_ranking_max.csv", index=False)
        df_avg_scores.to_csv(reports_output_dir / "feature_weight_ranking_avg.csv", index=False)

    print("\n--- Feature Weight Scores (MAX'e göre sıralı, ilk 10) ---")
    print(df_max_scores.head(10).to_string(index=False))

    print("\n--- Feature Weight Scores (AVG'ye göre sıralı, ilk 10) ---")
    print(df_avg_scores.head(10).to_string(index=False))

    return df_weights, df_max_scores, df_avg_scores


def main(mode="original", feature_percent=20.0, ranking_type="avg"):
    feature_percent = validate_feature_percent(feature_percent)
    ranking_type = ranking_type.lower()
    if ranking_type not in {"avg", "max"}:
        raise ValueError("ranking-type sadece 'avg' veya 'max' olabilir.")

    # Random seed set et - Reproducible results
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    
    # 1. Veri yükleme - mode'a göre
    if mode == "original":
        print("\n[ORIGINAL MODE] Orijinal veri yükleniyor (31 feature)...")
        df = load_data("breast_cancer_data.csv", folder="raw")
    elif mode == "filtered":
        raw_df = load_data("breast_cancer_data.csv", folder="raw")
        excluded_columns_for_count = {"diagnosis", "target", "label", "class", "Unnamed: 0", "id", "ID"}
        total_features = len([c for c in raw_df.columns if c not in excluded_columns_for_count])
        expected_top_k = calculate_top_k_from_percent(total_features, feature_percent)
        filtered_file_name = f"breast_cancer_data_top_{expected_top_k}_{ranking_type}_features.csv"

        print("\n" + "="*80)
        print("="*80 + "\n")
        print(f"[FILTERED MODE] Yuklenecek dosya: {filtered_file_name}")
        df = load_data(filtered_file_name, "autoencoder", "breast_cancer_data", folder="filtered_datasets")
    else:
        raise ValueError("Mode 'original' veya 'filtered' olmalı.")

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

    if mode == "filtered" and X_train.shape[1] != expected_top_k:
        print(f"\n  UYARI: Filtrelenmiş modda {expected_top_k} feature bekleniyor ama {X_train.shape[1]} feature bulundu!")
        print("Lütfen filtered_datasets/autoencoder/breast_cancer_data/reports/ klasöründeki CSV dosyalarını kontrol edin.")

    # 3. Klasörü garanti et
    ensure_dir(AUTOENCODER_MODEL_DIR)
    output_reports_dir = get_model_output_dir("autoencoder", "breast_cancer_data", "reports")

    possible_target_columns = ["diagnosis", "target", "label", "class", "Unnamed: 0", "id", "ID"]
    excluded_columns = ["id", "ID"]
    for col in possible_target_columns:
        if col in df.columns:
            excluded_columns.append(col)
    feature_names = [col for col in df.columns if col not in excluded_columns]

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

    # 7. Eğitim geçmişini kaydet (ORIGINAL MODE)
    if mode == "original":
        save_json(history.history, AUTOENCODER_MODEL_DIR / "autoencoder_history.json")

    best_val_loss = min(history.history["val_loss"])
    best_epoch = history.history["val_loss"].index(best_val_loss) + 1

    best_info = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }
    if mode == "original":
        save_json(best_info, AUTOENCODER_MODEL_DIR / "autoencoder_best_info.json")

    # 8. Modeli kaydet (ORIGINAL MODE)
    if mode == "original":
        autoencoder.save(AUTOENCODER_MODEL_DIR / "autoencoder_model_full.keras")
        encoder.save(AUTOENCODER_MODEL_DIR / "encoder_model_full.keras")

    # 9. Test reconstruction metriği
    test_mse = float(autoencoder.evaluate(X_test, X_test, verbose=0))
    test_metrics = {"test_mse": test_mse}
    if mode == "filtered":
        metrics_path = Path("outputs") / "autoencoder" / "breast_cancer_data_filtered_metrics" / f"{ranking_type.upper()}_test_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        metrics_path = Path("outputs") / "autoencoder" / "breast_cancer_data" / "metrics" / "ORG_test_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(test_metrics, metrics_path)
    print(f"\n✓ Test metrigi kaydedildi: {metrics_path}")
    print(f"test_mse: {test_mse:.6f}")

    # 10. İlk encoder katman genel weight özeti (ORIGINAL MODE)
    if mode == "original":
        weight_summary = extract_feature_scores_from_first_encoder(
            autoencoder,
            layer_name="encoder_dense_1"
        )
        save_json(weight_summary, AUTOENCODER_MODEL_DIR / "autoencoder_first_encoder_weight_summary.json")

    # 11. İlk encoder ağırlıklarını kaydet ve ranking çıkar (ORIGINAL MODE)
    if mode == "original":
        weight_df, weight_max_df, weight_avg_df = save_first_encoder_weights(
            autoencoder,
            feature_names,
            AUTOENCODER_MODEL_DIR,
            reports_output_dir=output_reports_dir,
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

        top_k = create_filtered_feature_datasets(feature_percent)
        print(f"\n✓ %{feature_percent} secimi ile filtered datasetler uretildi (top_{top_k}).")
    else:
        print("\n" + "="*80)
        print("[FILTERED MODE] Tamamlandı!")
        print(f"✓ Test metrigi suraya kaydedildi: outputs/autoencoder/breast_cancer_data_filtered_metrics/{ranking_type.upper()}_test_metrics.json")
        print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder egitimi ve feature bazli filtreli dataset olusturma")
    parser.add_argument("--mode", choices=["original", "filtered"], default="original", help="Calisma modu")
    parser.add_argument("--feature-percent", type=float, default=20.0, help="Secilecek feature yuzdesi (or. 30)")
    parser.add_argument("--ranking-type", choices=["avg", "max"], default="avg", help="Filtered modda hangi ranking dosyasi kullanilsin")

    args = parser.parse_args()
    main(mode=args.mode, feature_percent=args.feature_percent, ranking_type=args.ranking_type)