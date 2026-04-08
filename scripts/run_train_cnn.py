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
from tensorflow.keras.optimizers import Adam

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.evaluate import evaluate_classification_model
from src.utils import ensure_dir, save_json
from src.models import build_cnn
from src.config import get_model_output_dir, get_data, RANDOM_STATE
from src.feature_selector import FeatureSelector, FeatureSelectionConfig


def validate_feature_percent(feature_percent: float) -> float:
    if feature_percent <= 0 or feature_percent > 100:
        raise ValueError("feature-percent 0 ile 100 arasında olmalı (100 dahil).")
    return feature_percent


def calculate_top_k_from_percent(total_features: int, feature_percent: float) -> int:
    selection_ratio = feature_percent / 100.0
    return max(int(total_features * selection_ratio), 1)


def normalize_dataset_name(dataset_name: str) -> tuple[str, str]:
    dataset_name = dataset_name.strip()
    if not dataset_name:
        raise ValueError("dataset-name boş olamaz.")

    dataset_filename = dataset_name if dataset_name.lower().endswith(".csv") else f"{dataset_name}.csv"
    dataset_folder = Path(dataset_filename).stem
    return dataset_filename, dataset_folder


def normalize_optional_column(column_name: str | None) -> str | None:
    if column_name is None:
        return None
    cleaned = column_name.strip()
    if cleaned.lower() in {"", "none", "null", "-"}:
        return None
    return cleaned


def create_filtered_feature_datasets(
    feature_percent: float,
    dataset_name: str,
    target_column: str,
    id_column: str | None,
) -> int:
    dataset_filename, dataset_folder = normalize_dataset_name(dataset_name)
    base_dir = Path(__file__).resolve().parent.parent
    ratio = feature_percent / 100.0

    config = FeatureSelectionConfig(
        dataset_path=str(get_data(dataset_filename)),
        ranking_max_path=str(get_model_output_dir("cnn", dataset_folder, "reports") / "feature_weight_ranking_max.csv"),
        ranking_avg_path=str(get_model_output_dir("cnn", dataset_folder, "reports") / "feature_weight_ranking_avg.csv"),
        output_dir=str(base_dir / "data" / "filtered_datasets" / "cnn" / dataset_folder / "reports"),
        label_column=target_column,
        id_column=id_column,
        excluded_columns=["id", "ID"],
        selection_ratio=ratio,
        min_features=1,
    )

    selector = FeatureSelector(config)
    selector.load_files()
    selector.create_both_datasets()

    total_features = len(selector._get_feature_columns())
    top_k = calculate_top_k_from_percent(total_features, feature_percent)

    print("\n--- Filtered Dataset Uretimi ---")
    print(f"Secilen oran: %{feature_percent}")
    print(f"Toplam feature: {total_features}")
    print(f"Secilen feature sayisi (top_k): {top_k}")

    return top_k

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


def save_first_conv_kernels(model, feature_names, output_dir_name, layer_name="feature_conv"):
    """
    İlk Conv1D katmanının kernel ağırlıklarını tablo halinde kaydeder.
    Ayrıca feature başına MAX ve AVG weight skorlarını üretir.
    """
    feature_layer = model.get_layer(layer_name)
    weights, bias = feature_layer.get_weights()

    print("\n--- First Conv Raw Kernel Shape ---")
    print(weights.shape)

    # Beklenen şekil: (n_features, 1, n_filters)
    # Gereksiz verileri kaldırıyor.
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

    kernel_csv_path = get_model_output_dir("cnn", output_dir_name, "reports") / "first_conv_kernels.csv"
    df_kernels.to_csv(kernel_csv_path)

    print("\n--- First Conv Kernel Weights (ilk 10 feature) ---")
    print(df_kernels.head(10).to_string())

    # Sadece kernel weight'lerine göre feature skorları
    abs_kernel_weights = np.abs(squeezed_weights)

    # skorları hesapla
    feature_max_from_weights = np.max(abs_kernel_weights, axis=1)
    feature_avg_from_weights = np.mean(abs_kernel_weights, axis=1)

    # güvenlik kontrolü
    if len(feature_names) != len(feature_avg_from_weights):
        raise ValueError(
            f"Feature name sayısı ({len(feature_names)}) ile weight sayısı ({len(feature_avg_from_weights)}) eşleşmiyor."
        )

    feature_ids = [f"F{i+1}" for i in range(len(feature_avg_from_weights))]

    # MAX dataframe
    df_max_scores = pd.DataFrame({
        "feature": feature_ids,
        "feature_name": feature_names,
        "max_weight_score": feature_max_from_weights
    }).sort_values(by="max_weight_score", ascending=False).reset_index(drop=True)

    # AVG dataframe
    df_avg_scores = pd.DataFrame({
        "feature": feature_ids,
        "feature_name": feature_names,
        "avg_weight_score": feature_avg_from_weights
    }).sort_values(by="avg_weight_score", ascending=False).reset_index(drop=True)

    # CSV kaydet
    df_max_scores.to_csv(get_model_output_dir("cnn", output_dir_name, "reports") / "feature_weight_ranking_max.csv", index=False)
    df_avg_scores.to_csv(get_model_output_dir("cnn", output_dir_name, "reports") / "feature_weight_ranking_avg.csv", index=False)
    print("\n--- Feature Weight Scores (AVG'ye göre sıralı, ilk 10) ---")
    print(df_avg_scores.head(10).to_string(index=False))

    return df_kernels, df_max_scores, df_avg_scores


def main(
    mode="original",
    feature_percent=20.0,
    ranking_type="avg",
    dataset_name="breast_cancer_data.csv",
    target_column="diagnosis",
    id_column="ID",
):

    feature_percent = validate_feature_percent(feature_percent)
    ranking_type = ranking_type.lower()
    dataset_filename, dataset_folder = normalize_dataset_name(dataset_name)
    target_column = target_column.strip()
    id_column = normalize_optional_column(id_column)
    if ranking_type not in {"avg", "max"}:
        raise ValueError("ranking-type sadece 'avg' veya 'max' olabilir.")

    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    
    # Veri yükleme - mode'a göre
    if mode == "original":
        print(f"\n[ORIGINAL MODE] Orijinal veri yükleniyor: {dataset_filename}")
        df = load_data(dataset_filename, folder="raw")
        output_prefix = dataset_folder
    elif mode == "filtered":
        raw_df = load_data(dataset_filename, folder="raw")
        excluded_columns_for_count = {target_column, "Unnamed: 0", "id", "ID"}
        if id_column:
            excluded_columns_for_count.add(id_column)
        total_features = len([c for c in raw_df.columns if c not in excluded_columns_for_count])
        expected_top_k = calculate_top_k_from_percent(total_features, feature_percent)
        filtered_file_name = f"{dataset_folder}_top_{expected_top_k}_{ranking_type}_features.csv"

        print("\n" + "="*80)
        print("="*80 + "\n")
        print(f"[FILTERED MODE] Yuklenecek dosya: {filtered_file_name}")
        df = load_data(filtered_file_name, "cnn", dataset_folder, folder="filtered_datasets")
        output_prefix = f"{dataset_folder}_filtered"
    else:
        raise ValueError("Mode 'original' veya 'filtered' olmalı.")

    # 2. Preprocessing
    processed = preprocess_data(df, target_column=target_column, id_column=id_column)

    X_train_tabular = processed["X_train"]
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
    
    # Feature sayısını kontrol et - Filtrelenmiş modda 6 feature olmalı
    if mode == "filtered" and X_train.shape[1] != expected_top_k:
        print(f"\n  UYARI: Filtrelenmiş modda {expected_top_k} feature bekleniyor ama {X_train.shape[1]} feature bulundu!")
        print(f"Lütfen filtered_datasets/cnn/{dataset_folder}/reports/ klasöründeki CSV dosyalarını kontrol edin.")

    # Modelin gerçekten gördüğü feature sırasını kullan.
    feature_names = list(X_train_tabular.columns)

    # 3. Klasörü garanti et
    #ensure_dir(get_model_output_dir("cnn","reports"))

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

    if mode == "original":
        # 8. Eğitim geçmişini kaydet
        history_path = get_model_output_dir("cnn", output_prefix, "metrics") / "history.json"
        save_json(history.history, history_path)
        print(f"✓ Eğitim geçmişi kaydedildi: {history_path}")

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
        best_info_path = get_model_output_dir("cnn", output_prefix, "metrics") / "best_info.json"
        save_json(best_info, best_info_path)
        print(f"✓ Best info kaydedildi: {best_info_path}")

        # 9. Tam modeli kaydet
        model_path = get_model_output_dir("cnn", output_prefix, "models") / "model_full.keras"
        model.save(model_path)
        print(f"✓ Model kaydedildi: {model_path}")

    # 10. Test değerlendirmesi
    metrics, y_pred = evaluate_classification_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name="feature_ranking_model"
    )

    print("\n--- CNN Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    if mode == "filtered":
        metrics_path = Path("outputs") / "cnn" / f"{dataset_folder}_filtered_metrics" / f"{ranking_type.upper()}_test_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        metrics_path = get_model_output_dir("cnn", output_prefix, "metrics") / "org_test_metrics.json"
    save_json(metrics, metrics_path)
    print(f"\n✓ Metrikleri kaydedildi: {metrics_path}")

    # 11. İlk katman genel weight özeti (SADECE ORIGINAL MODE'DA)
    if mode == "original":
        weight_summary = extract_feature_scores_from_first_conv(
            model,
            layer_name="feature_conv"
        )
        weight_summary_path = get_model_output_dir("cnn", output_prefix, "metrics") / "first_conv_weight_summary.json"
        save_json(weight_summary, weight_summary_path)
        print(f"✓ Weight özeti kaydedildi: {weight_summary_path}")

    # 12. İlk conv kernel ağırlıklarını kaydet (SADECE ORIGINAL MODE'DA)
    if mode == "original":
        kernel_df, weight_max_df, weight_avg_df = save_first_conv_kernels(
            model=model,
            feature_names=feature_names,
            output_dir_name=output_prefix,
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

        top_k = create_filtered_feature_datasets(
            feature_percent,
            dataset_filename,
            target_column,
            id_column,
        )
        print(f"\n✓ %{feature_percent} secimi ile filtered datasetler uretildi (top_{top_k}).")
    else:
        print("\n" + "="*80)
        print("[FILTERED MODE] Tamamlandı!")
        print(f"✓ Test metrigi suraya kaydedildi: outputs/cnn/{dataset_folder}_filtered_metrics/{ranking_type.upper()}_test_metrics.json")
        print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN egitimi ve feature bazli filtreli dataset olusturma")
    parser.add_argument("--mode", choices=["original", "filtered"], default="original", help="Calisma modu")
    parser.add_argument("--feature-percent", type=float, default=20.0, help="Secilecek feature yuzdesi (or. 30)")
    parser.add_argument("--ranking-type", choices=["avg", "max"], default="avg", help="Filtered modda hangi ranking dosyasi kullanilsin")
    parser.add_argument("--dataset-name", type=str, default="breast_cancer_data.csv", help="Raw dataset dosya adı (or. parkinson_dataset.csv)")
    parser.add_argument("--target-column", type=str, default="diagnosis", help="Hedef kolon adı")
    parser.add_argument("--id-column", type=str, default="ID", help="ID kolon adı. Kullanmak istemezsen 'none' ver")

    args = parser.parse_args()
    main(
        mode=args.mode,
        feature_percent=args.feature_percent,
        ranking_type=args.ranking_type,
        dataset_name=args.dataset_name,
        target_column=args.target_column,
        id_column=args.id_column,
    )