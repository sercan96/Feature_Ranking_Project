import os
import sys
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_selector import FeatureSelector, FeatureSelectionConfig
from src.config import get_model_output_dir, get_data
from pathlib import Path


def validate_feature_percent(feature_percent: float) -> float:
    if feature_percent <= 0 or feature_percent > 100:
        raise ValueError("feature-percent 0 ile 100 arasında olmalı (100 dahil).")
    return feature_percent


def main(feature_percent: float = 20.0):
    BASE_DIR = Path(__file__).resolve().parent.parent
    feature_percent = validate_feature_percent(feature_percent)
    selection_ratio = feature_percent / 100.0
    
    config = FeatureSelectionConfig(
        dataset_path=get_data(),
        ranking_max_path=str(get_model_output_dir("cnn", "breast_cancer_data", "reports") / "feature_weight_ranking_max.csv"),
        ranking_avg_path=str(get_model_output_dir("cnn", "breast_cancer_data", "reports") / "feature_weight_ranking_avg.csv"),
        # Çıktı: data/filtered_datasets/cnn/breast_cancer_data/reports/
        output_dir=str(BASE_DIR / "data" / "filtered_datasets" / "cnn" / "breast_cancer_data" / "reports"),
        label_column="diagnosis",
        excluded_columns=["id"],
        selection_ratio=selection_ratio,
        min_features=1
    )

    selector = FeatureSelector(config)
    selector.load_files()
    selector.create_both_datasets()
    
    # Test metrikleri göster
    print("\n" + "="*70)
    print("TEST METRİKLERİ")
    print("="*70)
    
    metrics_file = get_model_output_dir("autoencoder", "breast_cancer_data", "metrics") / "test_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        print("\nOrijinal Veri (31 feature) Test Metrikleri:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        print(f"⚠️  Metrikleri bulunamadı: {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature secim datasetlerini yuzdeye gore olusturur")
    parser.add_argument("--feature-percent", type=float, default=20.0, help="Secilecek feature yuzdesi (or. 30)")
    args = parser.parse_args()

    main(feature_percent=args.feature_percent)