import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_selector import FeatureSelector, FeatureSelectionConfig
from src.config import get_model_output_dir, get_raw_data_path


def main():
    config = FeatureSelectionConfig(
        dataset_path=get_raw_data_path(),
        ranking_max_path=str(get_model_output_dir("cnn", "breast_cancer_data", "reports") / "feature_weight_ranking_max.csv"),
        ranking_avg_path=str(get_model_output_dir("cnn", "breast_cancer_data", "reports") / "feature_weight_ranking_avg.csv"),
        output_dir=f"filtered_datasets/cnn/breast_cancer_data/reports",
        label_column="diagnosis",
        excluded_columns=["id"],
        selection_ratio=0.20,
        min_features=1
    )

    selector = FeatureSelector(config)
    selector.load_files()
    selector.create_both_datasets()


if __name__ == "__main__":
    main()