from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd


def validate_feature_percent(feature_percent: float) -> float:
    if feature_percent <= 0 or feature_percent > 100:
        raise ValueError("feature-percent 0 ile 100 arasında olmalı (100 dahil).")
    return feature_percent


def save_top_percent_features_by_abs_max_weight(
    weight_list_csv_path: Path,
    feature_names: list[str],
    feature_percent: float,
    output_path: Path,
) -> pd.DataFrame:
    """
    first_layer_W_list.csv içindeki weight_list alanından max(abs(...)) skorunu hesaplar,
    yüzdeye göre top-k feature seçer ve CSV olarak kaydeder.
    """
    feature_percent = validate_feature_percent(feature_percent)
    weight_df = pd.read_csv(weight_list_csv_path)

    if "feature" not in weight_df.columns or "weight_list" not in weight_df.columns:
        raise ValueError("weight list dosyasında 'feature' ve 'weight_list' kolonları bulunmalı.")

    if len(weight_df) != len(feature_names):
        raise ValueError(
            f"Feature sayisi ({len(feature_names)}) ile weight list satir sayisi ({len(weight_df)}) eslesmiyor."
        )

    feature_scores = weight_df["weight_list"].apply(
        lambda s: max(abs(x) for x in ast.literal_eval(s))
    )
    total_features = len(weight_df)
    top_k = max(math.ceil(total_features * (feature_percent / 100.0)), 1)

    feature_to_name = {f"F{i+1}": feature_names[i] for i in range(total_features)}

    ranking_df = (
        pd.DataFrame(
            {
                "feature": weight_df["feature"].astype(str),
                "feature_name": weight_df["feature"].astype(str).map(feature_to_name),
                "max_abs_weight_score": feature_scores,
            }
        )
        .sort_values(by="max_abs_weight_score", ascending=False)
        .reset_index(drop=True)
    )

    selected_df = ranking_df.head(top_k).copy()
    selected_df.to_csv(output_path, index=False)
    return selected_df


def save_filtered_dataset_from_selected_features(
    full_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    target_column: str,
    output_path: Path,
    id_column: str | None = None,
) -> pd.DataFrame:
    """
    Seçilen feature listesini kullanarak dataset'i filtreler ve CSV olarak kaydeder.
    Kolon sırası: id (varsa), seçilen feature'lar, target.
    """
    if "feature_name" not in selected_df.columns:
        raise ValueError("selected_df içinde 'feature_name' kolonu bulunmalı.")

    selected_features = selected_df["feature_name"].tolist()
    missing_features = [col for col in selected_features if col not in full_df.columns]
    if missing_features:
        raise ValueError(
            f"Seçilen feature'lar dataset içinde bulunamadı: {missing_features}"
        )

    if target_column not in full_df.columns:
        raise ValueError(f"Target kolonu dataset içinde bulunamadı: {target_column}")

    ordered_columns: list[str] = []
    if id_column and id_column in full_df.columns:
        ordered_columns.append(id_column)

    ordered_columns.extend(selected_features)
    ordered_columns.append(target_column)

    filtered_df = full_df[ordered_columns].copy()
    filtered_df.to_csv(output_path, index=False)
    return filtered_df
