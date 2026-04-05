from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import math
import pandas as pd


@dataclass
class FeatureSelectionConfig:
    dataset_path: str
    ranking_max_path: str
    ranking_avg_path: str
    output_dir: str
    label_column: Optional[str] = None
    id_column: Optional[str] = None
    excluded_columns: List[str] = field(default_factory=list)
    selection_ratio: float = 0.20
    min_features: int = 1


class FeatureSelector:
    """
    Ranking dosyalarına göre dataset'ten top-k feature seçip yeni CSV üretir.
    """

    def __init__(self, config: FeatureSelectionConfig) -> None:
        self.config = config
        self.dataset_path = Path(config.dataset_path)
        self.ranking_max_path = Path(config.ranking_max_path)
        self.ranking_avg_path = Path(config.ranking_avg_path)
        self.output_dir = Path(config.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df: Optional[pd.DataFrame] = None
        self.df_max: Optional[pd.DataFrame] = None
        self.df_avg: Optional[pd.DataFrame] = None

    def load_files(self) -> None:
        self.df = pd.read_csv(self.dataset_path)
        self.df_max = pd.read_csv(self.ranking_max_path)
        self.df_avg = pd.read_csv(self.ranking_avg_path)

    def _validate_loaded(self) -> None:
        if self.df is None or self.df_max is None or self.df_avg is None:
            raise ValueError("Önce load_files() çağrılmalı.")

    def _get_feature_columns(self) -> List[str]:
        """
        Sadece gerçek feature kolonlarını döndürür.
        Label ve excluded kolonlar çıkarılır.
        """
        self._validate_loaded()
        assert self.df is not None

        excluded = set(self.config.excluded_columns)

        if self.config.label_column:
            excluded.add(self.config.label_column)

        if self.config.id_column:
            excluded.add(self.config.id_column)

        feature_columns = [col for col in self.df.columns if col not in excluded]

        return feature_columns
    
    def _calculate_top_k(self, n_features: int) -> int:
        top_k = int(n_features * self.config.selection_ratio)
        return max(top_k, self.config.min_features)

    def _ranking_to_column_names(self, ranking_df: pd.DataFrame, top_k: int) -> List[str]:
        """
        Ranking dosyasındaki F1, F2, ... isimlerini gerçek dataset kolon isimlerine dönüştürür.
        """

        if "feature" not in ranking_df.columns:
            raise ValueError("Ranking dosyasında 'feature' kolonu bulunmalı.")

        selected_columns = ranking_df["feature_name"].head(top_k).tolist()

        return selected_columns

    def create_filtered_dataset(
        self,
        ranking_type: str = "max",
        save_filename: Optional[str] = None
    ) -> pd.DataFrame:
        self._validate_loaded()
        assert self.df is not None
        assert self.df_max is not None
        assert self.df_avg is not None

        feature_columns = self._get_feature_columns()
        top_k = self._calculate_top_k(len(feature_columns))

        if ranking_type == "max":
            ranking_df = self.df_max
            default_name = f"{self.dataset_path.stem}_top_{top_k}_max_features.csv"
        elif ranking_type == "avg":
            ranking_df = self.df_avg
            default_name = f"{self.dataset_path.stem}_top_{top_k}_avg_features.csv"
        else:
            raise ValueError("ranking_type sadece 'max' veya 'avg' olabilir.")

        selected_columns = self._ranking_to_column_names(ranking_df, top_k)

        ordered_columns = []

        # id en başta
        if self.config.id_column and self.config.id_column in self.df.columns:
            ordered_columns.append(self.config.id_column)

        # seçilen feature'lar ortada
        ordered_columns.extend(selected_columns)

        # label en sonda
        if self.config.label_column and self.config.label_column in self.df.columns:
            ordered_columns.append(self.config.label_column)

        filtered_df = self.df[ordered_columns].copy()

        output_name = save_filename if save_filename else default_name
        output_path = self.output_dir / output_name

        filtered_df.to_csv(output_path, index=False)

        print(f"\n[{ranking_type.upper()}] Yeni dataset oluşturuldu: {output_path}")
        print(f"Gerçek feature sayısı: {len(feature_columns)}")
        print(f"Seçilen feature sayısı: {top_k}")
        print("Seçilen feature kolonları:")
        for col in selected_columns:
            print(f" - {col}")

        if self.config.label_column:
            print(f"Eklenen target kolonu: {self.config.label_column}")

        return filtered_df

    def create_both_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        max_df = self.create_filtered_dataset("max")
        avg_df = self.create_filtered_dataset("avg")
        return max_df, avg_df