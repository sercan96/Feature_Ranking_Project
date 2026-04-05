""""" Info: Data loader dosyası, veri setini yüklemek ve temel veri kontrolü yapmak için kullanılır.
"""

import pandas as pd
from src.config import get_data


def load_data(dataset_name: str = "breast_cancer_data.csv", model_name: str = "", dataset_name_folder: str = "", folder: str = "raw") -> pd.DataFrame:
    df = pd.read_csv(get_data(dataset_name, model_name=model_name, dataset_name_folder=dataset_name_folder, folder=folder))
    return df


def basic_info(df: pd.DataFrame) -> None:
    print("\n--- İlk 5 Satır ---")
    print(df.head())

    print("\n--- Shape ---")
    print(df.shape)

    print("\n--- Sütunlar ---")
    print(df.columns.tolist())

    print("\n--- Eksik Veri Sayıları ---")
    print(df.isnull().sum())

    print("\n--- Veri Tipleri ---")
    print(df.dtypes)