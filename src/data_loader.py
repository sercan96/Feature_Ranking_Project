""""" Info: Data loader dosyası, veri setini yüklemek ve temel veri kontrolü yapmak için kullanılır.
"""

import pandas as pd
from src.config import RAW_DATA_PATH


def load_data() -> pd.DataFrame:
    """
    Excel dosyasından breast cancer veri setini yükler. 
    """
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def basic_info(df: pd.DataFrame) -> None:
    """
    Veri setinin temel bilgilerini ekrana yazdırır.
    """
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