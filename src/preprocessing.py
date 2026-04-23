""""" Info: Preprocessing dosyası, veri setini ön işlemek için kullanılır.

Adımlar:
1.Veriyi temizlemek
2.ID sütununu çıkarmak
3.diagnosis sütununu sayısallaştırmak
4.X ve y ayırmak
5.Train/test split yapmak
6.Scaling uygulamak
7.CNN için reshape etmek
8.İşlenmiş veriyi kaydetmek

"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import TARGET_COLUMN, ID_COLUMN, TEST_SIZE, RANDOM_STATE


def drop_id_column(df: pd.DataFrame, id_column: str | None = ID_COLUMN) -> pd.DataFrame:
    """
    Eğer ID sütunu varsa veri setinden kaldırır.
    """
    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])
    return df

#df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
def encode_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """
    diagnosis sütununu sayısal hale getirir.
    M -> 1
    B -> 0
    """
    df = df.copy()
    if target_column not in df.columns:
        raise ValueError(f"Target kolonu bulunamadı: {target_column}")

    y = df[target_column]

    if y.dtype == bool:
        df[target_column] = y.astype(int)
        return df

    if pd.api.types.is_numeric_dtype(y):
        # Sayısal target'ta sınıfları olduğu gibi koru; sadece int'e çevir.
        df[target_column] = y.astype(int)
        return df

    y_str = y.astype(str).str.strip()
    unique_labels = sorted(pd.Series(y_str).dropna().unique().tolist())

    if set(unique_labels) == {"B", "M"}:
        df[target_column] = y_str.map({"M": 1, "B": 0})
        return df

    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    df[target_column] = y_str.map(label_map).astype(int)
    return df


def split_features_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN):
    """
    Girdi özelliklerini (X) ve hedef değişkeni (y) ayırır.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target kolonu bulunamadı: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def sanitize_mixed_type_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    String formatta gelen sayisal degerleri temizleyip sayisala cevirir.
    - Virgullu ondaliklari noktaya cevirir (1,23 -> 1.23)
    - Bos/metin bozugu degerleri NaN yapar
    - Tamami NaN olan kolonlari atar
    - Kalan NaN degerleri kolon medyani ile doldurur
    """
    X_clean = X.copy()

    for col in X_clean.columns:
        series = X_clean[col]
        if pd.api.types.is_numeric_dtype(series):
            continue

        s = series.astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan, "N/A": np.nan, "?": np.nan})
        s = s.str.replace(",", ".", regex=False)
        X_clean[col] = pd.to_numeric(s, errors="coerce")

    # Sayisal gorunen tiplerde de kalan metin olabilir; son bir coercion uygula.
    X_clean = X_clean.apply(pd.to_numeric, errors="coerce")

    all_nan_cols = [col for col in X_clean.columns if X_clean[col].isna().all()]
    if all_nan_cols:
        X_clean = X_clean.drop(columns=all_nan_cols)

    if X_clean.shape[1] == 0:
        raise ValueError("Sayısal feature kolonu bulunamadı. Data dosyasını kontrol edin.")

    if X_clean.isna().any().any():
        X_clean = X_clean.fillna(X_clean.median(numeric_only=True))

    # Nadir durumda median da NaN kalirsa 0 ile tamamla.
    if X_clean.isna().any().any():
        X_clean = X_clean.fillna(0.0)

    return X_clean


def keep_numeric_features_only(X: pd.DataFrame) -> pd.DataFrame:
    """
    Sayısal olmayan feature kolonlarını otomatik olarak çıkarır.
    Örn: sample_id gibi string kolonlar.
    """
    return sanitize_mixed_type_features(X)


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int | None = RANDOM_STATE):
    """
    Veriyi train ve test olarak böler.
    stratify=y kullanarak sınıf dağılımını korur.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y # %63 benign %37 malignant ise train ve testte de buna yakın oran korunur. 
    )
    return X_train, X_test, y_train, y_test


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    StandardScaler ile veriyi ölçekler.
    Sadece X_train üzerinde fit yapılır. 0.24 leri 0-1 arasına getirir. 
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def reshape_for_cnn(X: np.ndarray):
    """
    1D CNN için veriyi (örnek_sayısı, özellik_sayısı, 1) formatına çevirir.
    """
    return X.reshape(X.shape[0], X.shape[1], 1)


def preprocess_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_column: str | None = ID_COLUMN,
    random_state: int | None = RANDOM_STATE,
):
    """
    Tüm preprocessing adımlarını sırasıyla uygular.
    """
    df = drop_id_column(df, id_column=id_column)
    df = encode_target(df, target_column=target_column)

    X, y = split_features_target(df, target_column=target_column)
    X = keep_numeric_features_only(X)
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=random_state)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    X_train_cnn = reshape_for_cnn(X_train_scaled)
    X_test_cnn = reshape_for_cnn(X_test_scaled)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_cnn": X_train_cnn,
        "X_test_cnn": X_test_cnn,
        "scaler": scaler,
    }