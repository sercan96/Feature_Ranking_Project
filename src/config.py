""" Info: Config dosyası, projenin genel ayarlarını ve parametrelerini tutmak için kullanılır.
Merkezi ayarlar burada olur.

Örneğin:

- veri dosyası yolu
- target sütunu adı
- test size
- random seed
- model kayıt yolları

Bu dosya sayesinde her yerde aynı bilgileri tekrar tekrar yazmayız.
"""

from pathlib import Path


# Proje ana klasörü
BASE_DIR = Path(__file__).resolve().parent.parent

# Veri yolları
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "breast_cancer_data.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


# Model kayıt yolları
CNN_MODEL_DIR = BASE_DIR / "models" / "cnn"
AUTOENCODER_MODEL_DIR = BASE_DIR / "models" / "autoencoder"
BASELINE_MODEL_DIR = BASE_DIR / "models" / "baseline"

# Çıktı yolları
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

# Veri sütun bilgileri
TARGET_COLUMN = "diagnosis"
ID_COLUMN = "ID"

# Deney ayarları
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

BASELINE_MODEL_DIR = BASE_DIR / "models" / "baseline"
# Model kayıt yolları
BASELINE_MODEL_DIR = BASE_DIR / "models" / "baseline"
CNN_MODEL_DIR = BASE_DIR / "models" / "cnn"
AUTOENCODER_MODEL_DIR = BASE_DIR / "models" / "autoencoder"

def get_data(dataset_name: str = "breast_cancer_data.csv",model_name: str ="", folder: str = "raw") -> Path:
    if folder == "raw":
        return BASE_DIR / "data" / "raw" / dataset_name
    elif folder == "filtered_datasets":
        return BASE_DIR / "filtered_datasets" / model_name / dataset_name
    else:
        raise ValueError(f"Geçersiz folder: {folder}. 'raw' veya 'filtered_datasets' olmalı.")

from pathlib import Path

BASE_OUTPUT_DIR = Path("outputs")


def get_model_output_dir(model_name: str, dataset_name: str = "breast_cancer_data", subfolder: str = "") -> Path:
    """
    Model çıktıları için yol oluştur.
    
    Örnekler:
    - get_model_output_dir("cnn", "breast_cancer_data") → outputs/cnn/breast_cancer_data
    - get_model_output_dir("cnn", "breast_cancer_data", "reports") → outputs/cnn/breast_cancer_data/reports
    - get_model_output_dir("cnn", "breast_cancer_data", "metrics") → outputs/cnn/breast_cancer_data/metrics
    """
    if subfolder:
        output_dir = BASE_OUTPUT_DIR / model_name / dataset_name / subfolder
    else:
        output_dir = BASE_OUTPUT_DIR / model_name / dataset_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir