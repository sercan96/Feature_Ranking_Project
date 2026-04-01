import sys
import os
from pathlib import Path
import joblib
import numpy as np
from src.config import PROCESSED_DATA_DIR
from src.utils import ensure_dir


# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data
from src.preprocessing import preprocess_data


def save_processed_data(processed: dict):
    """
    İşlenmiş veri ve scaler nesnesini kaydeder.
    """
    ensure_dir(PROCESSED_DATA_DIR)

    np.save(PROCESSED_DATA_DIR / "X_train_scaled.npy", processed["X_train_scaled"])
    np.save(PROCESSED_DATA_DIR / "X_test_scaled.npy", processed["X_test_scaled"])
    np.save(PROCESSED_DATA_DIR / "X_train_cnn.npy", processed["X_train_cnn"])
    np.save(PROCESSED_DATA_DIR / "X_test_cnn.npy", processed["X_test_cnn"])

    np.save(PROCESSED_DATA_DIR / "y_train.npy", processed["y_train"].to_numpy())
    np.save(PROCESSED_DATA_DIR / "y_test.npy", processed["y_test"].to_numpy())

    joblib.dump(processed["scaler"], PROCESSED_DATA_DIR / "scaler.pkl")

def main():
    df = load_data()
    processed = preprocess_data(df)
    save_processed_data(processed)
    print("\n--- X_train shape ---")
    print(processed["X_train"].shape)

    print("\n--- X_test shape ---")
    print(processed["X_test"].shape)

    print("\n--- y_train shape ---")
    print(processed["y_train"].shape)

    print("\n--- y_test shape ---")
    print(processed["y_test"].shape)

    print("\n--- X_train_scaled shape ---")
    print(processed["X_train_scaled"].shape)

    print("\n--- X_test_scaled shape ---")
    print(processed["X_test_scaled"].shape)

    print("\n--- X_train_cnn shape ---")
    print(processed["X_train_cnn"].shape)

    print("\n--- X_test_cnn shape ---")
    print(processed["X_test_cnn"].shape)

    print("\n--- y_train sınıf dağılımı ---")
    print(processed["y_train"].value_counts())

    print("\n--- y_test sınıf dağılımı ---")
    print(processed["y_test"].value_counts())


if __name__ == "__main__":
    main()

