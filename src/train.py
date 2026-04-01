""""" Info: Train dosyası, modeli eğitmek için kullanılır."""

import joblib
from pathlib import Path


def save_sklearn_model(model, path: Path):
    """
    Scikit-learn modelini kaydeder.
    """
    joblib.dump(model, path)


def load_sklearn_model(path: Path):
    """
    Scikit-learn modelini yükler.
    """
    return joblib.load(path)