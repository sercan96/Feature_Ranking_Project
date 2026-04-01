""""" Info: Bu dosyalar, projenin farklı aşamalarında kullanılan temel bileşenleri içerir. Her dosya, belirli bir görevi yerine getirmek için tasarlanmıştır ve projenin genel yapısını oluşturur."""

from pathlib import Path
import json


def ensure_dir(path: Path) -> None:
    """
    Verilen klasör yoksa oluşturur.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    """
    Dictionary verisini JSON olarak kaydeder.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)