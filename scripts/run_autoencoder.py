import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Proje kokunu import path'ine ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RANDOM_STATE
from src.data_loader import convert_txt_dataset_to_csv, load_data
from src.models import build_sigmoid_autoencoder, build_latent_classifier
from src.preprocessing import preprocess_data
from src.autoencoder_feature_selection import (
	save_top_percent_features_by_abs_max_weight,
	save_filtered_dataset_from_selected_features,
	validate_feature_percent,
)
from src.utils import ensure_dir, save_json

EPOCHS = 50
BATCH_SIZE = 16
CLASSIFIER_VALIDATION_SPLIT = 0.1
THRESHOLD = 0.5


def set_reproducible(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	tf.keras.utils.set_random_seed(seed)
	try:
		tf.config.experimental.enable_op_determinism()
	except Exception:
		pass


def parse_random_state(value: str | None) -> int | None:
	if value is None:
		return RANDOM_STATE
	text = str(value).strip().lower()
	if text in {"none", "null", "random", "-"}:
		return None
	return int(text)


def save_feature_weighted_lists(autoencoder, X_train: np.ndarray, feature_names: list[str], output_path: Path) -> None:
	"""
	Her feature icin bagli oldugu nöronlara sample-bazli katkı listesi uretir:
	contribution_list_i[j] = mean_s( abs(x_s,i * w_i,j) )
	"""
	first_encoder_layer = None
	for layer_name in ("enc_dense_1", "encoder_dense_1"):
		try:
			first_encoder_layer = autoencoder.get_layer(layer_name)
			break
		except ValueError:
			continue

	if first_encoder_layer is None:
		raise ValueError(
			"Encoder ilk katmanı bulunamadı. Beklenen isimler: 'enc_dense_1' veya 'encoder_dense_1'. "
			f"Mevcut katmanlar: {[layer.name for layer in autoencoder.layers]}"
		)

	weights = first_encoder_layer.get_weights()[0]  # (n_features, n_neurons)
	if X_train.ndim != 2:
		raise ValueError(f"X_train 2 boyutlu olmali, gelen shape: {X_train.shape}")

	if weights.shape[0] != X_train.shape[1]: # Ağırlık(Satır sayısı) = Feature sayısı
		raise ValueError(
			f"X_train feature boyutu ({X_train.shape[1]}) ile agirlik satir sayisi ({weights.shape[0]}) eslesmiyor."
		)

	# (n_samples, n_features, 1) * (1, n_features, n_neurons)
	# -> (n_samples, n_features, n_neurons)
	contributions = np.abs(X_train[:, :, np.newaxis] * weights[np.newaxis, :, :])
	weighted = np.mean(contributions, axis=0)  # (n_features, n_neurons)

	df = pd.DataFrame(
		{
			"feature": [f"F{i+1}" for i in range(weighted.shape[0])],
			"weight_list": [weighted[i].tolist() for i in range(weighted.shape[0])],
		}
	)
	df.to_csv(output_path, index=False)


def normalize_id_column(id_column: str | None) -> str | None:
	if id_column and id_column.lower() in {"none", "null", "-", ""}:
		return None
	return id_column


def format_feature_percent_tag(feature_percent: float) -> str:
	if float(feature_percent).is_integer():
		return str(int(feature_percent))
	return str(feature_percent).replace(".", "_")


def unpack_processed_arrays(processed: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	X_train = processed["X_train_scaled"]
	X_test = processed["X_test_scaled"]
	y_train = processed["y_train"].to_numpy().astype(np.float32)
	y_test = processed["y_test"].to_numpy().astype(np.float32)
	return X_train, X_test, y_train, y_test


def train_and_evaluate_pipeline(
	X_train: np.ndarray,
	X_test: np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
	encoding_dim: int,
) -> tuple[float, float, tf.keras.Model, tf.keras.Model]:
	autoencoder, encoder = build_sigmoid_autoencoder(
		input_dim=X_train.shape[1],
		encoding_dim=encoding_dim,
	)

	autoencoder.fit(
		X_train,
		X_train,
		validation_data=(X_test, X_test),
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		shuffle=True,
		verbose=1,
	)

	test_mse = float(autoencoder.evaluate(X_test, X_test, verbose=0))

	X_train_encoded = encoder.predict(X_train, verbose=0)
	X_test_encoded = encoder.predict(X_test, verbose=0)

	classifier = build_latent_classifier(input_dim=X_train_encoded.shape[1])
	classifier.fit(
		X_train_encoded,
		y_train,
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		validation_split=CLASSIFIER_VALIDATION_SPLIT,
		verbose=1,
	)

	y_pred_prob = classifier.predict(X_test_encoded, verbose=0)
	y_pred = (y_pred_prob > THRESHOLD).astype(int).ravel()
	test_accuracy = float(accuracy_score(y_test.astype(int), y_pred))

	return test_mse, test_accuracy, autoencoder, encoder


def main(
	dataset_name: str = "breast_cancer_data.csv",
	target_column: str = "target",
	id_column: str | None = "ID",
	encoding_dim: int = 8,
	feature_percent: float = 50.0,
	random_state: int | None = RANDOM_STATE,
) -> None:
	if random_state is not None:
		set_reproducible(random_state)
	feature_percent = validate_feature_percent(feature_percent)
	id_column = normalize_id_column(id_column)

	dataset_filename = convert_txt_dataset_to_csv(dataset_name)
	dataset_folder = Path(dataset_filename).stem

	print(f"[INFO] Veri yukleniyor: {dataset_filename}")
	df = load_data(dataset_filename, folder="raw", target_column=target_column)

	processed = preprocess_data(
		df,
		target_column=target_column,
		id_column=id_column,
		random_state=random_state,
	)
	X_train_raw = processed["X_train"]
	X_train, X_test, y_train, y_test = unpack_processed_arrays(processed)

	print(f"[INFO] X_train shape: {X_train.shape}")
	print(f"[INFO] X_test shape : {X_test.shape}")

	test_mse, test_accuracy, autoencoder, _ = train_and_evaluate_pipeline(
		X_train,
		X_test,
		y_train,
		y_test,
		encoding_dim,
	)

	output_dir = Path("outputs") / "autoencoder" / dataset_folder
	metrics_dir = output_dir / "metrics"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)

	feature_names = X_train_raw.columns.tolist()
	weights_path = output_dir / "first_layer_W_list.csv"
	save_feature_weighted_lists(autoencoder, X_train, feature_names, weights_path)

	feature_percent_tag = format_feature_percent_tag(feature_percent)
	selected_features_path = output_dir / f"top_{feature_percent_tag}_max_abs_features.csv"
	selected_df = save_top_percent_features_by_abs_max_weight(
		weight_list_csv_path=weights_path,
		feature_names=feature_names,
		feature_percent=feature_percent,
		output_path=selected_features_path,
	)

	filtered_data_dir = Path("data") / "autoencoder" / dataset_folder
	ensure_dir(filtered_data_dir)
	filtered_dataset_path = filtered_data_dir / f"top_{feature_percent_tag}_max_abs_features_dataset.csv"
	filtered_df = save_filtered_dataset_from_selected_features(
		full_df=df,
		selected_df=selected_df,
		target_column=target_column,
		output_path=filtered_dataset_path,
		id_column=id_column,
	)

	filtered_train_df = pd.read_csv(filtered_dataset_path)
	processed_filtered = preprocess_data(
		filtered_train_df,
		target_column=target_column,
		id_column=id_column,
		random_state=random_state,
	)
	X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = unpack_processed_arrays(processed_filtered)
	filtered_test_mse, filtered_test_accuracy, _, _ = train_and_evaluate_pipeline(
		X_train_filtered,
		X_test_filtered,
		y_train_filtered,
		y_test_filtered,
		encoding_dim,
	)

	save_json(
		{
			"test_mse": test_mse,
			"test_accuracy": test_accuracy,
			"threshold": THRESHOLD,
		},
		metrics_dir / "ORG_test_metrics.json",
	)

	save_json(
		{
			"feature_percent": feature_percent,
			"selected_feature_count": len(selected_df),
			"test_mse": filtered_test_mse,
			"test_accuracy": filtered_test_accuracy,
			"threshold": THRESHOLD,
		},
		metrics_dir / f"top_{feature_percent_tag}_test_metrics.json",
	)

	print("\n[OK] Autoencoder egitimi tamamlandi.")
	print(f"[OK] test_mse: {test_mse:.6f}")
	print(f"[OK] test_accuracy: {test_accuracy:.6f}")
	print(f"[OK] Feature weighted listeleri: {weights_path}")
	print(f"[OK] Top %{feature_percent} secilen feature sayisi: {len(selected_df)}")
	print(f"[OK] Secilen feature CSV: {selected_features_path}")
	print(f"[OK] Filterlenmis dataset CSV: {filtered_dataset_path} (satir: {len(filtered_df)})")
	print(f"[OK] Top %{feature_percent} dataset test_mse: {filtered_test_mse:.6f}")
	print(f"[OK] Top %{feature_percent} dataset test_accuracy: {filtered_test_accuracy:.6f}")
	filtered_metrics_path = metrics_dir / f"top_{feature_percent_tag}_test_metrics.json"
	print(f"[OK] Top %{feature_percent} metrik dosyasi: {filtered_metrics_path}")
	print(f"[OK] Output klasoru: {output_dir}")
	print(f"[OK] Metrik dosyasi: {metrics_dir / 'ORG_test_metrics.json'}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Basit autoencoder egitimi")
	parser.add_argument("--dataset-name", type=str, default="breast_cancer_data.csv", help="Raw data dosyasi (.csv veya .txt)")
	parser.add_argument("--target-column", type=str, default="target", help="Hedef kolon adi")
	parser.add_argument("--id-column", type=str, default="ID", help="ID kolon adi, kullanmak istemezsen 'none' ver")
	parser.add_argument("--encoding-dim", type=int, default=8, help="Encoding boyutu")
	parser.add_argument("--feature-percent", type=float, default=20.0, help="Secilecek feature yuzdesi")
	parser.add_argument("--random-state", type=str, default=str(RANDOM_STATE), help="Sabit tohum icin sayi ver. Rastgele icin 'none' ver")


	args = parser.parse_args()
	main(
		dataset_name=args.dataset_name,
		target_column=args.target_column,
		id_column=args.id_column,
		encoding_dim=args.encoding_dim,
		feature_percent=args.feature_percent,
		random_state=parse_random_state(args.random_state),
	)
