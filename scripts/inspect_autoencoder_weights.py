from src.models import build_autoencoder
from src.config import AUTOENCODER_MODEL_DIR


def main():
    autoencoder, encoder = build_autoencoder(input_dim=30, encoding_dim=8)
    autoencoder.load_weights(AUTOENCODER_MODEL_DIR / "autoencoder_best.weights.h5")

    print("\n--- Autoencoder Best Weights Inspection ---")

    for layer in autoencoder.layers:
        weights = layer.get_weights()
        print(f"\nLayer: {layer.name}")

        if not weights:
            print("Bu katmanda ağırlık yok.")
            continue

        for i, w in enumerate(weights):
            print(f"Weight {i+1} shape: {w.shape}")


if __name__ == "__main__":
    main()