from src.models import build_cnn_model
from src.config import CNN_MODEL_DIR


def main():
    model = build_cnn_model()
    model.load_weights(CNN_MODEL_DIR / "cnn_best.weights.h5")

    print("\n--- CNN Best Weights Inspection ---")

    for layer in model.layers:
        weights = layer.get_weights()
        print(f"\nLayer: {layer.name}")

        if not weights:
            print("Bu katmanda ağırlık yok.")
            continue

        for i, w in enumerate(weights):
            print(f"Weight {i+1} shape: {w.shape}")


if __name__ == "__main__":
    main()