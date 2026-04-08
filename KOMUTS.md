CNN :
    python scripts/run_train_cnn.py --mode original --feature-percent 30
    python scripts/run_train_cnn.py --mode filtered --feature-percent 30 --ranking-type avg
AUTOENCODER :
    python scripts/run_train_autoencoder.py --mode original --feature-percent 20
    python scripts/run_train_autoencoder.py --mode filtered --feature-percent 20 --ranking-type avg
