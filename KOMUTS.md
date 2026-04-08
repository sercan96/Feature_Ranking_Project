CNN original
python scripts/run_train_cnn.py --mode original --dataset-name breast_cancer_data.csv --target-column diagnosis --id-column none --feature-percent 20

CNN filtered
python scripts/run_train_cnn.py --mode filtered --dataset-name breast_cancer_data.csv --target-column diagnosis --id-column none --feature-percent 20 --ranking-type avg

Autoencoder original
python scripts/run_train_autoencoder.py --mode original --dataset-name breast_cancer_data.csv --target-column diagnosis --id-column none --feature-percent 20

Autoencoder filtered
python scripts/run_train_autoencoder.py --mode filtered --dataset-name breast_cancer_data.csv --target-column diagnosis --id-column none --feature-percent 20 --ranking-type max