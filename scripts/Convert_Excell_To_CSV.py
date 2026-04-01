import sys
import os

# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

df = pd.read_excel("data/raw/breast_cancer_data.xlsx")
df.to_csv("data/raw/breast_cancer_data.csv", index=False)