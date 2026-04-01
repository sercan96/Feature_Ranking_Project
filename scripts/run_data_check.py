import sys
import os

# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data, basic_info


def main():
    df = load_data()
    basic_info(df)


if __name__ == "__main__":
    main()