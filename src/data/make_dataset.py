import os
import pandas as pd

from src.utils.config import load_config

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

filepath = os.path.join(PROJECT_DIR, "data", "train.csv")

def load_data():

    config = load_config()

    df = pd.read_csv(filepath, parse_dates=[config["timestamp"]])

    return df

if __name__ == "__main__":
    load_data()