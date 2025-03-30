# loading the data using the pandas, the data path is set using the config module
import pandas as pd
from config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Type 2", "Type 3", "Type 4"])
    return df
