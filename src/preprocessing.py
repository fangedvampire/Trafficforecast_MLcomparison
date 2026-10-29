from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_PATH = next(DATA_DIR.glob("*.csv"), None)

def load_data():
    if DATA_PATH is None:
        raise FileNotFoundError("Place the traffic CSV inside the data folder.")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    if "date_time" not in df.columns or "traffic_volume" not in df.columns:
        raise ValueError("Expected date_time and traffic_volume columns.")

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = df.dropna(subset=["date_time", "traffic_volume"])
    df = df.drop_duplicates(subset="date_time")
    df = df.sort_values("date_time").reset_index(drop=True)

    return df