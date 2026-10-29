
from pathlib import Path
import pandas as pd

# Project root: trafficcast/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Find CSV files in the data directory
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV file found in {DATA_DIR}. "
        "Place the downloaded dataset inside the data folder."
    )

print("CSV files found:")
for i, file in enumerate(csv_files, start=1):
    print(f"{i}. {file.name}")

# Load the first CSV file
df = pd.read_csv(csv_files[0])

print("\n--- DATASET OVERVIEW ---")
print(f"File: {csv_files[0].name}")
print(f"Shape: {df.shape}")

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- FIRST FIVE ROWS ---")
print(df.head().to_string())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- NUMERICAL SUMMARY ---")
print(df.describe(include="all").T)
