import pandas as pd

def create_features(df):
    df = df.copy().sort_values("date_time").reset_index(drop=True)
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["day_of_year"] = df["date_time"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    for lag in [1, 2, 3, 24, 48, 168]:
        df[f"traffic_lag_{lag}"] = df["traffic_volume"].shift(lag)

    df["traffic_rolling_24h"] = df["traffic_volume"].shift(1).rolling(24).mean()
    df["traffic_rolling_168h"] = df["traffic_volume"].shift(1).rolling(168).mean()

    # Predict the traffic volume one hour after the input timestamp.
    df["target"] = df["traffic_volume"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    return df