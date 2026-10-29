from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from src.preprocessing import load_data
from src.feature_engineering import create_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = create_features(load_data())

# Avoid using the future target or contemporaneous traffic volume as input.
excluded = {"date_time", "target", "traffic_volume"}
features = [c for c in df.columns if c not in excluded and df[c].dtype.kind in "biuf"]
X, y = df[features], df["target"]

# Keep the final 20% of observations untouched for testing.
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=42, n_jobs=-1),
    "ANN": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=250, random_state=42))
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")

# Save the exact feature order and the final holdout data for evaluation.
joblib.dump(features, MODEL_DIR / "features.joblib")
joblib.dump({"X_test": X_test, "y_test": y_test, "dates": df.loc[X_test.index, "date_time"]}, MODEL_DIR / "test_data.joblib")
print(f"Finished training. Models saved to {MODEL_DIR}")