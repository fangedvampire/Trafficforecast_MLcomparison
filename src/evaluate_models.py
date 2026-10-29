from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
test = joblib.load(MODEL_DIR / "test_data.joblib")
X_test, y_test = test["X_test"], test["y_test"]

results, predictions = [], pd.DataFrame({"date_time": test["dates"].values, "actual": y_test.values})

for path in MODEL_DIR.glob("*.joblib"):
    if path.name in {"features.joblib", "test_data.joblib"}:
        continue
    model = joblib.load(path)
    pred = model.predict(X_test)
    results.append({"model": path.stem, "MAE": mean_absolute_error(y_test, pred), "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred)})
    predictions[path.stem] = pred

# Seasonal-naive benchmark: predict the next hour using traffic from 24 hours ago.
if "traffic_lag_24" in X_test:
    pred = X_test["traffic_lag_24"].to_numpy()
    results.append({"model": "seasonal_naive_24h", "MAE": mean_absolute_error(y_test, pred), "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred)})
    predictions["seasonal_naive_24h"] = pred

scores = pd.DataFrame(results).sort_values("RMSE")
scores.to_csv(MODEL_DIR / "evaluation_results.csv", index=False)
predictions.to_csv(MODEL_DIR / "test_predictions.csv", index=False)
print(scores.to_string(index=False))