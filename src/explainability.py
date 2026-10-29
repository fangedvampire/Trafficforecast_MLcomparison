from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "reports" / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

test = joblib.load(MODEL_DIR / "test_data.joblib")
X_test = test["X_test"]
model = joblib.load(MODEL_DIR / "xgboost.joblib")

explainer = shap.TreeExplainer(model)
sample = X_test.sample(min(500, len(X_test)), random_state=42)
values = explainer(sample)

shap.plots.beeswarm(values, max_display=15, show=False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()

importance = pd.Series(model.feature_importances_, index=X_test.columns).sort_values()
importance.tail(15).plot(kind="barh", figsize=(9, 6), title="XGBoost Feature Importance")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance.png", dpi=200)
plt.close()

print(f"Saved explainability plots to {PLOT_DIR}")