from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from src.preprocessing import load_data
from src.feature_engineering import create_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "reports" / "figures"

st.set_page_config(page_title="TrafficCast", page_icon="🚦", layout="wide")
st.title(" TrafficCast: Traffic Forecasting & Analytics")
st.caption("Explore traffic patterns, compare ML models, and analyze one-hour-ahead predictions.")

@st.cache_data
def get_data():
    return load_data()

df = get_data()
tab1, tab2, tab3 = st.tabs(["Traffic Explorer", "Model Comparison", "Forecast & Explainability"])

with tab1:
    st.subheader("Historical traffic patterns")
    hourly = df.groupby(df["date_time"].dt.hour)["traffic_volume"].mean().reset_index()
    hourly.columns = ["Hour", "Average traffic volume"]
    st.plotly_chart(px.line(hourly, x="Hour", y="Average traffic volume", markers=True, title="Average Traffic by Hour"), use_container_width=True)

    heat = df.assign(hour=df["date_time"].dt.hour, day=df["date_time"].dt.dayofweek).pivot_table(index="day", columns="hour", values="traffic_volume", aggfunc="mean")
    st.plotly_chart(px.imshow(heat, aspect="auto", labels={"x": "Hour", "y": "Day of week", "color": "Traffic"}, title="Traffic Heatmap"), use_container_width=True)

    st.plotly_chart(px.line(df.tail(500), x="date_time", y="traffic_volume", title="Recent 500 Hourly Observations"), use_container_width=True)


with tab2:
    results_path = MODEL_DIR / "evaluation_results.csv"
    if results_path.exists():
        scores = pd.read_csv(results_path)
        st.dataframe(scores, use_container_width=True, hide_index=True)
        metric = st.selectbox("Compare metric", ["RMSE", "MAE", "R2"])
        st.plotly_chart(px.bar(scores.sort_values(metric, ascending=(metric != "R2")), x="model", y=metric, color="model", title=f"Model Comparison — {metric}"), use_container_width=True)
    else:
        st.info("Train and evaluate the models first using the commands in the setup instructions.")

with tab3:
    predictions_path = MODEL_DIR / "test_predictions.csv"
    if predictions_path.exists():
        preds = pd.read_csv(predictions_path, parse_dates=["date_time"])
        available = [c for c in preds.columns if c not in ["date_time", "actual"]]
        selected = st.selectbox("Choose a model", available)
        view = preds.tail(500)
        chart = view.melt(id_vars=["date_time"], value_vars=["actual", selected], var_name="Series", value_name="Traffic volume")
        st.plotly_chart(px.line(chart, x="date_time", y="Traffic volume", color="Series", title="Actual vs Predicted Traffic"), use_container_width=True)
        preds["error"] = preds[selected] - preds["actual"]
        st.plotly_chart(px.histogram(preds, x="error", nbins=50, title="Prediction Error Distribution"), use_container_width=True)
    else:
        st.info("Run model training and evaluation to enable prediction analysis.")

    for image_name in ["shap_summary.png", "feature_importance.png"]:
        image_path = PLOT_DIR / image_name
        if image_path.exists():
            st.image(str(image_path), caption=image_name.replace("_", " ").replace(".png", "").title(), use_container_width=True)

st.sidebar.header("About")
st.sidebar.write("TrafficCast is a historical traffic analytics research prototype. Forecast performance is evaluated on a chronological holdout set.")