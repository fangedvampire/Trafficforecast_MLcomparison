
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.preprocessing import load_data

# --------------------------------------------------
# PROJECT PATHS AND THEME
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "reports" / "figures"

NAVY = "#14283F"
TEAL = "#12A89D"
BLUE = "#3977C3"
PALETTE = [TEAL, BLUE, "#7B61A8", "#E5A64B", "#D96868"]

st.set_page_config(
    page_title="TrafficCast | Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROFESSIONAL CSS
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #14283F;
        --teal: #12A89D;
        --background: #F4F7FB;
        --text: #25364A;
        --muted: #6B7A90;
        --border: #E3EAF2;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--navy) !important;
        letter-spacing: -0.5px;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        background: #14283F;
        border-right: 1px solid #263D55;
    }

    [data-testid="stSidebar"] * {
        color: #F4F8FC;
    }

    [data-testid="stSidebar"] hr {
        border-color: #344A60;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 19px 20px;
        box-shadow: 0 3px 12px rgba(20, 40, 63, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.86rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--navy);
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }

    [data-testid="stTabs"] [role="tablist"] {
        gap: 8px;
        border-bottom: 1px solid var(--border);
    }

    [data-testid="stTabs"] button[role="tab"] {
        border-radius: 9px 9px 0 0;
        padding: 12px 18px;
        font-weight: 600;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--teal);
        border-bottom: 3px solid var(--teal);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 8px;
        box-shadow: 0 3px 12px rgba(20, 40, 63, 0.025);
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    .hero {
        background:
            linear-gradient(90deg, rgba(13, 31, 51, 0.96), rgba(13, 31, 51, 0.72)),
            url("https://images.unsplash.com/photo-1449824913935-59a10b8d2000?auto=format&fit=crop&w=2000&q=85");
        background-size: cover;
        background-position: center 55%;
        border-radius: 18px;
        padding: 36px 38px;
        margin-bottom: 26px;
        color: white;
        min-height: 205px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 10px 28px rgba(20, 40, 63, 0.14);
    }

    .hero h1 {
        color: white !important;
        font-size: 2.35rem;
        margin: 0 0 8px 0;
        font-weight: 800;
    }

    .hero p {
        color: #DCE8F3;
        font-size: 1.02rem;
        max-width: 720px;
        margin: 0;
        line-height: 1.7;
    }

    .eyebrow {
        color: #70E0D3;
        text-transform: uppercase;
        font-size: 0.76rem;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .section-caption {
        color: var(--muted);
        font-size: 0.94rem;
        margin-top: -7px;
        margin-bottom: 20px;
    }

    .status-pill {
        display: inline-block;
        color: #087E75;
        background: #DDF7F2;
        border: 1px solid #BDECE4;
        padding: 5px 11px;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 700;
    }

    .footer {
        border-top: 1px solid var(--border);
        margin-top: 35px;
        padding-top: 18px;
        color: var(--muted);
        font-size: 0.82rem;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 25px 22px;
            min-height: 180px;
        }
        .hero h1 {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def get_data():
    return load_data()

try:
    df = get_data()
except Exception as exc:
    st.error(f"Unable to load the traffic dataset: {exc}")
    st.stop()

if df.empty:
    st.error("The traffic dataset contains no usable records.")
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">Traffic intelligence platform</div>
    <h1>TrafficCast</h1>
    <p>
        Understand traffic patterns, compare forecasting models, and
        explore data-driven insights for smarter transportation planning.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# KPI SUMMARY
# --------------------------------------------------
total_records = len(df)
average_traffic = df["traffic_volume"].mean()
peak_hour = int(
    df.groupby(df["date_time"].dt.hour)["traffic_volume"]
      .mean()
      .idxmax()
)
date_start = df["date_time"].min().strftime("%d %b %Y")
date_end = df["date_time"].max().strftime("%d %b %Y")

st.markdown("### At a glance")
st.markdown(
    '<p class="section-caption">A summary of the historical traffic dataset.</p>',
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Total observations", f"{total_records:,}")

with k2:
    st.metric("Average traffic volume", f"{average_traffic:,.0f}")

with k3:
    st.metric("Busiest average hour", f"{peak_hour:02d}:00")

with k4:
    st.metric("Dataset coverage", f"{date_start} – {date_end}")

st.markdown("")

# --------------------------------------------------
# NAVIGATION TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Traffic Explorer",
    "Model Performance",
    "Forecast & Explainability"
])

# Consistent Plotly styling

PLOT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="DM Sans, sans-serif", color=NAVY, size=12),
    title_font=dict(family="Manrope, sans-serif", size=17, color=NAVY),
    margin=dict(l=25, r=25, t=65, b=30),
    hoverlabel=dict(bgcolor="#14283F", bordercolor="#12A89D",font=dict( family="DM Sans, sans-serif",size=13,color="#FFFFFF" ),align="left", namelength=-1),
    legend=dict( orientation="h",yanchor="bottom",y=1.02, xanchor="right",x=1)
)


# --------------------------------------------------
# TAB 1: TRAFFIC EXPLORER
# --------------------------------------------------
with tab1:
    st.subheader("Historical traffic patterns")
    st.markdown(
        '<p class="section-caption">Explore hourly demand, weekly trends, and changes over time.</p>',
        unsafe_allow_html=True
    )

    hourly = (
        df.groupby(df["date_time"].dt.hour)["traffic_volume"]
        .mean()
        .reset_index()
    )
    hourly.columns = ["Hour", "Average traffic volume"]

    daily = (
        df.assign(day=df["date_time"].dt.dayofweek)
        .groupby("day")["traffic_volume"]
        .mean()
        .reindex(range(7))
        .reset_index()
    )
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    daily["Day"] = daily["day"].map(dict(enumerate(day_names)))

    left, right = st.columns(2)

    with left:
        fig = px.line(
            hourly,
            x="Hour",
            y="Average traffic volume",
            markers=True,
            title="Average Traffic by Hour",
            color_discrete_sequence=[TEAL]
        )
        fig.update_traces(line_width=3, marker_size=6)
        fig.update_layout(**PLOT_LAYOUT)
        fig.update_xaxes(dtick=2, title="Hour of day")
        fig.update_yaxes(title="Average traffic volume")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(
            daily,
            x="Day",
            y="traffic_volume",
            title="Average Traffic by Day of Week",
            color="traffic_volume",
            color_continuous_scale=["#C8F0EA", TEAL, NAVY]
        )
        fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Average traffic volume")
        st.plotly_chart(fig, use_container_width=True)

    heat = (
        df.assign(
            hour=df["date_time"].dt.hour,
            day=df["date_time"].dt.dayofweek
        )
        .pivot_table(
            index="day",
            columns="hour",
            values="traffic_volume",
            aggfunc="mean"
        )
        .reindex(range(7))
    )
    heat.index = day_names

    fig = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale=["#F0F7FC", "#71CFC4", "#087E75", NAVY],
        labels={
            "x": "Hour of day",
            "y": "Day of week",
            "color": "Traffic volume"
        },
        title="Traffic Intensity: Day vs Hour"
    )
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    recent = df.tail(500)
    fig = px.line(
        recent,
        x="date_time",
        y="traffic_volume",
        title="Recent Traffic Observations",
        color_discrete_sequence=[BLUE]
    )
    fig.update_traces(line_width=1.8)
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_xaxes(title="Date and time")
    fig.update_yaxes(title="Traffic volume")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# TAB 2: MODEL PERFORMANCE
# --------------------------------------------------
with tab2:
    st.subheader("Model performance comparison")
    st.markdown(
        '<p class="section-caption">Compare forecasting errors and goodness of fit on the held-out test set.</p>',
        unsafe_allow_html=True
    )

    results_path = MODEL_DIR / "evaluation_results.csv"

    if results_path.exists():
        scores = pd.read_csv(results_path)
        required = {"model", "RMSE", "MAE", "R2"}

        if scores.empty or not required.issubset(scores.columns):
            st.warning(
                "The evaluation file is empty or has unexpected columns. "
                "Run the evaluation script again."
            )
        else:
            for col in ["RMSE", "MAE", "R2"]:
                scores[col] = pd.to_numeric(scores[col], errors="coerce")

            scores = scores.dropna(subset=["RMSE", "MAE", "R2"])

            if scores.empty:
                st.warning("No valid evaluation scores were found.")
            else:
                best_rmse = scores.loc[scores["RMSE"].idxmin()]
                best_r2 = scores.loc[scores["R2"].idxmax()]

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Models evaluated", len(scores))
                with m2:
                    st.metric("Lowest RMSE", f"{best_rmse['RMSE']:,.2f}")
                    st.caption(f"Model: {best_rmse['model']}")
                with m3:
                    st.metric("Highest R²", f"{best_r2['R2']:.4f}")
                    st.caption(f"Model: {best_r2['model']}")

                st.markdown("#### Evaluation results")
                st.dataframe(
                    scores.sort_values("RMSE").round(4),
                    use_container_width=True,
                    hide_index=True
                )

                metric = st.selectbox(
                    "Select evaluation metric",
                    ["RMSE", "MAE", "R2"],
                    format_func=lambda x: {
                        "RMSE": "RMSE — lower is better",
                        "MAE": "MAE — lower is better",
                        "R2": "R² — higher is better"
                    }[x]
                )

                ascending = metric != "R2"
                chart_data = scores.sort_values(metric, ascending=ascending)

                fig = px.bar(
                    chart_data,
                    x="model",
                    y=metric,
                    color="model",
                    text=metric,
                    title=f"Model Comparison: {metric}",
                    color_discrete_sequence=PALETTE
                )
                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                    cliponaxis=False
                )
                fig.update_layout(
                    **PLOT_LAYOUT,
                    showlegend=False,
                    xaxis_title="Model",
                    yaxis_title=metric
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "RMSE and MAE measure prediction error; lower is better. "
                    "R² measures explained variance; values closer to 1 are generally better."
                )
    else:
        st.warning(f"Evaluation results not found at: {results_path}")
        st.code("python -m src.evaluate_models", language="powershell")

# --------------------------------------------------
# TAB 3: FORECAST & EXPLAINABILITY
# --------------------------------------------------
with tab3:
    st.subheader("Forecast analysis")
    st.markdown(
        '<p class="section-caption">Inspect model predictions, error patterns, and feature importance.</p>',
        unsafe_allow_html=True
    )

    predictions_path = MODEL_DIR / "test_predictions.csv"

    if predictions_path.exists():
        preds = pd.read_csv(
            predictions_path,
            parse_dates=["date_time"]
        )

        if {"date_time", "actual"}.issubset(preds.columns):
            available = [
                c for c in preds.columns
                if c not in ["date_time", "actual"]
            ]

            if available:
                selected = st.selectbox(
                    "Choose forecasting model",
                    available
                )

                valid = preds.dropna(
                    subset=["date_time", "actual", selected]
                ).copy()

                if not valid.empty:
                    valid["error"] = valid[selected] - valid["actual"]
                    valid["absolute_error"] = valid["error"].abs()

                    a, b, c = st.columns(3)
                    with a:
                        st.metric(
                            "Test observations",
                            f"{len(valid):,}"
                        )
                    with b:
                        st.metric(
                            "Prediction MAE",
                            f"{valid['absolute_error'].mean():,.2f}"
                        )
                    with c:
                        st.metric(
                            "Mean signed error",
                            f"{valid['error'].mean():,.2f}"
                        )

                    view = valid.tail(500)
                    chart = view.melt(
                        id_vars=["date_time"],
                        value_vars=["actual", selected],
                        var_name="Series",
                        value_name="Traffic volume"
                    )

                    fig = px.line(
                        chart,
                        x="date_time",
                        y="Traffic volume",
                        color="Series",
                        color_discrete_map={
                            "actual": NAVY,
                            selected: TEAL
                        },
                        title="Actual vs Predicted Traffic"
                    )
                    fig.update_layout(**PLOT_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                    left, right = st.columns(2)

                    with left:
                        fig = px.histogram(
                            valid,
                            x="error",
                            nbins=45,
                            title="Prediction Error Distribution",
                            color_discrete_sequence=[BLUE]
                        )
                        fig.add_vline(
                            x=0,
                            line_dash="dash",
                            line_color="#D96868"
                        )
                        fig.update_layout(**PLOT_LAYOUT)
                        fig.update_xaxes(title="Prediction error")
                        st.plotly_chart(fig, use_container_width=True)

                    with right:
                        fig = px.scatter(
                            valid.tail(1500),
                            x="actual",
                            y=selected,
                            title="Actual vs Predicted Values",
                            color_discrete_sequence=[TEAL],
                            opacity=0.55
                        )
                        low = min(
                            valid["actual"].min(),
                            valid[selected].min()
                        )
                        high = max(
                            valid["actual"].max(),
                            valid[selected].max()
                        )
                        fig.add_shape(
                            type="line",
                            x0=low, y0=low,
                            x1=high, y1=high,
                            line=dict(color="#D96868", dash="dash")
                        )
                        fig.update_layout(**PLOT_LAYOUT)
                        fig.update_xaxes(title="Actual traffic")
                        fig.update_yaxes(title="Predicted traffic")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid predictions are available for this model.")
            else:
                st.warning("No model prediction columns were found.")
        else:
            st.warning(
                "The prediction file must contain 'date_time' and 'actual' columns."
            )
    else:
        st.info(
            "Prediction results are not available yet. "
            "Run the model evaluation script first."
        )

    st.markdown("---")
    st.subheader("Model explainability")
    st.markdown(
        '<p class="section-caption">Explore which input features contribute to the trained model predictions.</p>',
        unsafe_allow_html=True
    )

    image_files = [
        ("shap_summary.png", "SHAP Summary"),
        ("feature_importance.png", "Feature Importance")
    ]

    image_columns = st.columns(2)
    found_images = 0

    for i, (filename, title) in enumerate(image_files):
        image_path = PLOT_DIR / filename

        if image_path.exists():
            with image_columns[i]:
                st.markdown(f"#### {title}")
                st.image(str(image_path), use_container_width=True)
            found_images += 1

    if found_images == 0:
        st.info(
            "Explainability plots have not been generated yet. "
            "Run your explainability script to create them."
        )

# --------------------------------------------------
# SIDEBAR AND FOOTER
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🚦 TrafficCast")
    st.caption("Traffic intelligence & forecasting")
    st.divider()

    st.markdown("### Dashboard")
    st.markdown("**Traffic Explorer**")
    st.caption("Historical patterns and traffic intensity")
    st.markdown("**Model Performance**")
    st.caption("Forecast accuracy and model comparison")
    st.markdown("**Explainability**")
    st.caption("Prediction errors and feature insights")

    st.divider()
    st.markdown("### Dataset information")
    st.write(f"**Records:** {total_records:,}")
    st.write(f"**Start:** {date_start}")
    st.write(f"**End:** {date_end}")

    st.divider()
    st.markdown(
        '<span class="status-pill">Historical data mode</span>',
        unsafe_allow_html=True
    )
    st.caption(
        "Research prototype. Forecasts and model metrics are based on "
        "the available dataset and held-out test observations."
    )

st.markdown("""
<div class="footer">
    <strong>TrafficCast</strong> · Traffic Analytics & Machine Learning
    <br>
    Research dashboard for historical traffic analysis and forecasting.
</div>
""", unsafe_allow_html=True)
