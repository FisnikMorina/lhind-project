from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import root_mean_squared_error as rmse

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

st.set_page_config(page_title="Store x Product Sales Dashboard", layout="wide")


@st.cache_data
def load_history():
    train = pd.read_csv(RAW / "train.csv", parse_dates=["Date"])
    test = pd.read_csv(RAW / "test.csv", parse_dates=["Date"])
    df = pd.concat([train, test], ignore_index=True)
    df["year"] = df.Date.dt.year
    return df


@st.cache_data
def load_stats():
    return pd.read_csv(PROCESSED / "series_descriptive_stats.csv")


@st.cache_data
def load_predictions(year: int):
    return pd.read_csv(PROCESSED / f"hybrid_predictions_{year}.csv", parse_dates=["Date"])


def filter_by_selection(df, stores, products, date_range):
    mask = df.store.isin(stores) & df["product"].isin(products)
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask &= df.Date.between(start, end)
    return df[mask]


def error_pct(actual, predicted):
    return (actual - predicted) / actual * 100


def anomaly_chart(df, date_col, actual_col, predicted_col, threshold):
    df = df.sort_values(date_col)
    err = error_pct(df[actual_col], df[predicted_col])
    anomalies = df[err.abs() > threshold]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_col], name="actual",
                              mode="lines", line=dict(color="steelblue", width=1)))
    fig.add_trace(go.Scatter(x=df[date_col], y=df[predicted_col], name="predicted",
                              mode="lines", line=dict(color="darkorange", width=2)))
    fig.add_trace(go.Scatter(x=anomalies[date_col], y=anomalies[actual_col], name=f"|error| > {threshold}%",
                              mode="markers", marker=dict(color="red", size=7, symbol="x")))
    fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig, anomalies


history = load_history()
stats = load_stats()

st.title("Store x Product Sales Forecast Dashboard")


st.sidebar.header("Filters")
all_stores = sorted(history.store.unique())
all_products = sorted(history["product"].unique())
selected_stores = st.sidebar.multiselect("Store", all_stores, default=all_stores)
selected_products = st.sidebar.multiselect("Product", all_products, default=all_products)

min_date, max_date = history.Date.min().date(), history.Date.max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)

if not selected_stores or not selected_products:
    st.warning("Select at least one store and one product in the sidebar.")
    st.stop()

filtered = filter_by_selection(history, selected_stores, selected_products, date_range)
filtered_stats = stats[stats.store.isin(selected_stores) & stats["product"].isin(selected_products)]

tab_trends, tab_explorer, tab_predictions = st.tabs(
    ["Trends & Seasonality", "Store / Product Explorer", "Predictions vs Actuals"]
)

with tab_trends:
    st.subheader("Daily total sales")
    daily = filtered.groupby("Date").number_sold.sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily.values, name="daily total",
                              line=dict(color="steelblue", width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=daily.index, y=daily.rolling(90, center=True).mean(),
                              name="90-day average", line=dict(color="darkorange", width=2.5)))
    fig.update_layout(height=380, yaxis_title="units sold per day")
    st.plotly_chart(fig, use_container_width=True, key="trends_daily_total")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Year-over-year total sales")
        yearly = filtered.groupby("year").number_sold.sum()
        yoy = pd.DataFrame({"total_units": yearly, "yoy_pct": yearly.pct_change() * 100}).round(2)
        st.dataframe(yoy, use_container_width=True, key="trends_yoy_table")

    with col2:
        st.subheader("Day-of-week pattern")
        dow = filtered.groupby(filtered.Date.dt.dayofweek).number_sold.mean()
        dow = dow / dow.mean() * 100 - 100
        dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig = px.bar(dow, labels={"index": "", "value": "deviation from normal day (%)"})
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="trends_day_of_week")

    st.subheader("Mean sales heatmap (store x product)")
    pivot = filtered.pivot_table(index="store", columns="product", values="number_sold", aggfunc="mean")
    fig = px.imshow(pivot, text_auto=".0f", color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(height=280)
    st.plotly_chart(fig, use_container_width=True, key="trends_store_product_heatmap")

    st.subheader("Monthly seasonality by pair (index, 100 = that pair's average day)")
    month_pivot = filtered.pivot_table(index=["store", "product"], columns=filtered.Date.dt.month,
                                        values="number_sold", aggfunc="mean")
    month_pivot = month_pivot.div(month_pivot.mean(axis=1), axis=0) * 100
    month_pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_pivot.index = [f"store {s} - product {p}" for s, p in month_pivot.index]
    fig = px.imshow(month_pivot, color_continuous_scale="RdBu_r", color_continuous_midpoint=100, aspect="auto")
    fig.update_layout(height=max(300, 18 * len(month_pivot)))
    st.plotly_chart(fig, use_container_width=True, key="trends_monthly_seasonality_heatmap")

with tab_explorer:
    st.subheader("Descriptive statistics per store-product pair")
    st.dataframe(filtered_stats.sort_values("mean", ascending=False), use_container_width=True,
                 key="explorer_stats_table")

    st.subheader("Mean sales vs. day-to-day variability")
    fig = px.scatter(filtered_stats, x="mean", y="variation_coefficient_pct",
                      color="growth_2010_2018_pct", color_continuous_scale="RdBu",
                      color_continuous_midpoint=0, hover_data=["store", "product"],
                      labels={"mean": "mean units / day", "variation_coefficient_pct": "variation coefficient (%)",
                              "growth_2010_2018_pct": "growth 2010-2018 (%)"})
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True, key="explorer_variability_scatter")

    st.subheader("Single pair viewer")
    pair_options = list(filtered_stats[["store", "product"]].itertuples(index=False, name=None))
    if pair_options:
        chosen = st.selectbox("Store - product pair", pair_options,
                               format_func=lambda p: f"store {p[0]} - product {p[1]}",
                               key="explorer_pair_select")
        pair_series = history[(history.store == chosen[0]) & (history["product"] == chosen[1])].set_index("Date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pair_series.index, y=pair_series.number_sold,
                                  line=dict(color="steelblue", width=1), opacity=0.5, name="daily"))
        fig.add_trace(go.Scatter(x=pair_series.index, y=pair_series.number_sold.rolling(30).mean(),
                                  line=dict(color="darkorange", width=2), name="30-day average"))
        fig.update_layout(height=380, yaxis_title="units sold per day")
        st.plotly_chart(fig, use_container_width=True, key="explorer_pair_chart")

with tab_predictions:
    year_choice = st.radio("Evaluation year", [2018, 2019], index=1, horizontal=True,
                            help="2018 = validation year, 2019 = final held-out test year",
                            key="predictions_year")
    preds = load_predictions(year_choice)
    preds_filtered = filter_by_selection(preds, selected_stores, selected_products, date_range)

    if preds_filtered.empty:
        st.warning("No predictions for the current filter selection.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Base MAPE", f"{mape(preds_filtered.number_sold, preds_filtered.base) * 100:.2f}%")
        col2.metric("Hybrid MAPE", f"{mape(preds_filtered.number_sold, preds_filtered.hybrid) * 100:.2f}%")
        col3.metric("Base RMSE", f"{rmse(preds_filtered.number_sold, preds_filtered.base):.1f}")
        col4.metric("Hybrid RMSE", f"{rmse(preds_filtered.number_sold, preds_filtered.hybrid):.1f}")

        view = st.radio("View", ["Aggregated (sum of selected pairs)", "Single pair"], horizontal=True,
                         key="predictions_view")
        if view == "Single pair":
            pair_options = sorted(set(zip(preds_filtered.store, preds_filtered["product"])))
            chosen = st.selectbox("Store - product pair", pair_options,
                                   format_func=lambda p: f"store {p[0]} - product {p[1]}",
                                   key="predictions_pair_select")
            series = preds_filtered[(preds_filtered.store == chosen[0]) & (preds_filtered["product"] == chosen[1])]
        else:
            series = preds_filtered.groupby("Date")[["number_sold", "base", "hybrid"]].sum().reset_index()

        threshold = st.slider("Anomaly threshold (|error| %)", 1, 20, 5, key="predictions_threshold")
        fig, anomalies = anomaly_chart(series, "Date", "number_sold", "hybrid", threshold)
        st.subheader("Actual vs. hybrid prediction")
        st.plotly_chart(fig, use_container_width=True, key="predictions_anomaly_chart")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Error distribution")
            err = error_pct(preds_filtered.number_sold, preds_filtered.hybrid)
            fig = px.histogram(err, nbins=60, labels={"value": "error (%), positive = predicted too low"})
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="predictions_error_hist")

        with col2:
            st.subheader("MAPE by pair")
            pair_mape = (preds_filtered.groupby(["store", "product"])
                         .apply(lambda g: mape(g.number_sold, g.hybrid) * 100, include_groups=False)
                         .rename("hybrid_MAPE").reset_index().sort_values("hybrid_MAPE", ascending=False))
            st.dataframe(pair_mape, use_container_width=True, height=340, key="predictions_pair_mape_table")

        st.subheader(f"Days flagged as anomalies (|error| > {threshold}%)")
        if anomalies.empty:
            st.write("None in the current selection.")
        else:
            st.dataframe(anomalies, use_container_width=True, key="predictions_anomaly_table")

        importance_path = FIGURES / "test_2019_feature_importance.png"
        if importance_path.exists():
            with st.expander("LightGBM feature importance (final tuned model, precomputed)"):
                st.image(str(importance_path))
