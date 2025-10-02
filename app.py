import streamlit as st
import pandas as pd

from components.data_loader   import load_file
from components.data_cleaner  import impute_missing, remove_outliers_iqr
from components.visualizer    import plot_histogram, plot_boxplot, plot_scatter, plot_heatmap
from components.model_trainer import train_regression
from utils.helpers            import split_features_target

st.set_page_config(page_title="DS Toolbox", layout="wide")

# ---------- File upload ----------
uploaded_file = st.file_uploader(
    "Upload CSV/Excel file",
    type=["csv", "xlsx"],
    accept_multiple_files=False,
)

if uploaded_file:
    df, sep_used = load_file(uploaded_file)
    st.write(f"Separator detected: **{sep_used}**")
    st.dataframe(df.head())

    # ---------- Data cleaning ----------
    with st.expander("Data Cleaning"):
        impute_method = st.selectbox(
            "Impute missing values",
            ["None", "Mean", "Median", "Mode"],
            key="impute",
        )
        df_clean = df.copy()
        if impute_method != "None":
            df_clean = impute_missing(df_clean, method=impute_method.lower())

        outlier_option = st.checkbox("Remove IQR outliers (numeric columns)", key="outliers")
        if outlier_option:
            df_clean = remove_outliers_iqr(df_clean)

        st.write("Cleaned Data:")
        st.dataframe(df_clean.head())

    # ---------- Visualisation ----------
    with st.expander("Visualisation"):
        viz_type = st.radio(
            "Select plot type",
            ["Histogram", "Boxplot", "Scatter", "Heatmap"],
            key="viz_type",
        )
        if viz_type in ("Histogram", "Boxplot"):
            col = st.selectbox("Column", df_clean.columns, key="col_sel")
            fig = (
                plot_histogram(df_clean, col)
                if viz_type == "Histogram"
                else plot_boxplot(df_clean, col)
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Scatter":
            x_col = st.selectbox("X-axis", df_clean.columns, key="x_sel")
            y_col = st.selectbox("Y-axis", df_clean.columns, key="y_sel")
            fig = plot_scatter(df_clean, x_col, y_col)
            st.plotly_chart(fig, use_container_width=True)

        else:  # Heatmap
            fig = plot_heatmap(df_clean)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- Modeling ----------
    with st.expander("Modeling (Regression)"):
        target = st.selectbox("Target column", df_clean.columns, key="target_sel")
        X, y = split_features_target(df_clean, target)

        numeric_X = X.select_dtypes(include=["number"])
        if not pd.api.types.is_numeric_dtype(y):
            st.warning("Target column is **not** numeric – regression skipped.")
        elif numeric_X.empty:
            st.warning("No numeric features available for regression.")
        else:
            model, preds, y_test, metrics = train_regression(numeric_X, y)

            st.write("Evaluation Metrics:")
            st.json(metrics)

            plot_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
            fig = plot_scatter(plot_df, "Actual", "Predicted")
            st.plotly_chart(fig, use_container_width=True)

