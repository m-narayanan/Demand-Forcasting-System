import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing_modeling import (
    load_and_preprocess_data, 
    get_top_products, 
    prepare_time_series, 
    train_forecast_model, 
)

st.set_page_config(page_title="Demand Forecasting System", layout="wide")

@st.cache_data
def load_data():
    return load_and_preprocess_data()

df_transactions, df_products, df_customers = load_data()

st.title("Demand Forecasting")

top_products = get_top_products(df_transactions, df_products, by='quantity', n=10).drop_duplicates('StockCode')

# Move 'Top 10 Unique Products (by Quantity)' table to the left sidebar
with st.sidebar:
    st.subheader("Top 10 Unique Products (by Quantity)")
    top_products = top_products.reset_index(drop=True)
    top_products['Serial Number'] = range(1, len(top_products) + 1)
    st.dataframe(top_products[[ 'StockCode', 'Description', 'quantity']])

    # Move 'Select a Stock Code:' dropdown to the left sidebar
    selected_stock_code = st.selectbox(
        "Select a Stock Code:",
        top_products['StockCode'].tolist(),
        format_func=lambda x: f"{x} - {top_products[top_products['StockCode'] == x]['Description'].values[0]}"
    )

if selected_stock_code:
    selected_product = top_products[top_products['StockCode'] == selected_stock_code].iloc[0]
    st.header(f"Demand Overview for {selected_stock_code} - {selected_product['Description']}")

    time_series_data = prepare_time_series(df_transactions, selected_stock_code)

    if not time_series_data.empty:
        forecast_weeks = 15
        model, history, forecast = train_forecast_model(df_transactions, selected_stock_code, forecast_weeks)

        if model is not None:
            fig, ax = plt.subplots(figsize=(12, 6))
            train_size = len(history) - len(forecast)
            ax.plot(history.index[:train_size], history.values[:train_size], label='Train Actual Demand', color='blue')
            ax.plot(history.index[train_size:], history.values[train_size:], label='Test Actual Demand', color='green')
            ax.plot(forecast.index, forecast.values, label='Forecasted Demand', color='red')
            ax.set_title(f"Actual vs Predicted Demand for {selected_stock_code}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Demand")
            ax.legend()
            st.pyplot(fig)

            train_error = history.values[:train_size] - model.fittedvalues[:train_size]
            test_error = history.values[train_size:] - forecast.values[:len(history) - train_size]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            sns.histplot(train_error, kde=True, ax=ax1, color='blue')
            ax1.set_title("Training Error Distribution")
            ax1.set_xlabel("Error")

            sns.histplot(test_error, kde=True, ax=ax2, color='green')
            ax2.set_title("Testing Error Distribution")
            ax2.set_xlabel("Error")

            st.pyplot(fig)

            train_rmse = np.sqrt(np.mean(train_error**2))
            test_rmse = np.sqrt(np.mean(test_error**2))

            st.subheader("Model Evaluation")
            col1, col2 = st.columns(2)
            col1.metric("Training RMSE", f"{train_rmse:.2f}")
            col2.metric("Testing RMSE", f"{test_rmse:.2f}")

            forecast_df = pd.DataFrame({'Date': forecast.index, 'Forecasted_Demand': forecast.values})
            st.download_button(
                label="Download Forecast CSV",
                data=forecast_df.to_csv(index=False).encode('utf-8'),
                file_name=f"forecast_{selected_stock_code}.csv",
                mime="text/csv"
            )
        else:
            st.error(f"Failed to train the model for stock code {selected_stock_code}. Please check the data and try again.")
    else:
        st.warning(f"Not enough data for stock code {selected_stock_code} to perform forecasting.")
else:
    st.write("Please select a stock code to view the demand forecast.")
