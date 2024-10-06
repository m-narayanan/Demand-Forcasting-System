import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import sys

sys.setrecursionlimit(3000)

def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    df_transactions_01 = pd.read_csv('Transactional_data_retail_01.csv')
    df_transactions_02 = pd.read_csv('Transactional_data_retail_02.csv')
    df_products = pd.read_csv('ProductInfo.csv')
    df_customers = pd.read_csv('CustomerDemographics.csv')
    
    df_transactions = pd.concat([df_transactions_01, df_transactions_02], ignore_index=True)
    df_transactions['InvoiceDate'] = pd.to_datetime(df_transactions['InvoiceDate'], dayfirst=True, errors='coerce')
    df_transactions.dropna(subset=['StockCode', 'Quantity', 'InvoiceDate'], inplace=True)
    df_transactions = df_transactions[df_transactions['Quantity'] > 0]
    df_transactions['TotalPrice'] = df_transactions['Quantity'] * df_transactions['Price']
    
    print("Data preprocessing completed.")
    return df_transactions, df_products, df_customers

def get_top_products(df_transactions, df_products, by='quantity', n=10):
    print(f"Getting top {n} unique products by {by}...")
    
    # Merge transactions with product info to get descriptions
    df_merged = df_transactions.merge(df_products, on='StockCode', how='left')
    
    if by == 'quantity':
        top_products = df_merged.groupby(['StockCode', 'Description'])['Quantity'].sum().reset_index()
        top_products = top_products.rename(columns={'Quantity': by})
    elif by == 'revenue':
        top_products = df_merged.groupby(['StockCode', 'Description'])['TotalPrice'].sum().reset_index()
        top_products = top_products.rename(columns={'TotalPrice': by})
    else:
        raise ValueError("'by' parameter must be either 'quantity' or 'revenue'")

    # Get unique top products
    top_products = top_products.drop_duplicates(subset=['StockCode']).nlargest(n, by)
    
    print("Top unique products retrieved successfully.")
    return top_products

def prepare_time_series(df_transactions, stock_code):
    print(f"Preparing time series for stock code: {stock_code}")
    try:
        stock_data = df_transactions[df_transactions['StockCode'] == stock_code].copy()
        if stock_data.empty:
            print(f"No data found for stock code: {stock_code}.")
            return pd.Series()
        stock_data['InvoiceDate'] = pd.to_datetime(stock_data['InvoiceDate'])
        stock_data['Quantity'] = pd.to_numeric(stock_data['Quantity'], errors='coerce')
        stock_data = stock_data.dropna(subset=['Quantity'])
        weekly_sales = stock_data.groupby(pd.Grouper(key='InvoiceDate', freq='W'))['Quantity'].sum()
        return weekly_sales
    except Exception as e:
        print(f"Error in prepare_time_series: {e}")
        return pd.Series()

def train_forecast_model(df_transactions, stock_code, forecast_weeks):
    print(f"Training forecast model for stock code: {stock_code} for {forecast_weeks} weeks")
    try:
        weekly_sales = prepare_time_series(df_transactions, stock_code)
        if weekly_sales.empty:
            print(f"No sales data available for stock code: {stock_code}")
            return None, None, None

        # Determine the maximum number of weeks we can forecast
        max_forecast_weeks = min(forecast_weeks, len(weekly_sales) // 2)
        if max_forecast_weeks < 1:
            print(f"Insufficient data to forecast for stock code: {stock_code}")
            return None, None, None

        train_size = len(weekly_sales) - max_forecast_weeks
        train, test = weekly_sales[:train_size], weekly_sales[train_size:]

        print(f"Train size: {train_size}, Test size: {len(test)}")
        print("Fitting ARIMA model...")
        model = ARIMA(train, order=(1, 1, 0))
        results = model.fit()
        print("Model fitting completed.")

        forecast = results.get_forecast(steps=max_forecast_weeks)
        forecast_mean = forecast.predicted_mean
        print(f"Forecasting {max_forecast_weeks} weeks ahead: {forecast_mean}")

        return results, weekly_sales, forecast_mean

    except Exception as e:
        print(f"Error in train_forecast_model for stock code {stock_code}: {e}")
        return None, None, None

def evaluate_model(model, test_data):
    print("Evaluating model")
    try:
        if model is None or test_data.empty:
            print("Invalid model or empty test data.")
            return None

        predictions = model.get_forecast(steps=len(test_data))
        predictions_mean = predictions.predicted_mean
        rmse = np.sqrt(mean_squared_error(test_data, predictions_mean))
        print(f"Model evaluation completed. RMSE: {rmse}")
        return rmse
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None

def main():
    df_transactions, df_products, df_customers = load_and_preprocess_data()
    top_stock_codes = df_transactions['StockCode'].value_counts().index[:5]
    chosen_stock_code = top_stock_codes[0]
    print(f"Chosen StockCode for time series forecasting: {chosen_stock_code}")
    forecast_weeks = 15
    model, weekly_sales, forecast_mean = train_forecast_model(df_transactions, chosen_stock_code, forecast_weeks)
    if model is not None:
        rmse = evaluate_model(model, weekly_sales[-len(forecast_mean):])
        print(f"Final RMSE: {rmse}")
    else:
        print("Model training failed.")

if __name__ == "__main__":
    main()

    