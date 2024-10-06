import pandas as pd
import numpy as np

def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    # Load datasets
    df_transactions_01 = pd.read_csv('Transactional_data_retail_01.csv')
    df_transactions_02 = pd.read_csv('Transactional_data_retail_02.csv')
    df_products = pd.read_csv('ProductInfo.csv')
    df_customers = pd.read_csv('CustomerDemographics.csv')
    
    # Combine transaction datasets
    df_transactions = pd.concat([df_transactions_01, df_transactions_02], ignore_index=True)
    
    # Convert InvoiceDate to datetime
    df_transactions['InvoiceDate'] = pd.to_datetime(df_transactions['InvoiceDate'])
    
    # Remove cancelled orders (those with negative quantities)
    df_transactions = df_transactions[df_transactions['Quantity'] > 0]
    
    # Calculate total price for each transaction
    df_transactions['TotalPrice'] = df_transactions['Quantity'] * df_transactions['Price']
    
    print("Data preprocessing completed.")
    return df_transactions, df_products, df_customers

def get_top_products(df_transactions, df_products, by='quantity', n=10):
    print(f"Getting top {n} products by {by}...")
    if by == 'quantity':
        top_products = df_transactions.groupby('StockCode')['Quantity'].sum().nlargest(n).reset_index()
    elif by == 'revenue':
        top_products = df_transactions.groupby('StockCode')['TotalPrice'].sum().nlargest(n).reset_index()
    else:
        raise ValueError("'by' parameter must be either 'quantity' or 'revenue'")
    
    # Merge with product info to get descriptions
    top_products = top_products.merge(df_products, on='StockCode', how='left')
    
    print("Top products retrieved successfully.")
    return top_products

def prepare_time_series(df_transactions, stock_code):
    print(f"Preparing time series data for stock code {stock_code}...")
    # Filter for the specific stock code
    df_product = df_transactions[df_transactions['StockCode'] == stock_code]
    
    # Aggregate daily sales
    daily_sales = df_product.groupby('InvoiceDate')['Quantity'].sum().reset_index()
    daily_sales.set_index('InvoiceDate', inplace=True)
    
    # Resample to weekly frequency
    weekly_sales = daily_sales.resample('W').sum()
    
    print("Time series data prepared successfully.")
    return weekly_sales

def get_customer_summary(df_transactions):
    print("Generating customer summary...")
    customer_summary = df_transactions.groupby('Customer ID').agg({
        'Invoice': 'count',
        'TotalPrice': 'sum'
    }).rename(columns={'Invoice': 'Total_Orders', 'TotalPrice': 'Total_Revenue'})
    print("Customer summary generated.")
    return customer_summary

def get_item_summary(df_transactions):
    print("Generating item summary...")
    item_summary = df_transactions.groupby('StockCode').agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum'
    }).rename(columns={'Quantity': 'Total_Quantity_Sold', 'TotalPrice': 'Total_Revenue'})
    print("Item summary generated.")
    return item_summary

def get_transaction_summary(df_transactions):
    print("Generating transaction summary...")
    transaction_summary = df_transactions.groupby('Invoice').agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum'
    }).rename(columns={'Quantity': 'Total_Items', 'TotalPrice': 'Total_Amount'})
    print("Transaction summary generated.")
    return transaction_summary

def get_monthly_sales(df_transactions):
    print("Calculating monthly sales...")
    monthly_sales = df_transactions.set_index('InvoiceDate').resample('M')['TotalPrice'].sum()
    print("Monthly sales calculated.")
    return monthly_sales