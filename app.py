import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def get_stock_data(ticker_symbol, start_date, end_date):
    """Download stock data using yfinance"""
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return stock_data

def prepare_data(df):
    """Prepare data for prediction"""
    # Create features from previous days' prices
    df['Next_Day_Price'] = df['Close'].shift(-1)
    df['Prev_Day_Price'] = df['Close'].shift(1)
    df['2_Day_Price'] = df['Close'].shift(2)
    df['3_Day_Price'] = df['Close'].shift(3)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Create feature matrix X and target vector y
    X = df[['Close', 'Prev_Day_Price', '2_Day_Price', '3_Day_Price', 'Volume']]
    y = df['Next_Day_Price']
    
    return X, y

# Streamlit app
st.title('Stock Price Predictor')

# User inputs
ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, GOOGL):', 'AAPL')
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

if st.button('Predict'):
    try:
        # Get data
        df = get_stock_data(ticker, start_date, end_date)
        
        if df.empty:
            st.error('No data found for the ticker symbol')
        else:
            # Prepare data
            X, y = prepare_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make prediction for next day
            last_data = X.iloc[-1].values.reshape(1, -1)
            prediction = model.predict(last_data)[0]
            
            # Convert prices to float for proper formatting
            current_price = float(df["Close"].iloc[-1])
            predicted_price = float(prediction)
            price_change = ((predicted_price - current_price) / current_price * 100)
            
            # Display results
            st.write(f'Current Price: ${current_price:.2f}')
            st.write(f'Predicted Next Day Price: ${predicted_price:.2f}')
            st.write(f'Predicted Change: {price_change:.2f}%')
            
            # Display model accuracy
            accuracy = model.score(X_test, y_test)
            st.write(f'Model Accuracy (R² Score): {accuracy:.2f}')
            
    except Exception as e:
        st.error(f'An error occurred: {str(e)}. Please check the ticker symbol and try again.')