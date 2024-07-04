import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

# Function to prepare data
def prepare_data(df):
    output_var = df['Adj Close']
    features = ['Open', 'High', 'Low', 'Volume']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    X = df[features]
    y = output_var
    return X, y

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Function to train LSTM model
def train_lstm_model(model, X_train, y_train):
    with st.spinner("Training the model..."):
        model.fit(X_train, y_train, epochs=100, batch_size=6, verbose=1, shuffle=False)
    return model

# Function to make predictions
def make_predictions(model, X_test):
    return model.predict(X_test)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

# Function to download stock data from Yahoo Finance
def download_stock_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)

# Streamlit application
def main():
    st.title('Stock Price Prediction with LSTM')

    # User input for stock symbol and date range
    company_name = st.text_input("Enter the stock name (e.g., AAPL for Apple Inc.):")
    start_date = st.date_input("Select start date:", pd.to_datetime('2020-01-01'))
    end_date = st.date_input("Select end date:", pd.to_datetime('2024-01-01'))

    if company_name:
        # Download stock data
        stock_data = download_stock_data(company_name, start_date, end_date)

        st.subheader('Data Summary')
        st.write(stock_data.head())

        st.subheader('Historical Adj Close Prices')
        st.line_chart(stock_data['Adj Close'])

        # Prepare data
        X, y = prepare_data(stock_data)

        # Split data using time series cross-validation
        timesplit = TimeSeriesSplit(n_splits=10)
        for train_index, test_index in timesplit.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Reshape data for LSTM model
        X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

        # Build and train LSTM model
        lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        lstm_model = train_lstm_model(lstm_model, X_train, y_train)

        # Make predictions
        y_pred = make_predictions(lstm_model, X_test)

        # Calculate metrics
        mse, r2 = calculate_metrics(y_test, y_pred)

        # Visualization
        st.subheader('Prediction vs True Adj Close Values')
        fig, ax = plt.subplots()
        ax.plot(y_test.values, label="True Value")
        ax.plot(y_pred, label="LSTM Value")
        ax.set_title('Prediction by LSTM')
        ax.set_xlabel('Time Scale')
        ax.set_ylabel('Scaled USD')
        ax.legend()
        st.pyplot(fig)

        # Display metrics
        st.subheader('Metrics')
        st.write("Mean Squared Error:", mse)
        st.write("R-squared:", r2)

if __name__ == "__main__":
    main()
