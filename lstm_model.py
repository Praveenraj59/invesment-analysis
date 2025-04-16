import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
import pennylane as qml
from pennylane.optimize import AdamOptimizer
import torch

# Create directory for saved models if it doesn't exist
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

scaler = MinMaxScaler(feature_range=(0, 1))

# Fetch historical stock data from Yahoo Finance
def get_stock_data(ticker, start_date='2018-01-01', end_date=None):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data returned for {ticker}")
            return None
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Prepare data for LSTM
def prepare_data(data, time_step=60):
    data = data['Close'].values.reshape(-1, 1)
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])

    return np.array(X), np.array(y)

# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the LSTM model
def train_lstm(ticker):
    stock_data = get_stock_data(ticker)
    if stock_data is None or stock_data.empty:
        print(f"No data available for {ticker}")
        return None

    time_step = 60
    X, y = prepare_data(stock_data, time_step)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    model.save(f'saved_models/{ticker}_lstm.h5')
    joblib.dump(scaler, f'saved_models/{ticker}_scaler.pkl')

    # Predict the stock prices using the model
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    # Evaluate the model using metrics
    actual_prices = scaler.inverse_transform(y.reshape(-1, 1))  # Reversing the scaling on y
    mae = mean_absolute_error(actual_prices, predictions)
    mse = mean_squared_error(actual_prices, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_prices, predictions)

    print(f"Model Evaluation for {ticker}:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")

    return stock_data
def create_metrics_graph(mae, mse, rmse, r2, ticker):
    metrics_fig = go.Figure(data=[
        go.Bar(name='MAE', x=[ticker], y=[mae], marker=dict(color='red')),
        go.Bar(name='MSE', x=[ticker], y=[mse], marker=dict(color='blue')),
        go.Bar(name='RMSE', x=[ticker], y=[rmse], marker=dict(color='green')),
        go.Bar(name='R²', x=[ticker], y=[r2], marker=dict(color='orange')),
    ])

    metrics_fig.update_layout(
        title=f"Model Evaluation Metrics for {ticker}",
        xaxis_title="Metrics",
        yaxis_title="Value",
        barmode='group',
        plot_bgcolor='rgba(240,240,240,0.9)'
    )
    return metrics_fig

# Function to create prediction vs actual graph
def create_prediction_graph(actual_prices, predicted_prices, ticker):
    prediction_fig = go.Figure()
    prediction_fig.add_trace(go.Scatter(x=range(len(actual_prices)), y=actual_prices, mode='lines', name='Actual Price', line=dict(color='blue')))
    prediction_fig.add_trace(go.Scatter(x=range(len(predicted_prices)), y=predicted_prices, mode='lines', name='Predicted Price', line=dict(color='red')))
    
    prediction_fig.update_layout(
        title=f"{ticker} - Prediction vs Actual Prices",
        xaxis_title="Time",
        yaxis_title="Stock Price",
    )
    return prediction_fig


# Modify the predict_stock_price function to include evaluation
def predict_stock_price(ticker):
    model_path = f'saved_models/{ticker}_lstm.h5'
    scaler_path = f'saved_models/{ticker}_scaler.pkl'

    stock_data = get_stock_data(ticker)
    if stock_data is None or stock_data.empty:
        print(f"No data available for {ticker}")
        return None, "No stock data available.", None

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Training model for {ticker} as no saved model was found.")
        stock_data = train_lstm(ticker)
        if stock_data is None or stock_data.empty:
            return None, "Failed to train model or get data.", None

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, "Error loading saved model or scaler.", None

    data = stock_data['Close'].values.reshape(-1, 1)
    data = scaler.transform(data)

    time_step = 60
    if len(data) <= time_step:
        return None, "Not enough data for prediction.", None

    X_test = []
    for i in range(time_step, len(data)):
        X_test.append(data[i-time_step:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    actual_prices = stock_data['Close'][-len(predictions):]
    predicted_prices = predictions.flatten()

    # Evaluation metrics on predictions
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_prices, predicted_prices)

    print(f"Model Evaluation for {ticker}:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")


    predicted_price = float(predicted_prices[-1])  # Ensures it’s a single float
    


    return predicted_price, actual_prices, predicted_prices, mae, mse, rmse, r2


# Quantum optimization using PennyLane
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def quantum_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[2], wires=2)
    return qml.expval(qml.PauliZ(0))  # Measurement on qubit 0

# Hybrid loss function that combines classical and quantum losses
def hybrid_loss_function(predictions, targets, quantum_params):
    mse_loss = torch.mean((predictions - targets) ** 2)
    quantum_loss = quantum_circuit(quantum_params)
    return mse_loss + quantum_loss

# Training loop with quantum optimization
def train_with_quantum_optimization(ticker):
    stock_data = get_stock_data(ticker)
    if stock_data is None or stock_data.empty:
        print(f"No data available for {ticker}")
        return None

    time_step = 60
    X, y = prepare_data(stock_data, time_step)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build the LSTM model
    model = build_lstm_model((X.shape[1], 1))
    
    # Initialize quantum optimizer
    quantum_params = np.random.randn(3)
    quantum_optimizer = AdamOptimizer(stepsize=0.1)

    # Train the model
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    # Quantum optimization step
    for epoch in range(10):
        predictions = model.predict(X)
        loss = hybrid_loss_function(torch.tensor(predictions), torch.tensor(y), quantum_params)
        quantum_params = quantum_optimizer.step(quantum_circuit, quantum_params)

        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
    
    # Save model
    model.save(f'saved_models/{ticker}_quantum_lstm.h5')
    joblib.dump(scaler, f'saved_models/{ticker}_scaler.pkl')

    return model