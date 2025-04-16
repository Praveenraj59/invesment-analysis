import lstm_model
import torch

import matplotlib.pyplot as plt  # Make sure this is at the top of your file
import os
import matplotlib.pyplot as plt

def compare_models(ticker):
    model_classical_path = f'saved_models/{ticker}_lstm.h5'
    model_quantum_path = f'saved_models/{ticker}_quantum_lstm.h5'

    # 🚀 Train Classical Model only if not already trained
    if not os.path.exists(model_classical_path):
        print(f"\n--- Training Classical LSTM Model for {ticker} ---")
        train_lstm(ticker)
    else:
        print(f"\n✅ Classical LSTM model found. Skipping training.")

    # 🚀 Predict using Classical Model
    print(f"\n--- Evaluating Classical LSTM Model for {ticker} ---")
    _, actual_prices_lstm, predicted_prices_lstm = predict_stock_price(ticker)

    # 🚀 Train Quantum Model only if not already trained
    if not os.path.exists(model_quantum_path):
        print(f"\n--- Training Quantum Optimized LSTM Model for {ticker} ---")
        train_with_quantum_optimization(ticker)
    else:
        print(f"\n✅ Quantum LSTM model found. Skipping training.")

    # 🚀 Predict using Quantum Model
    print(f"\n--- Evaluating Quantum Optimized LSTM Model for {ticker} ---")
    model = load_model(model_quantum_path)
    scaler = joblib.load(f'saved_models/{ticker}_scaler.pkl')

    stock_data = get_stock_data(ticker)
    time_step = 60
    data = stock_data['Close'].values.reshape(-1, 1)
    data = scaler.transform(data)

    X_test = [data[i - time_step:i, 0] for i in range(time_step, len(data))]
    X_test = np.array(X_test).reshape(len(X_test), time_step, 1)

    predictions_q = scaler.inverse_transform(model.predict(X_test)).flatten()
    actual_q = stock_data['Close'][-len(predictions_q):]

    # 📊 Model Evaluation
    mae_q = mean_absolute_error(actual_q, predictions_q)
    mse_q = mean_squared_error(actual_q, predictions_q)
    rmse_q = np.sqrt(mse_q)
    r2_q = r2_score(actual_q, predictions_q)

    print(f"\n🔍 Quantum Model Evaluation for {ticker}:")
    print(f"MAE: {mae_q:.4f}")
    print(f"MSE: {mse_q:.4f}")
    print(f"RMSE: {rmse_q:.4f}")
    print(f"R-squared: {r2_q:.4f}")

    print(f"\n📊 Summary Comparison for {ticker}:")
    print(f"{'Metric':<10} {'Classical':>15} {'Quantum':>15}")
    print(f"{'-'*40}")
    print(f"{'MAE':<10} {mean_absolute_error(actual_prices_lstm, predicted_prices_lstm):>15.4f} {mae_q:>15.4f}")
    print(f"{'MSE':<10} {mean_squared_error(actual_prices_lstm, predicted_prices_lstm):>15.4f} {mse_q:>15.4f}")
    print(f"{'RMSE':<10} {np.sqrt(mean_squared_error(actual_prices_lstm, predicted_prices_lstm)):>15.4f} {rmse_q:>15.4f}")
    print(f"{'R2':<10} {r2_score(actual_prices_lstm, predicted_prices_lstm):>15.4f} {r2_q:>15.4f}")

    # 📊 Bar Chart for Model Error Comparison
    metrics = ["MAE", "MSE", "RMSE", "R²"]
    classical_values = [
        mean_absolute_error(actual_prices_lstm, predicted_prices_lstm),
        mean_squared_error(actual_prices_lstm, predicted_prices_lstm),
        np.sqrt(mean_squared_error(actual_prices_lstm, predicted_prices_lstm)),
        r2_score(actual_prices_lstm, predicted_prices_lstm),
    ]
    quantum_values = [mae_q, mse_q, rmse_q, r2_q]

    x = np.arange(len(metrics))  
    width = 0.35  

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, classical_values, width, label="Classical LSTM", color="blue")
    bars2 = ax.bar(x + width/2, quantum_values, width, label="Quantum LSTM", color="green")

    ax.set_ylabel("Error Value")
    ax.set_title(f"Error Comparison: Classical vs Quantum LSTM for {ticker}")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.show()
    
    
    # 📉 Residual Plot: Actual vs Prediction Errors
    residuals_classical = actual_prices_lstm - predicted_prices_lstm
    residuals_quantum = actual_q - predictions_q

    plt.figure(figsize=(12, 5))
    plt.plot(residuals_classical, label="Classical LSTM Residuals", color="blue", linestyle="dotted")
    plt.plot(residuals_quantum, label="Quantum LSTM Residuals", color="green", linestyle="dashed")

    plt.axhline(y=0, color="black", linestyle="--")
    plt.title(f"Residuals of Predictions for {ticker}")
    plt.xlabel("Time Steps")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(actual_prices_lstm, predicted_prices_lstm, label="Classical LSTM", color="blue", alpha=0.5)
    plt.scatter(actual_q, predictions_q, label="Quantum LSTM", color="green", alpha=0.5)

    plt.plot([min(actual_q), max(actual_q)], [min(actual_q), max(actual_q)], color="red", linestyle="--")  # Ideal Line
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"Actual vs Predicted Prices for {ticker}")
    plt.legend()
    plt.grid(True)
    plt.show()
        

if __name__ == "__main__":
    compare_models("AAPL")