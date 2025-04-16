from flask import Flask, render_template, request
import yfinance as yf
from lstm_model import predict_stock_price
from sentiment_analysis import get_stock_sentiment
import plotly.graph_objs as go
import json
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)

# Available stocks for dropdown


# Available stocks for dropdown (Indian stocks)
STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
   # "BHARTIARTL.NS": "Bharti Airtel",
    #"KOTAKBANK.NS": "Kotak Mahindra Bank",
    #"HINDUNILVR.NS": "Hindustan Unilever",
    #"LARSEN.NS": "Larsen & Toubro",
    #"M&M.NS": "Mahindra & Mahindra"
}

# Home page
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/model_evaluation", methods=["GET", "POST"])
def model_evaluation():
    try:
        ticker = request.form.get("ticker", "RELIANCE.NS")
        result = predict_stock_price(ticker)

        if result is None or len(result) != 7:
            return render_template("error.html", message="Prediction failed or model not trained.")

        predicted_price, actual_prices, predicted_prices, mae, mse, rmse, r2 = result

        return render_template("model_evaluation.html", 
                               ticker=ticker, 
                               mae=mae, 
                               mse=mse,
                               rmse=rmse, 
                               r2=r2, 
                               predicted_price=predicted_price)
    except Exception as e:
        return render_template("error.html", message=f"An error occurred: {str(e)}")
@app.route("/planning_to_invest", methods=["GET", "POST"])
def planning_to_invest():
    if request.method == "POST":
        try:
            ticker = request.form["ticker"]
            stock_name = STOCKS.get(ticker)
            investment_amount = float(request.form["investment_amount"])
            review_frequency = request.form["review_frequency"]
            risk_tolerance = request.form["risk_tolerance"]

            if not stock_name:
                return render_template("error.html", message="Invalid stock selection.")

            predicted_price, actual_prices, predicted_prices, mae, mse, rmse, r2 = predict_stock_price(ticker)

            if actual_prices is None or actual_prices.empty:
                return render_template("error.html", message=f"No data available for {stock_name}. Please try again later.")

            current_price = float(actual_prices.iloc[-1].iloc[0])

            if predicted_price is None or not isinstance(predicted_price, (int, float)):
                return render_template("error.html", message="Prediction data is not available. Please try again later.")

            # Calculate shares to buy and actual invested amount
            shares_to_buy = investment_amount // current_price
            actual_invested_amount = shares_to_buy * current_price
            remaining_cash = investment_amount - actual_invested_amount

            predicted_price = round(predicted_price, 2)
            predicted_investment_value = round(shares_to_buy * predicted_price, 2)
            potential_return = round(actual_invested_amount * (predicted_price / current_price), 2)

            # Sentiment Analysis
            sentiment, sentiment_score = get_stock_sentiment(stock_name)

            if sentiment_score > 0.2:
                sentiment_message = f"The market outlook for {stock_name} is optimistic, suggesting potential growth ahead."
            elif sentiment_score < -0.2:
                sentiment_message = f"The market sentiment for {stock_name} is cautious, indicating possible risks."
            else:
                sentiment_message = f"The market sentiment for {stock_name} is stable, with no strong signals in either direction."

            # 📊 Table Data
            table_data = {
                "Stock Name": stock_name,
                "Current Price (₹)": round(current_price, 2),
                "Predicted Price (₹)": predicted_price,
                "Investment Amount (₹)": round(actual_invested_amount, 2),  # <- actual invested
                "Shares to Buy": int(shares_to_buy),
                "Predicted Investment Value (₹)": predicted_investment_value,
                "Uninvested Amount (₹)": round(remaining_cash, 2)  # <- optional
            }

            # 🧠 Recommendation Logic
            recommendation = "Wait and Observe"
            if predicted_price > current_price:
                if risk_tolerance == "High":
                    recommendation = "Strong Buy"
                elif risk_tolerance == "Medium":
                    recommendation = "Buy"
                elif risk_tolerance == "Low":
                    recommendation = "Cautious Buy"

            # 📈 Comparative Analysis
            comparison_data = []
            for ticker_symbol, name in STOCKS.items():
                try:
                    pred_price, act_prices, pred_prices = predict_stock_price(ticker_symbol)[:3]
                    if act_prices is not None and not act_prices.empty and pred_prices is not None and len(pred_prices) > 0:
                        current = float(act_prices.iloc[-1].iloc[0])
                        growth_potential = round((pred_price - current) / current * 100, 2)
                        comparison_data.append((name, growth_potential))
                except Exception as e:
                    print(f"Could not fetch data for {name}: {str(e)}")

            best_stock = max(comparison_data, key=lambda x: x[1], default=None)
            if best_stock and best_stock[0] != stock_name and best_stock[1] > (predicted_price - current_price) / current_price * 100:
                recommendation = f"Consider investing in {best_stock[0]} instead — it shows better growth potential ({best_stock[1]}%)"

            insights = f"{sentiment_message}  Based on your risk tolerance ({risk_tolerance}) and review frequency ({review_frequency}), we suggest: {recommendation}. Estimated return on ₹{round(actual_invested_amount, 2)} investment: ₹{potential_return}."

            # 📉 Price Prediction Graph
            graph = go.Figure()
            graph.add_trace(go.Scatter(x=actual_prices.index, y=actual_prices.iloc[:, 0], mode='lines', name='Actual Price'))
            graph.add_trace(go.Scatter(x=actual_prices.index, y=predicted_prices, mode='lines', name='Predicted Price'))
            graph.update_layout(title=f"{stock_name} Price Prediction", xaxis_title="Date", yaxis_title="Price (INR)")
            graph_json = json.dumps(graph, cls=PlotlyJSONEncoder)

            # 📊 Comparison Bar Chart
            comparison_graph = go.Figure()
            comparison_graph.add_trace(go.Bar(
                x=[item[0] for item in comparison_data],
                y=[item[1] for item in comparison_data],
                text=[f"{item[1]}%" for item in comparison_data],
                textposition='outside',
                marker=dict(color='blue')
            ))
            comparison_graph.update_layout(
                title="📊 Growth Potential Comparison of Selected Stocks",
                xaxis_title="Stock",
                yaxis_title="Growth Potential (%)",
                yaxis=dict(tickformat=".2f"),
                plot_bgcolor='rgba(240,240,240,0.9)'
            )
            comparison_graph_json = json.dumps(comparison_graph, cls=PlotlyJSONEncoder)
            print(table_data)
            return render_template("result.html", stock_name=stock_name, ticker=ticker,
                                   predicted_price=predicted_price,
                                   sentiment_message=sentiment_message,
                                   recommendation=recommendation,
                                   insights=insights,
                                   graph_json=graph_json,
                                   comparison_graph=comparison_graph_json,
                                   table_data=table_data,
                                   is_planning=True)

        except Exception as e:
            return render_template("error.html", message=f"An error occurred: {str(e)}")

    return render_template("planning_to_invest.html", stocks=STOCKS)

import plotly.graph_objects as go
import json

@app.route("/already_invested", methods=["GET", "POST"])
def already_invested():
    if request.method == "POST":
        try:
            ticker = request.form["ticker"]
            stock_name = STOCKS.get(ticker)
            investment_amount = float(request.form["investment_amount"])
            purchase_price = float(request.form["purchase_price"])
            risk_tolerance = request.form["risk_tolerance"]
    
            if not stock_name:
                return render_template("error.html", message="Invalid stock selection.")

            volume = investment_amount / purchase_price
            predicted_price, actual_prices, predicted_prices = predict_stock_price(ticker)[:3]

            if actual_prices is None or actual_prices.empty:
                return render_template("error.html", message=f"No data available for {stock_name}. Please try again later.")

            current_price = float(actual_prices.iloc[-1].iloc[0])

            if predicted_price is None or not isinstance(predicted_price, (int, float)):
                return render_template("error.html", message="Prediction data is not available. Please try again later.")

            sentiment, sentiment_score = get_stock_sentiment(stock_name)

            if sentiment_score > 0.2:
                sentiment_message = f"The market outlook for {stock_name} is optimistic, suggesting potential growth ahead."
            elif sentiment_score < -0.2:
                sentiment_message = f"The market sentiment for {stock_name} is cautious, indicating possible risks."
            else:
                sentiment_message = f"The market sentiment for {stock_name} is stable, with no strong signals in either direction."

            current_value = round(volume * current_price, 2)
            gain_loss = round(current_value - investment_amount, 2)
            predicted_price = round(predicted_price, 2)

            if predicted_price > current_price:
                price_message = f"{stock_name}'s predicted price is ₹{predicted_price}, indicating potential growth."
            elif predicted_price < current_price:
                price_message = f"{stock_name}'s predicted price is ₹{predicted_price}, suggesting a possible decline."
            else:
                price_message = f"{stock_name}'s predicted price is ₹{predicted_price}, showing stability."

            price_change_percentage = (predicted_price - current_price) / current_price * 100

            if abs(price_change_percentage) <= 2:
                recommendation = "Hold"
            elif predicted_price > current_price:
                if risk_tolerance == "High":
                    recommendation = "Strong Buy"
                elif risk_tolerance == "Medium":
                    recommendation = "Consider Buying More"
                else:
                    recommendation = "Hold"
            else:
                if price_change_percentage < -20 and risk_tolerance == "Low":
                    recommendation = "Strongly Consider Reducing Exposure"
                elif price_change_percentage < -15 and risk_tolerance == "Medium":
                    recommendation = "Consider Selling"
                elif price_change_percentage < -10 and risk_tolerance == "High":
                    recommendation = "Reassess and Monitor Closely"
                else:
                    recommendation = "Hold and Review"

            if "Buy" in recommendation:
                insights = f"{sentiment_message} {price_message} Based on your current holding and market trends, adding to your position could be beneficial. Current portfolio value: ₹{current_value} (Gain/Loss: ₹{gain_loss})."
            elif "Sell" in recommendation or "Reducing Exposure" in recommendation:
                insights = f"{sentiment_message} {price_message} Given the projected decline, reconsidering your investment strategy may be wise. Current portfolio value: ₹{current_value} (Gain/Loss: ₹{gain_loss})."
            else:
                insights = f"{sentiment_message} {price_message} A balanced approach is advised. Monitor the market and review your investment periodically. Current portfolio value: ₹{current_value} (Gain/Loss: ₹{gain_loss})."

            graph = go.Figure()
            graph.add_trace(go.Scatter(x=actual_prices.index, y=actual_prices.iloc[:, 0], mode='lines', name='Actual Price'))
            graph.add_trace(go.Scatter(x=actual_prices.index, y=predicted_prices, mode='lines', name='Predicted Price'))
            graph.update_layout(title=f"{stock_name} Price Analysis", xaxis_title="Date", yaxis_title="Price (INR)")  # Fixed here

            graph_json = json.dumps(graph, cls=PlotlyJSONEncoder)
            
            bar_chart_json = None
            if investment_amount and current_value:
                try:
                    bar_chart = go.Figure()
                    bar_chart.add_trace(go.Bar(x=['Investment Amount', 'Current Value'], 
                                               y=[investment_amount, current_value], 
                                               marker_color=['blue', 'green']))
                    bar_chart.update_layout(title='Investment Performance: Gain/Loss', 
                                            yaxis_title='Amount (INR)', 
                                            xaxis_title='Metrics')

                    bar_chart_json = json.dumps(bar_chart, cls=PlotlyJSONEncoder)
                except Exception as e:
                    print(f"Error creating bar chart: {str(e)}")

            return render_template("result.html", stock_name=stock_name, ticker=ticker,
                                   predicted_price=predicted_price,
                                   sentiment_message=sentiment_message,
                                   recommendation=recommendation,
                                   insights=insights,
                                   graph_json=graph_json,
                                   bar_chart_json=bar_chart_json)

        except Exception as e:
            return render_template("error.html", message=f"An error occurred: {str(e)}")

    return render_template("already_invested.html", stocks=STOCKS)
  
# Error page
@app.errorhandler(500)
def internal_error(e):
    return render_template("error.html", message="Something went wrong. Please try again later.")

if __name__ == "__main__":
    app.run(debug=True)
