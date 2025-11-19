from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

# Custom RSI calculation compatible with NumPy arrays
def calculate_rsi(prices, period=14):
    prices = np.array(prices)  # Ensure input is a NumPy array
    delta = np.diff(prices, prepend=prices[0])  # Compute differences, prepend first value to match length
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    # Pad with NaN to match original length, then fill initial values
    rsi_full = np.full(len(prices), np.nan)
    rsi_full[period-1:] = rsi
    return pd.Series(rsi_full, index=range(len(prices)))  # Return as pandas Series for backtesting

class RSIStrategy(Strategy):
    n1 = 14  # RSI period

    def init(self):
        # Calculate RSI using the custom function
        self.rsi = calculate_rsi(self.data.Close, self.n1)

    def next(self):
        if crossover(self.rsi, 30) and not self.position:  # Buy when RSI crosses above 30
            self.buy()
        elif crossover(70, self.rsi) and self.position:  # Sell when RSI crosses below 70
            self.position.close()

class BacktestingAgent:
    def backtest(self, ticker: str, period: str = "1y") -> dict:
        stock = yf.Ticker(ticker)  # e.g., NTPC.NS
        df = stock.history(period=period)
        if df.empty:
            return {"error": f"No data available for {ticker}"}

        # Ensure required columns are present with correct names
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0  # Fallback if column is missing (e.g., Volume)
        df = df[required_cols]  # Select only required columns

        bt = Backtest(df, RSIStrategy, cash=1000000, commission=.002)  # ₹10,00,000
        stats = bt.run()

        # Debug: Print available keys to verify
        print("Available stats keys:", stats.index.tolist())

        prompt = f"Analyze backtest results for {ticker} (e.g., NTPC.NS): {stats}. Provide a brief summary and recommendation."
        analysis = llm.invoke(prompt).content.strip()

        return {
            "ticker": ticker,
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "initial_capital": stats.get("Starting Cash", 1000000),  # Updated key
            "ending_capital": stats.get("Equity Final [$]", 0),      # Updated key
            "total_return": stats.get("Return [%]", 0) * 100,        # Updated key
            "sharpe_ratio": stats.get("Sharpe Ratio", 0),            # Updated key
            "max_drawdown": stats.get("Max. Drawdown [%]", 0) * 100, # Updated key
            "analysis": analysis
        }