from models.simulation import Trade, PortfolioState
import yfinance as yf
from dotenv import load_dotenv
import os

load_dotenv()
USD_INR = float(os.getenv("USD_TO_INR", 83))

class PortfolioManager:
    def __init__(self):
        self.state = PortfolioState()

    def execute_trade(self, trade):
        # Handle both Trade object and dict input
        if isinstance(trade, dict):
            ticker = trade.get("ticker")
            action = trade.get("action")
            quantity = trade.get("quantity")
            price = trade.get("price")
        else:  # Assume Trade object
            ticker = trade.ticker
            action = trade.action
            quantity = trade.quantity
            price = trade.price

        if not ticker or action not in ["Buy", "Sell"] or quantity is None or price is None:
            return self.state

        price_usd = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        price_inr = price_usd * USD_INR
        cost = quantity * price_inr

        if action == "Buy" and self.state.balance >= cost:
            self.state.balance -= cost
            self.state.holdings[ticker] = self.state.holdings.get(ticker, 0) + quantity
            self.state.trades.append({"ticker": ticker, "action": action, "quantity": quantity, "price": price_inr})
        elif action == "Sell" and ticker in self.state.holdings and self.state.holdings[ticker] >= quantity:
            self.state.balance += cost
            self.state.holdings[ticker] -= quantity
            if self.state.holdings[ticker] == 0:
                del self.state.holdings[ticker]
            self.state.trades.append({"ticker": ticker, "action": action, "quantity": quantity, "price": price_inr})

        self.state.update_value(USD_INR)
        return self.state