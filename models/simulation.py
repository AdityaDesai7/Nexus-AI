from pydantic import BaseModel
from typing import Dict
import yfinance as yf

class Trade(BaseModel):
    ticker: str
    action: str
    quantity: int
    price: float

class PortfolioState:
    def __init__(self):
        self.balance = 1000000.0  # Initial balance in INR (from .env)
        self.holdings: Dict[str, int] = {}  # Ticker: quantity
        self.trades = []
        self.total_value = self.balance

    def update_value(self, usd_to_inr: float = 83.0):
        holdings_value = sum(holdings * yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1] * usd_to_inr
                            for ticker, holdings in self.holdings.items())
        self.total_value = self.balance + holdings_value