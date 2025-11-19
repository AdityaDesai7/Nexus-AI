# # data/data_fetcher.py
# import yfinance as yf
# import pandas as pd
# from datetime import datetime


# def fetch_data(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
#     df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    
#     if df.empty:
#         raise ValueError(f"No data for '{ticker}'")

#     if isinstance(df.columns, pd.MultiIndex):
#         if 'Ticker' in df.columns.names:
#             df = df.droplevel('Ticker', axis=1)
#         if isinstance(df.columns, pd.MultiIndex) and 'Price' in df.columns.names:
#             df = df.droplevel('Price', axis=1)

#     df = df.reset_index()
#     return df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]


# data/data_fetcher.py
"""
Data Fetcher - Only responsible for fetching raw data
No calculations, no indicators, just fetch!
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_data(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch raw OHLCV data from Yahoo Finance
    
    Args:
        ticker: Stock ticker (e.g., 'RELIANCE.NS')
        start: Start date
        end: End date
    
    Returns:
        pd.DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
    """
    logger.info(f"📊 Fetching data for {ticker} from {start.date()} to {end.date()}")
    
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        
        if df.empty:
            raise ValueError(f"No data for '{ticker}'")
        
        # Handle MultiIndex columns (when yfinance returns multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            if 'Ticker' in df.columns.names:
                df = df.droplevel('Ticker', axis=1)
            if isinstance(df.columns, pd.MultiIndex) and 'Price' in df.columns.names:
                df = df.droplevel('Price', axis=1)
        
        # Reset index and select only needed columns
        df = df.reset_index()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
        logger.info(f"✅ Fetched {len(df)} rows for {ticker}")
        return df
    
    except Exception as e:
        logger.error(f"❌ Error fetching data: {e}")
        raise


def fetch_multiple_stocks(tickers: list, start: datetime, end: datetime) -> dict:
    """
    Fetch data for multiple stocks
    
    Args:
        tickers: List of tickers
        start: Start date
        end: End date
    
    Returns:
        Dictionary with {ticker: DataFrame}
    """
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = fetch_data(ticker, start, end)
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch {ticker}: {e}")
    
    return data


def fetch_fii_data() -> dict:
    """
    Placeholder for FII data fetching
    Would connect to real API (NSE, Moneycontrol, etc.)
    """
    # TODO: Implement real FII API integration
    return {
        'today': 0,
        '30_day_avg': 0,
        'trend': []
    }


def fetch_block_deals() -> list:
    """
    Placeholder for block deals fetching
    Would connect to NSE API
    """
    # TODO: Implement real block deals API
    return []


def fetch_holdings_data(ticker: str) -> dict:
    """
    Placeholder for MF holdings data
    """
    # TODO: Implement real holdings API
    return {
        'previous': 0,
        'current': 0
    }

