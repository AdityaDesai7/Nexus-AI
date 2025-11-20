# trading_bot/tools/toolbox.py
import logging
from datetime import datetime
from data.data_fetcher import fetch_data
from data.new_news_fetcher import NewNewsFetcher

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_news = NewNewsFetcher()

def tool_fetch_price(ticker: str, start: datetime, end: datetime):
    """
    Return a raw pandas.DataFrame containing OHLCV with a guaranteed 'Close' column.
    This function raises on errors so callers (wrappers) can handle exceptions.
    """
    try:
        df = fetch_data(ticker, start, end)
        if df is None or len(df) == 0:
            raise RuntimeError("No price data returned")
        if "Close" not in df.columns:
            raise RuntimeError("Price df missing Close column")
        return df
    except Exception as e:
        logger.exception("tool_fetch_price ERROR")
        raise

def tool_fetch_news(ticker: str, limit: int = 15):
    """
    Return list of article dicts. If error, return empty list.
    Each article dict should contain title, url, source, summary (best-effort).
    """
    try:
        articles = _news.get_news(company_name=ticker, ticker=ticker, max_results=limit, days=5, dedupe=True)
        return articles or []
    except Exception:
        logger.exception("tool_fetch_news ERROR")
        return []

def tool_fetch_fundamentals(ticker: str):
    # Placeholder
    return {"pe_ratio": None, "eps": None, "market_cap": None, "debt_to_equity": None}

TOOLS = {
    "fetch_price": tool_fetch_price,
    "fetch_news": tool_fetch_news,
    "fetch_fundamentals": tool_fetch_fundamentals
}
