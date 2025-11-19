# from data.data_fetcher import fetch_technical_indicators
# from data.news_fetcher import fetch_news
# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv

# load_dotenv()
# llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

# class DataCollectionAgent:
#     def collect(self, ticker: str) -> dict:
#         indicators = fetch_technical_indicators(ticker)
#         news = fetch_news(ticker)
#         fundamentals = {"pe_ratio": 30.0}  # Placeholder for NTPC.NS; expand with yf.info
#         return {"indicators": indicators, "news": news, "fundamentals": fundamentals}

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class DataCollectionAgent:
    """Autonomous data collection - NO LLM DEPENDENCY"""
    
    def __init__(self):
        self.data_cache = {}
        self.last_fetch_time = {}
        self.cache_timeout = 300  # 5 minutes in seconds
    
    def collect(self, 
                ticker: str,
                start_date: datetime = None,
                end_date: datetime = None,
                include_news: bool = True,
                include_fundamentals: bool = True) -> Dict:
        """
        Collect all data for analysis
        
        Args:
            ticker: Stock ticker (e.g., 'RELIANCE.NS')
            start_date: Start date for price data
            end_date: End date for price data
            include_news: Whether to fetch news
            include_fundamentals: Whether to fetch fundamentals
        
        Returns:
            Dictionary with technical, news, and fundamental data
        """
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=180)
        
        collection = {
            "ticker": ticker,
            "collected_at": datetime.now().isoformat(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
        try:
            # Collect technical data
            collection["technical_data"] = self._collect_technical_data(
                ticker, 
                start_date, 
                end_date
            )
        except Exception as e:
            collection["technical_data"] = {"error": str(e)}
        
        try:
            # Collect news data
            if include_news:
                collection["news_data"] = self._collect_news_data(ticker)
            else:
                collection["news_data"] = []
        except Exception as e:
            collection["news_data"] = []
        
        try:
            # Collect fundamental data
            if include_fundamentals:
                collection["fundamental_data"] = self._collect_fundamental_data(ticker)
            else:
                collection["fundamental_data"] = {}
        except Exception as e:
            collection["fundamental_data"] = {}
        
        # Validate and structure data
        collection["validation"] = self._validate_collection(collection)
        
        return collection
    
    def _collect_technical_data(self, 
                               ticker: str, 
                               start_date: datetime, 
                               end_date: datetime) -> Dict:
        """Collect technical/price data"""
        
        try:
            import yfinance as yf
        except ImportError:
            return {
                "error": "yfinance not installed",
                "status": "FAILED",
                "data": None
            }
        
        try:
            # Fetch stock data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                return {
                    "status": "EMPTY",
                    "message": f"No data found for {ticker}",
                    "data": None
                }
            
            # Get latest values
            latest = df.iloc[-1]
            
            return {
                "status": "SUCCESS",
                "dataframe": df,
                "record_count": len(df),
                "date_range": {
                    "start": df.index[0].isoformat(),
                    "end": df.index[-1].isoformat()
                },
                "latest_close": float(latest['Close']),
                "latest_open": float(latest['Open']),
                "latest_high": float(latest['High']),
                "latest_low": float(latest['Low']),
                "latest_volume": int(latest['Volume']),
                "price_change_pct": self._calculate_price_change(df),
                "volatility": self._calculate_volatility(df),
                "volume_avg": self._calculate_volume_avg(df)
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "data": None
            }
    
    def _collect_news_data(self, ticker: str) -> List[Dict]:
        """
        Collect news data
        
        This can be extended to use real news APIs like:
        - NewsAPI (newsapi.org)
        - Alpha Vantage News
        - Finnhub
        - RSS feeds
        
        For now, returns empty list to be populated by external service
        """
        
        news_data = []
        
        # Try to fetch from NewsAPI if available
        try:
            import requests
            
            # You can configure this with your own API key
            # For now, return structure for news
            return self._prepare_news_structure(news_data)
        
        except ImportError:
            # NewsAPI not available, return empty
            return []
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def _collect_fundamental_data(self, ticker: str) -> Dict:
        """Collect fundamental data"""
        
        try:
            import yfinance as yf
        except ImportError:
            return {"error": "yfinance not installed"}
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key fundamentals
            fundamentals = {
                "ticker": ticker,
                "company_name": info.get('longName', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                
                # Valuation
                "pe_ratio": info.get('trailingPE', None),
                "forward_pe": info.get('forwardPE', None),
                "price_to_book": info.get('priceToBook', None),
                "price_to_sales": info.get('priceToSalesTrailing12Months', None),
                
                # Growth metrics
                "eps": info.get('trailingEps', None),
                "forward_eps": info.get('forwardEps', None),
                "revenue_growth": info.get('revenueGrowth', None),
                "earnings_growth": info.get('earningsGrowth', None),
                
                # Profitability
                "profit_margin": info.get('profitMargins', None),
                "operating_margin": info.get('operatingMargins', None),
                "roe": info.get('returnOnEquity', None),
                "roa": info.get('returnOnAssets', None),
                
                # Dividends
                "dividend_yield": info.get('dividendYield', None),
                "dividend_rate": info.get('dividendRate', None),
                
                # Debt & Balance Sheet
                "total_debt": info.get('totalDebt', None),
                "total_cash": info.get('totalCash', None),
                "debt_to_equity": info.get('debtToEquity', None),
                "current_ratio": info.get('currentRatio', None),
                
                # Market data
                "market_cap": info.get('marketCap', None),
                "enterprise_value": info.get('enterpriseValue', None),
                "fifty_two_week_high": info.get('fiftyTwoWeekHigh', None),
                "fifty_two_week_low": info.get('fiftyTwoWeekLow', None),
                "fifty_day_avg": info.get('fiftyDayAverage', None),
                "two_hundred_day_avg": info.get('twoHundredDayAverage', None),
                
                # Trading
                "volume_avg": info.get('averageVolume', None),
                "beta": info.get('beta', None)
            }
            
            return {
                "status": "SUCCESS",
                "fundamentals": fundamentals,
                "data_points": sum(1 for v in fundamentals.values() if v is not None)
            }
        
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "fundamentals": {}
            }
    
    def _calculate_price_change(self, df: pd.DataFrame) -> Dict:
        """Calculate price change metrics"""
        
        if len(df) < 2:
            return {"change": 0, "change_pct": 0}
        
        first_close = df['Close'].iloc[0]
        last_close = df['Close'].iloc[-1]
        
        change = last_close - first_close
        change_pct = (change / first_close) * 100
        
        return {
            "change": float(change),
            "change_pct": float(change_pct),
            "from": float(first_close),
            "to": float(last_close)
        }
    
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility metrics"""
        
        returns = df['Close'].pct_change().dropna()
        
        if len(returns) == 0:
            return {"daily": 0, "annual": 0}
        
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        return {
            "daily": float(daily_vol),
            "annual": float(annual_vol),
            "coefficient": float(annual_vol) if returns.mean() != 0 else 0
        }
    
    def _calculate_volume_avg(self, df: pd.DataFrame) -> Dict:
        """Calculate volume metrics"""
        
        if 'Volume' not in df.columns or len(df) == 0:
            return {"avg": 0, "latest": 0}
        
        avg_vol = df['Volume'].mean()
        latest_vol = df['Volume'].iloc[-1]
        
        return {
            "avg": float(avg_vol),
            "latest": float(latest_vol),
            "ratio": float(latest_vol / avg_vol) if avg_vol > 0 else 0
        }
    
    def _prepare_news_structure(self, news_list: List[Dict]) -> List[Dict]:
        """Prepare news data in standard format"""
        
        prepared = []
        for news in news_list:
            prepared.append({
                "title": news.get('title', 'N/A'),
                "content": news.get('description', ''),
                "source": news.get('source', 'Unknown'),
                "published_at": news.get('publishedAt', datetime.now().isoformat()),
                "url": news.get('url', '')
            })
        
        return prepared
    
    def _validate_collection(self, collection: Dict) -> Dict:
        """Validate collected data"""
        
        validation = {
            "status": "VALID",
            "warnings": [],
            "errors": []
        }
        
        # Check technical data
        if "error" in collection.get("technical_data", {}):
            validation["errors"].append(f"Technical data error: {collection['technical_data']['error']}")
            validation["status"] = "INCOMPLETE"
        
        # Check news data
        if not collection.get("news_data"):
            validation["warnings"].append("No news data available")
        
        # Check fundamentals
        if "error" in collection.get("fundamental_data", {}):
            validation["warnings"].append(f"Fundamental data error: {collection['fundamental_data']['error']}")
        
        return validation
    
    def format_for_analysis(self, collection: Dict) -> Dict:
        """Format collected data for agent analysis"""
        
        tech_data = collection.get("technical_data", {})
        news_data = collection.get("news_data", [])
        fund_data = collection.get("fundamental_data", {})
        
        return {
            "ticker": collection["ticker"],
            "technical": {
                "dataframe": tech_data.get("dataframe"),
                "latest_close": tech_data.get("latest_close"),
                "price_change": tech_data.get("price_change_pct"),
                "volatility": tech_data.get("volatility"),
                "volume": tech_data.get("latest_volume"),
                "records": tech_data.get("record_count")
            },
            "news": news_data,
            "fundamentals": fund_data.get("fundamentals", {}),
            "validation": collection.get("validation")
        }
    
    def get_data_summary(self, collection: Dict) -> str:
        """Generate summary of collected data"""
        
        tech = collection.get("technical_data", {})
        news = collection.get("news_data", [])
        fund = collection.get("fundamental_data", {})
        
        summary = f"""
╔════════════════════════════════════════╗
║       DATA COLLECTION SUMMARY          ║
╚════════════════════════════════════════╝

Ticker: {collection['ticker']}
Collected: {collection['collected_at']}

TECHNICAL DATA:
  Status: {tech.get('status', 'N/A')}
  Records: {tech.get('record_count', 0)}
  Latest Close: ₹{tech.get('latest_close', 0):,.2f}
  Price Change: {tech.get('price_change_pct', {}).get('change_pct', 0):.2f}%
  Volatility (Annual): {tech.get('volatility', {}).get('annual', 0):.2%}
  Avg Volume: {tech.get('volume_avg', {}).get('avg', 0):,.0f}

NEWS DATA:
  Articles: {len(news)}
  Sources: {', '.join(set(n.get('source', 'Unknown') for n in news)[:5])}

FUNDAMENTAL DATA:
  Company: {fund.get('fundamentals', {}).get('company_name', 'N/A')}
  PE Ratio: {fund.get('fundamentals', {}).get('pe_ratio', 'N/A')}
  Market Cap: ₹{fund.get('fundamentals', {}).get('market_cap', 0):,.0f}
  
VALIDATION:
  Status: {collection.get('validation', {}).get('status', 'UNKNOWN')}
  Warnings: {len(collection.get('validation', {}).get('warnings', []))}
  Errors: {len(collection.get('validation', {}).get('errors', []))}
"""
        
        return summary