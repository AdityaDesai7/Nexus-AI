# # ui/stock_analysis.py
# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from main import fetch_and_enhance
# from data.news_fetcher import NewsFetcher
# from datetime import datetime, timedelta

# def show_stock_analysis():
#     """Display comprehensive stock analysis page"""
    
#     ticker = st.session_state.get('ticker', 'RELIANCE.NS')
    
#     # Header with back button
#     col1, col2, col3 = st.columns([1, 6, 1])
#     with col1:
#         if st.button("← Back", use_container_width=True):
#             st.session_state.page = 'home'
#             st.rerun()
#     with col2:
#         st.markdown(f"<h1 style='text-align:center; color:#00b894;'>📊 {ticker} Analysis</h1>", unsafe_allow_html=True)
#     with col3:
#         if st.button("🤖 AI Agents", type="primary", use_container_width=True):
#             st.session_state.page = 'ai_agents'
#             st.rerun()
    
#     # Fetch basic info
#     with st.spinner("Loading stock data..."):
#         try:
#             stock = yf.Ticker(ticker)
#             info = stock.info
#             company_name = info.get('longName', ticker)
            
#             st.markdown(f"<h3 style='text-align:center; color:#888;'>{company_name}</h3>", unsafe_allow_html=True)
            
#         except Exception as e:
#             st.error(f"Error loading stock data: {str(e)}")
#             return
    
#     # Create tabs
#     tab1, tab2, tab3, tab4 = st.tabs(["📈 Technical Analysis", "💼 Fundamental Analysis", "📰 Market News", "📊 Charts & Data"])
    
#     # ========================================
#     # TAB 1: TECHNICAL ANALYSIS
#     # ========================================
#     with tab1:
#         st.markdown("## Technical Indicators & Metrics")
        
#         # Fetch technical data
#         start_date = datetime.now() - timedelta(days=365)
#         end_date = datetime.now()
        
#         with st.spinner("Calculating technical indicators..."):
#             df = fetch_and_enhance(ticker, start_date, end_date)
#             latest = df.iloc[-1]
        
#         st.success(f"✅ Loaded {len(df):,} trading days")
        
#         # Key Technical Metrics
#         cols = st.columns(6)
#         with cols[0]:
#             st.metric("Close Price", f"₹{latest['Close']:.2f}")
#         with cols[1]:
#             rsi_color = "🟢" if latest['RSI'] < 30 else "🔴" if latest['RSI'] > 70 else "🟡"
#             st.metric("RSI", f"{latest['RSI']:.1f} {rsi_color}")
#         with cols[2]:
#             macd_signal = "🟢 Bullish" if latest['MACD'] > latest['MACD_Signal'] else "🔴 Bearish"
#             st.metric("MACD", f"{latest['MACD']:.2f}", macd_signal)
#         with cols[3]:
#             st.metric("VWAP", f"₹{latest['VWAP']:.2f}")
#         with cols[4]:
#             st.metric("Volume", f"{latest['Volume']:,.0f}")
#         with cols[5]:
#             change_pct = ((latest['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100
#             st.metric("YoY Change", f"{change_pct:+.2f}%")
        
#         st.markdown("---")
        
#         # Technical Signals
#         st.markdown("### 📊 Technical Signals")
        
#         signals = []
        
#         # RSI Signals
#         if latest['RSI'] < 30:
#             signals.append(("🟢 BUY Signal", "RSI Oversold", f"RSI at {latest['RSI']:.1f} indicates oversold conditions"))
#         elif latest['RSI'] > 70:
#             signals.append(("🔴 SELL Signal", "RSI Overbought", f"RSI at {latest['RSI']:.1f} indicates overbought conditions"))
        
#         # MACD Signals
#         if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > 0:
#             signals.append(("🟢 BUY Signal", "MACD Bullish Crossover", "MACD crossed above signal line"))
#         elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < 0:
#             signals.append(("🔴 SELL Signal", "MACD Bearish Crossover", "MACD crossed below signal line"))
        
#         # Bollinger Bands
#         if latest['Close'] < latest['BB_Lower']:
#             signals.append(("🟢 BUY Signal", "Price Below Lower BB", "Potential bounce from support"))
#         elif latest['Close'] > latest['BB_Upper']:
#             signals.append(("🔴 SELL Signal", "Price Above Upper BB", "Potential reversal from resistance"))
        
#         # EMA Trend
#         if latest['EMA12'] > latest['EMA26']:
#             signals.append(("🟢 Bullish Trend", "EMA Crossover", "Short-term EMA above long-term EMA"))
#         else:
#             signals.append(("🔴 Bearish Trend", "EMA Crossover", "Short-term EMA below long-term EMA"))
        
#         for signal_type, signal_name, description in signals:
#             cols = st.columns([1, 2, 4])
#             with cols[0]:
#                 st.markdown(f"**{signal_type}**")
#             with cols[1]:
#                 st.markdown(f"`{signal_name}`")
#             with cols[2]:
#                 st.markdown(description)
        
#         st.markdown("---")
        
#         # Candlestick Chart
#         st.markdown("### 📈 Price Action Chart")
        
#         fig = make_subplots(
#             rows=3, cols=1,
#             shared_xaxes=True,
#             row_heights=[0.6, 0.2, 0.2],
#             subplot_titles=("Price & Indicators", "RSI", "Volume"),
#             vertical_spacing=0.05
#         )
        
#         # Candlestick
#         fig.add_trace(go.Candlestick(
#             x=df["Date"],
#             open=df["Open"],
#             high=df["High"],
#             low=df["Low"],
#             close=df["Close"],
#             name="Price"
#         ), row=1, col=1)
        
#         # Bollinger Bands
#         fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], line=dict(color="rgba(255,255,255,0.2)", dash='dash'), name="BB Upper", showlegend=False), row=1, col=1)
#         fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], line=dict(color="rgba(255,255,255,0.2)", dash='dash'), name="BB Lower", fill='tonexty', fillcolor='rgba(255,255,255,0.05)', showlegend=False), row=1, col=1)
        
#         # EMAs
#         fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA12"], line=dict(color="#00b894", width=2), name="EMA 12"), row=1, col=1)
#         fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA26"], line=dict(color="#ff7675", width=2), name="EMA 26"), row=1, col=1)
        
#         # RSI
#         fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], line=dict(color="#6c5ce7", width=2), name="RSI"), row=2, col=1)
#         fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,118,117,0.5)", row=2, col=1)
#         fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,184,148,0.5)", row=2, col=1)
        
#         # Volume
#         vol_colors = ['#00b894' if df["Close"].iloc[i] >= df["Open"].iloc[i] else '#ff7675' for i in range(len(df))]
#         fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], marker_color=vol_colors, name="Volume"), row=3, col=1)
        
#         fig.update_layout(height=800, template="plotly_dark", showlegend=True, hovermode='x unified')
#         fig.update_xaxes(rangeslider_visible=False)
        
#         st.plotly_chart(fig, use_container_width=True)
    
#     # ========================================
#     # TAB 2: FUNDAMENTAL ANALYSIS
#     # ========================================
#     with tab2:
#         st.markdown("## Fundamental Analysis")
        
#         with st.spinner("Loading fundamental data..."):
#             try:
#                 income_stmt = stock.income_stmt
#                 balance_sheet = stock.balance_sheet
#                 cashflow = stock.cashflow
                
#                 # Filter for recent years
#                 target_years = [str(year) for year in range(2022, 2026)]
#                 income_stmt = income_stmt[[col for col in income_stmt.columns if str(col.year) in target_years]]
#                 balance_sheet = balance_sheet[[col for col in balance_sheet.columns if str(col.year) in target_years]]
#                 cashflow = cashflow[[col for col in cashflow.columns if str(col.year) in target_years]]
                
#                 # Financial Ratios
#                 ratios = {
#                     "P/E Ratio": info.get("trailingPE"),
#                     "Forward P/E": info.get("forwardPE"),
#                     "Price/Book": info.get("priceToBook"),
#                     "Price/Sales": info.get("priceToSalesTrailing12Months"),
#                     "ROE": info.get("returnOnEquity"),
#                     "ROA": info.get("returnOnAssets"),
#                     "Debt/Equity": info.get("debtToEquity") / 100 if info.get("debtToEquity") else None,
#                     "Current Ratio": info.get("currentRatio"),
#                     "Dividend Yield": info.get("dividendYield"),
#                 }
                
#                 st.markdown("### 💰 Key Financial Ratios")
                
#                 cols = st.columns(4)
#                 for idx, (key, value) in enumerate(ratios.items()):
#                     if value is not None:
#                         with cols[idx % 4]:
#                             if key in ["ROE", "ROA", "Dividend Yield"]:
#                                 formatted_value = f"{value * 100:.2f}%"
#                             else:
#                                 formatted_value = f"{value:.2f}"
#                             st.metric(key, formatted_value)
                
#                 st.markdown("---")
                
#                 # Income Statement
#                 if not income_stmt.empty:
#                     st.markdown("### 📊 Income Statement Highlights")
                    
#                     latest_col = income_stmt.columns[0]
                    
#                     cols = st.columns(3)
#                     if "Total Revenue" in income_stmt.index:
#                         with cols[0]:
#                             st.metric("Total Revenue", f"₹{income_stmt.loc['Total Revenue', latest_col]:,.0f}")
#                     if "Net Income" in income_stmt.index:
#                         with cols[1]:
#                             st.metric("Net Income", f"₹{income_stmt.loc['Net Income', latest_col]:,.0f}")
#                     if "Diluted EPS" in income_stmt.index:
#                         with cols[2]:
#                             st.metric("EPS", f"₹{income_stmt.loc['Diluted EPS', latest_col]:.2f}")
                    
#                     with st.expander("📄 View Full Income Statement"):
#                         st.dataframe(income_stmt, use_container_width=True)
                
#                 # Balance Sheet
#                 if not balance_sheet.empty:
#                     st.markdown("### 🏦 Balance Sheet Highlights")
                    
#                     latest_bs = balance_sheet.iloc[:, 0]
                    
#                     cols = st.columns(3)
#                     if "Stockholders Equity" in balance_sheet.index:
#                         with cols[0]:
#                             st.metric("Total Equity", f"₹{latest_bs.get('Stockholders Equity'):,.0f}")
#                     if "Total Debt" in balance_sheet.index:
#                         with cols[1]:
#                             st.metric("Total Debt", f"₹{latest_bs.get('Total Debt'):,.0f}")
#                     if "Cash And Cash Equivalents" in balance_sheet.index:
#                         with cols[2]:
#                             st.metric("Cash", f"₹{latest_bs.get('Cash And Cash Equivalents'):,.0f}")
                    
#                     with st.expander("📄 View Full Balance Sheet"):
#                         st.dataframe(balance_sheet, use_container_width=True)
                
#                 # Cash Flow
#                 if not cashflow.empty:
#                     st.markdown("### 💵 Cash Flow Highlights")
                    
#                     latest_cf = cashflow.columns[0]
                    
#                     cols = st.columns(3)
#                     if "Operating Cash Flow" in cashflow.index:
#                         with cols[0]:
#                             st.metric("Operating Cash Flow", f"₹{cashflow.loc['Operating Cash Flow', latest_cf]:,.0f}")
#                     if "Capital Expenditure" in cashflow.index:
#                         with cols[1]:
#                             st.metric("CapEx", f"₹{cashflow.loc['Capital Expenditure', latest_cf]:,.0f}")
#                     if "Free Cash Flow" in cashflow.index:
#                         with cols[2]:
#                             st.metric("Free Cash Flow", f"₹{cashflow.loc['Free Cash Flow', latest_cf]:,.0f}")
                    
#                     with st.expander("📄 View Full Cash Flow Statement"):
#                         st.dataframe(cashflow, use_container_width=True)
                
#             except Exception as e:
#                 st.error(f"Error loading fundamental data: {str(e)}")
    
#     # ========================================
#     # TAB 3: MARKET NEWS
#     # ========================================
#     with tab3:
#         st.markdown("## 📰 Latest Market News & Sentiment")
    
#     with st.spinner("Fetching latest news..."):
#         try:
#             news_fetcher = NewsFetcher()
            
#             # ✅ FIXED: Get sentiment summary
#             sentiment_data = news_fetcher.get_sentiment_summary(ticker, company_name)
            
#             # ✅ FIXED: Check all required keys exist
#             sentiment = sentiment_data.get('sentiment', 'neutral')
#             confidence = sentiment_data.get('confidence', 0)
#             article_count = sentiment_data.get('article_count', 0)
#             summary = sentiment_data.get('summary', 'No sentiment data')
            
#             # Display sentiment
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 sentiment_emoji = "🟢" if sentiment == 'positive' else "🔴" if sentiment == 'negative' else "🟡"
#                 st.metric("Overall Sentiment", f"{sentiment_emoji} {sentiment.upper()}")
            
#             with col2:
#                 st.metric("Confidence", f"{confidence:.1f}%")
            
#             with col3:
#                 st.metric("Articles Analyzed", article_count)
            
#             st.info(summary)
            
#             st.markdown("---")
            
#             # Get news articles
#             news_articles = news_fetcher.get_stock_news(ticker, company_name, max_results=10)
            
#             if news_articles:
#                 st.markdown(f"### 📑 Latest News Articles ({len(news_articles)} found)")
                
#                 for idx, article in enumerate(news_articles, 1):
#                     with st.expander(f"**{idx}. {article.get('title', 'No title')}**"):
#                         st.markdown(f"**Published:** {article.get('published_date', 'Unknown')}")
#                         st.markdown(f"**Source:** {article.get('source', 'Unknown')}")
#                         st.markdown(f"**Relevance Score:** {article.get('score', 0):.2f}")
#                         st.markdown(f"**Content:** {article.get('content', 'No content')}")
#                         st.markdown(f"[Read Full Article]({article.get('url', '#')})")
#             else:
#                 st.warning("No news articles found for this stock.")
                
#         except Exception as e:
#             st.error(f"Error fetching news: {str(e)}")
#             logger.error(f"News fetching error: {e}")
    
#     # ========================================
#     # TAB 4: ADDITIONAL CHARTS & DATA
#     # ========================================
#     with tab4:
#         st.markdown("## 📊 Additional Charts & Data")
        
#         # Add more technical indicators
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Stochastic Oscillator
#             fig_stoch = go.Figure()
#             fig_stoch.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_K"], name="Stochastic %K", line=dict(color="#00b894")))
#             fig_stoch.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_D"], name="Stochastic %D", line=dict(color="#ff7675")))
#             fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
#             fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
#             fig_stoch.update_layout(title="Stochastic Oscillator", template="plotly_dark", height=300)
#             st.plotly_chart(fig_stoch, use_container_width=True)
        
#         with col2:
#             # OBV
#             fig_obv = go.Figure()
#             fig_obv.add_trace(go.Scatter(x=df["Date"], y=df["OBV"], name="On-Balance Volume", line=dict(color="#6c5ce7")))
#             fig_obv.update_layout(title="On-Balance Volume (OBV)", template="plotly_dark", height=300)
#             st.plotly_chart(fig_obv, use_container_width=True)
        
#         # Data table
#         st.markdown("### 📋 Recent Data (Last 20 Days)")
#         display_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']
#         st.dataframe(df[display_cols].tail(20), use_container_width=True)

# ui/stock_analysis.py - FIXED VERSION WITH FinBERT
# ui/stock_analysis.py - ENHANCED VERSION WITH PROFESSIONAL TECHNICAL & FUNDAMENTAL ANALYSIS

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from main import fetch_and_enhance
from datetime import datetime, timedelta
import logging
import requests
from bs4 import BeautifulSoup
import re
from collections import Counter
import time
from urllib.parse import quote_plus, quote, urljoin
import json
import feedparser
import warnings
warnings.filterwarnings('ignore')

# ==================== TECHNICAL ANALYSIS ENHANCEMENTS ====================

class AdvancedTechnicalAnalyzer:
    """Enhanced technical analysis with professional indicators"""
    
    def __init__(self, df):
        self.df = df
        self.calculate_advanced_indicators()
    
    def calculate_advanced_indicators(self):
        """Calculate professional trading indicators"""
        df = self.df
        
        # Price Channels
        df['Price_Channel_High'] = df['High'].rolling(20).max()
        df['Price_Channel_Low'] = df['Low'].rolling(20).min()
        
        # Donchian Channels
        df['Donchian_High'] = df['High'].rolling(20).max()
        df['Donchian_Low'] = df['Low'].rolling(20).min()
        df['Donchian_Middle'] = (df['Donchian_High'] + df['Donchian_Low']) / 2
        
        # Keltner Channels
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        atr = df['High'].rolling(10).std()  # Simplified ATR
        df['Keltner_Middle'] = typical_price.rolling(20).mean()
        df['Keltner_Upper'] = df['Keltner_Middle'] + 2 * atr
        df['Keltner_Lower'] = df['Keltner_Middle'] - 2 * atr
        
        # Volume-based indicators
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        df['Price_Acceleration'] = df['Momentum'] - df['Momentum'].shift(5)
        
        # Support and Resistance
        self.calculate_support_resistance()
        
        # Trend Strength
        df['Trend_Strength'] = self.calculate_trend_strength()
        
        self.df = df
    
    def calculate_support_resistance(self):
        """Calculate dynamic support and resistance levels"""
        df = self.df
        
        # Pivot Points (simplified)
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        
        # Recent swing highs and lows
        df['Swing_High'] = df['High'].rolling(5, center=True).max()
        df['Swing_Low'] = df['Low'].rolling(5, center=True).min()
    
    def calculate_trend_strength(self):
        """Calculate trend strength indicator"""
        price_change = self.df['Close'].pct_change(10)
        volume_trend = self.df['Volume'].rolling(10).mean() / self.df['Volume'].rolling(30).mean()
        return price_change * volume_trend
    
    def get_enhanced_signals(self):
        """Generate comprehensive trading signals"""
        latest = self.df.iloc[-1]
        signals = []
        
        # Multi-timeframe analysis
        price_1d = latest['Close']
        price_5d = self.df['Close'].iloc[-5]
        price_20d = self.df['Close'].iloc[-20]
        
        # Trend Analysis
        short_trend = "🟢 Bullish" if price_1d > self.df['EMA12'].iloc[-1] else "🔴 Bearish"
        medium_trend = "🟢 Bullish" if price_1d > self.df['EMA26'].iloc[-1] else "🔴 Bearish"
        long_trend = "🟢 Bullish" if price_1d > self.df['Close'].iloc[-50] else "🔴 Bearish"
        
        signals.append((f"📈 Trend: {short_trend} (S)", "Trend", f"Short-term: {short_trend}"))
        signals.append((f"📊 Trend: {medium_trend} (M)", "Trend", f"Medium-term: {medium_trend}"))
        signals.append((f"📋 Trend: {long_trend} (L)", "Trend", f"Long-term: {long_trend}"))
        
        # RSI with momentum
        if latest['RSI'] < 30 and latest['Momentum'] > 0:
            signals.append(("🟢 STRONG BUY", "Momentum", "RSI oversold with positive momentum"))
        elif latest['RSI'] > 70 and latest['Momentum'] < 0:
            signals.append(("🔴 STRONG SELL", "Momentum", "RSI overbought with negative momentum"))
        
        # Volume confirmation
        if latest['Volume_Ratio'] > 1.5 and latest['Close'] > latest['Open']:
            signals.append(("🟢 VOLUME CONFIRMATION", "Volume", "High volume on up move"))
        elif latest['Volume_Ratio'] > 1.5 and latest['Close'] < latest['Open']:
            signals.append(("🔴 VOLUME DISTRIBUTION", "Volume", "High volume on down move"))
        
        # Bollinger Band squeeze
        bb_width = (latest['BB_Upper'] - latest['BB_Lower']) / latest['Close']
        avg_bb_width = (self.df['BB_Upper'] - self.df['BB_Lower']).tail(20).mean() / self.df['Close'].tail(20).mean()
        if bb_width < avg_bb_width * 0.7:
            signals.append(("⚡ VOLATILITY SQUEEZE", "Volatility", "Bollinger Bands tightening - big move coming"))
        
        # Price channel breakout
        if latest['Close'] > latest['Price_Channel_High']:
            signals.append(("🟢 CHANNEL BREAKOUT", "Breakout", "Price broke above 20-day high"))
        elif latest['Close'] < latest['Price_Channel_Low']:
            signals.append(("🔴 CHANNEL BREAKDOWN", "Breakdown", "Price broke below 20-day low"))
        
        return signals

# ==================== FUNDAMENTAL ANALYSIS ENHANCEMENTS ====================

class AdvancedFundamentalAnalyzer:
    """Enhanced fundamental analysis with comprehensive metrics"""
    
    def __init__(self, stock):
        self.stock = stock
        self.info = stock.info
        self.enhance_fundamental_data()
    
    def enhance_fundamental_data(self):
        """Calculate advanced fundamental metrics"""
        
        # Valuation Metrics
        self.valuation_metrics = {
            "Market Cap": self.format_currency(self.info.get('marketCap')),
            "Enterprise Value": self.format_currency(self.info.get('enterpriseValue')),
            "Trailing P/E": self.info.get('trailingPE'),
            "Forward P/E": self.info.get('forwardPE'),
            "PEG Ratio": self.info.get('pegRatio'),
            "Price/Sales": self.info.get('priceToSalesTrailing12Months'),
            "Price/Book": self.info.get('priceToBook'),
            "Enterprise Value/Revenue": self.info.get('enterpriseToRevenue'),
            "Enterprise Value/EBITDA": self.info.get('enterpriseToEbitda'),
        }
        
        # Profitability Metrics
        self.profitability_metrics = {
            "ROE %": self.info.get('returnOnEquity', 0) * 100 if self.info.get('returnOnEquity') else None,
            "ROA %": self.info.get('returnOnAssets', 0) * 100 if self.info.get('returnOnAssets') else None,
            "Operating Margin %": self.info.get('operatingMargins', 0) * 100 if self.info.get('operatingMargins') else None,
            "Profit Margin %": self.info.get('profitMargins', 0) * 100 if self.info.get('profitMargins') else None,
            "Gross Margin %": self.info.get('grossMargins', 0) * 100 if self.info.get('grossMargins') else None,
        }
        
        # Growth Metrics
        self.growth_metrics = {
            "Revenue Growth (Q) %": self.info.get('revenueGrowth', 0) * 100 if self.info.get('revenueGrowth') else None,
            "Earnings Growth (Q) %": self.info.get('earningsGrowth', 0) * 100 if self.info.get('earningsGrowth') else None,
            "EPS Growth (Q) %": self.info.get('earningsQuarterlyGrowth', 0) * 100 if self.info.get('earningsQuarterlyGrowth') else None,
        }
        
        # Financial Health
        self.health_metrics = {
            "Current Ratio": self.info.get('currentRatio'),
            "Quick Ratio": self.info.get('quickRatio'),
            "Debt/Equity": self.info.get('debtToEquity'),
            "Interest Coverage": self.info.get('interestCoverage'),
            "Free Cash Flow": self.format_currency(self.info.get('freeCashflow')),
        }
        
        # Dividend Information
        self.dividend_metrics = {
            "Dividend Yield %": self.info.get('dividendYield', 0) * 100 if self.info.get('dividendYield') else None,
            "Dividend Rate": self.format_currency(self.info.get('dividendRate')),
            "Payout Ratio": self.info.get('payoutRatio'),
            "5Y Dividend Growth %": self.info.get('dividendGrowth', 0) * 100 if self.info.get('dividendGrowth') else None,
        }
    
    def format_currency(self, value):
        """Format large currency values"""
        if value is None:
            return "N/A"
        if value >= 1e12:
            return f"₹{value/1e12:.2f}T"
        elif value >= 1e9:
            return f"₹{value/1e9:.2f}B"
        elif value >= 1e6:
            return f"₹{value/1e6:.2f}M"
        else:
            return f"₹{value:,.0f}"
    
    def get_valuation_grade(self):
        """Grade the stock's valuation"""
        grades = []
        
        pe = self.info.get('trailingPE')
        if pe:
            if pe < 15:
                grades.append(("🟢 P/E", "Undervalued"))
            elif pe > 25:
                grades.append(("🔴 P/E", "Overvalued"))
            else:
                grades.append(("🟡 P/E", "Fair Value"))
        
        pb = self.info.get('priceToBook')
        if pb:
            if pb < 1.5:
                grades.append(("🟢 P/B", "Undervalued"))
            elif pb > 3:
                grades.append(("🔴 P/B", "Overvalued"))
            else:
                grades.append(("🟡 P/B", "Fair Value"))
        
        return grades
    
    def get_financial_health_grade(self):
        """Grade the company's financial health"""
        grades = []
        
        current_ratio = self.info.get('currentRatio')
        if current_ratio:
            if current_ratio > 2:
                grades.append(("🟢 Liquidity", "Strong"))
            elif current_ratio < 1:
                grades.append(("🔴 Liquidity", "Weak"))
            else:
                grades.append(("🟡 Liquidity", "Adequate"))
        
        debt_equity = self.info.get('debtToEquity')
        if debt_equity:
            if debt_equity < 0.5:
                grades.append(("🟢 Debt", "Low"))
            elif debt_equity > 1:
                grades.append(("🔴 Debt", "High"))
            else:
                grades.append(("🟡 Debt", "Moderate"))
        
        return grades

# ==================== ENHANCED CHART FUNCTIONS ====================

def create_enhanced_candlestick_chart(df, analyzer):
    """Create professional candlestick chart with advanced indicators"""
    
    # Create subplot layout
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(
            "PRICE ACTION & TECHNICAL INDICATORS", 
            "VOLUME ANALYSIS", 
            "RSI MOMENTUM",
            "MACD SIGNALS"
        ),
        vertical_spacing=0.05
    )
    
    # 1. Main Price Chart
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ), row=1, col=1)
    
    # EMAs
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA12"], line=dict(color="#00b894", width=2), name="EMA 12"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA26"], line=dict(color="#ff7675", width=2), name="EMA 26"), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], line=dict(color="rgba(255,255,255,0.3)", dash='dash'), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], line=dict(color="rgba(255,255,255,0.3)", dash='dash'), name="BB Lower", fill='tonexty', fillcolor='rgba(255,255,255,0.1)'), row=1, col=1)
    
    # Price Channels
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Price_Channel_High"], line=dict(color="#e17055", width=1, dash='dot'), name="20D High"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Price_Channel_Low"], line=dict(color="#e17055", width=1, dash='dot'), name="20D Low"), row=1, col=1)
    
    # 2. Volume Analysis
    vol_colors = ['#00b894' if df["Close"].iloc[i] >= df["Open"].iloc[i] else '#ff7675' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], marker_color=vol_colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Volume_SMA"], line=dict(color="#fdcb6e", width=2), name="Vol SMA 20"), row=2, col=1)
    
    # 3. RSI
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], line=dict(color="#6c5ce7", width=2), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    # 4. MACD
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], line=dict(color="#00b894", width=2), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], line=dict(color="#ff7675", width=2), name="MACD Signal"), row=4, col=1)
    
    # MACD Histogram
    colors_hist = ['#00b894' if val >= 0 else '#ff7675' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"], marker_color=colors_hist, name="MACD Histogram", opacity=0.6), row=4, col=1)
    
    # Update layout
    fig.update_layout(
        height=1000,
        template="plotly_dark",
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def create_technical_metrics_panel(df, analyzer):
    """Create comprehensive technical metrics dashboard"""
    latest = df.iloc[-1]
    
    st.markdown("### 📊 TECHNICAL METRICS DASHBOARD")
    
    # Create columns for different metric groups
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🎯 PRICE ACTION**")
        st.metric("Close Price", f"₹{latest['Close']:.2f}")
        st.metric("Daily Change", f"{(latest['Close'] - latest['Open']):.2f}")
        st.metric("VWAP", f"₹{latest['VWAP']:.2f}")
        st.metric("20D High/Low", f"₹{latest['Price_Channel_High']:.2f}/₹{latest['Price_Channel_Low']:.2f}")
        
    with col2:
        st.markdown("**📈 MOMENTUM**")
        rsi_color = "🟢" if latest['RSI'] < 30 else "🔴" if latest['RSI'] > 70 else "🟡"
        st.metric("RSI (14)", f"{latest['RSI']:.1f}", delta=rsi_color)
        st.metric("Momentum", f"{latest['Momentum']:.2f}")
        st.metric("MACD", f"{latest['MACD']:.3f}")
        st.metric("Stochastic", f"{latest['Stoch_K']:.1f}" if 'Stoch_K' in latest else "N/A")
        
    with col3:
        st.markdown("**📊 TREND**")
        ema_trend = "🟢 Bullish" if latest['EMA12'] > latest['EMA26'] else "🔴 Bearish"
        st.metric("EMA Trend", ema_trend)
        st.metric("EMA 12/26", f"₹{latest['EMA12']:.2f}/₹{latest['EMA26']:.2f}")
        bb_position = "Upper" if latest['Close'] > latest['BB_Upper'] else "Lower" if latest['Close'] < latest['BB_Lower'] else "Middle"
        st.metric("BB Position", bb_position)
        st.metric("Trend Strength", f"{latest['Trend_Strength']:.4f}")
        
    with col4:
        st.markdown("**💎 VOLUME**")
        st.metric("Volume", f"{latest['Volume']:,.0f}")
        vol_ratio = latest['Volume_Ratio']
        vol_signal = "🟢 High" if vol_ratio > 1.5 else "🔴 Low" if vol_ratio < 0.7 else "🟡 Normal"
        st.metric("Volume Ratio", f"{vol_ratio:.2f}x", delta=vol_signal)
        st.metric("Volume SMA", f"{latest['Volume_SMA']:,.0f}")
        st.metric("OBV", f"{latest['OBV']:,.0f}" if 'OBV' in latest else "N/A")

# ==================== FinBERT INTEGRATION ====================

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    
    class FinBERTAnalyzer:
        """Professional FinBERT sentiment analyzer for financial news"""
        
        def __init__(self):
            st.info("🔄 Loading FinBERT AI Model...")
            self.model_name = "ProsusAI/finbert"
            
            # Use GPU if available
            self.device = 0 if torch.cuda.is_available() else -1
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline for easy use
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
        def analyze_sentiment(self, text, title=""):
            """Analyze sentiment using FinBERT"""
            if not text or len(text.strip()) < 10:
                return self._default_sentiment()
            
            try:
                # Combine title and text for better context
                full_text = f"{title}. {text}" if title else text
                
                # Truncate if too long (FinBERT has 512 token limit)
                if len(full_text) > 2000:
                    full_text = full_text[:2000]
                
                # Get prediction
                result = self.classifier(full_text)[0]
                
                # Map to our format
                sentiment_map = {
                    'positive': 'POSITIVE',
                    'negative': 'NEGATIVE', 
                    'neutral': 'NEUTRAL'
                }
                
                return {
                    'sentiment': sentiment_map.get(result['label'], 'NEUTRAL'),
                    'confidence': result['score'],
                    'model': 'FinBERT',
                    'pos_count': 0,
                    'neg_count': 0,
                    'neu_count': 0,
                    'pos_keywords': [],
                    'neg_keywords': []
                }
                
            except Exception as e:
                st.error(f"⚠️ FinBERT analysis failed: {e}")
                return self._default_sentiment()
        
        def _default_sentiment(self):
            """Return default neutral sentiment"""
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.5,
                'model': 'FinBERT',
                'pos_count': 0,
                'neg_count': 0,
                'neu_count': 0,
                'pos_keywords': [],
                'neg_keywords': []
            }
    
    # Initialize FinBERT
    finbert_analyzer = FinBERTAnalyzer()
    ML_AVAILABLE = True
    
except Exception as e:
    st.warning(f"🤖 FinBERT not available: {e}")
    st.info("🔧 Install with: `pip install transformers torch`")
    ML_AVAILABLE = False
    finbert_analyzer = None

# ==================== NEWS FETCHING ====================

class NewsFetcher:
    """Professional news fetcher with multiple sources"""
    
    def __init__(self, stock_name):
        self.stock = stock_name.upper()
        self.stock_lower = stock_name.lower()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_rss_news(self):
        """Get news from RSS feeds"""
        articles = []
        
        try:
            # Google News RSS
            feed_url = f"https://news.google.com/rss/search?q={quote(self.stock)}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:15]:
                articles.append({
                    'title': entry.title,
                    'url': entry.link,
                    'published': entry.get('published', 'N/A'),
                    'source': 'Google News'
                })
                
        except Exception as e:
            st.error(f"⚠️ RSS fetch failed: {e}")
        
        return articles
    
    def get_yahoo_news(self):
        """Get news from Yahoo Finance"""
        articles = []
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={self.stock}&region=US&lang=en-US"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:10]:
                articles.append({
                    'title': entry.title,
                    'url': entry.link,
                    'published': entry.get('published', 'N/A'),
                    'source': 'Yahoo Finance'
                })
        except:
            pass
        
        return articles
    
    def fetch_news(self):
        """Main news fetching method"""
        all_articles = []
        
        # Fetch from multiple sources
        all_articles.extend(self.get_rss_news())
        all_articles.extend(self.get_yahoo_news())
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        return unique_articles[:20]  # Limit to 20 articles

# ==================== CONTENT EXTRACTOR ====================

class ContentExtractor:
    """Extract article content"""
    
    @staticmethod
    def extract_content(url, title=""):
        """Extract main content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            resp = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            
            # Try to find main content
            article = soup.find('article')
            if article:
                return article.get_text(separator=' ', strip=True)[:3000]
            
            # Fallback: get all paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text(strip=True) for p in paragraphs[:10]])
            return text[:3000] if text else title * 3
            
        except:
            return title * 3  # Use title as fallback

# ==================== SENTIMENT ANALYSIS ====================

def analyze_news_sentiment(articles, use_ml=True):
    """Analyze news sentiment with optional ML enhancement"""
    analyzed_articles = []
    extractor = ContentExtractor()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, article in enumerate(articles):
        status_text.text(f"📰 Analyzing article {i+1}/{len(articles)}...")
        progress_bar.progress((i + 1) / len(articles))
        
        # Extract content
        content = extractor.extract_content(article['url'], article['title'])
        
        # Analyze sentiment
        if use_ml and ML_AVAILABLE:
            analysis = finbert_analyzer.analyze_sentiment(content, article['title'])
        else:
            # Fallback to basic analysis
            analysis = basic_sentiment_analysis(content, article['title'])
        
        analyzed_articles.append({
            'title': article['title'],
            'url': article['url'],
            'source': article['source'],
            'published': article['published'],
            'sentiment': analysis['sentiment'],
            'confidence': analysis['confidence'],
            'model': analysis.get('model', 'Basic')
        })
        
        time.sleep(0.2)  # Be polite to servers
    
    progress_bar.empty()
    status_text.empty()
    
    return analyzed_articles

def basic_sentiment_analysis(text, title=""):
    """Basic sentiment analysis as fallback"""
    full_text = f"{title} {text}".lower()
    
    positive_words = ['bullish', 'growth', 'profit', 'gain', 'positive', 'strong', 'beat', 'up', 'buy']
    negative_words = ['bearish', 'fall', 'loss', 'negative', 'weak', 'miss', 'down', 'sell', 'drop']
    
    pos_count = sum(1 for word in positive_words if word in full_text)
    neg_count = sum(1 for word in negative_words if word in full_text)
    
    if pos_count > neg_count:
        sentiment = 'POSITIVE'
        confidence = min(0.3 + (pos_count - neg_count) * 0.1, 0.9)
    elif neg_count > pos_count:
        sentiment = 'NEGATIVE'
        confidence = min(0.3 + (neg_count - pos_count) * 0.1, 0.9)
    else:
        sentiment = 'NEUTRAL'
        confidence = 0.5
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'model': 'Basic'
    }

# ==================== MAIN STOCK ANALYSIS ====================

def show_stock_analysis():
    """Display comprehensive stock analysis page"""
    
    ticker = st.session_state.get('ticker', 'RELIANCE.NS')
    
    # Header with back button
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
    with col2:
        st.markdown(f"<h1 style='text-align:center; color:#00b894;'>📊 {ticker} Analysis</h1>", unsafe_allow_html=True)
    with col3:
        if st.button("🤖 AI Agents", type="primary", use_container_width=True):
            st.session_state.page = 'ai_agents'
            st.rerun()
    
    # Fetch basic info
    with st.spinner("Loading stock data..."):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            company_name = info.get('longName', ticker)
            
            st.markdown(f"<h3 style='text-align:center; color:#888;'>{company_name}</h3>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading stock data: {str(e)}")
            return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Technical Analysis", "💼 Fundamental Analysis", "📰 Market News", "📊 Charts & Data"])
    
    # ========================================
    # TAB 1: ENHANCED TECHNICAL ANALYSIS
    # ========================================
    with tab1:
        st.markdown("## 🎯 Professional Technical Analysis")
        
        # Fetch technical data
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        with st.spinner("Calculating advanced technical indicators..."):
            try:
                df = fetch_and_enhance(ticker, start_date, end_date)
                
                # Initialize advanced technical analyzer
                analyzer = AdvancedTechnicalAnalyzer(df)
                df = analyzer.df
                latest = df.iloc[-1]
                
                st.success(f"✅ Loaded {len(df):,} trading days with advanced indicators")
            except Exception as e:
                st.error(f"Error calculating indicators: {str(e)}")
                return
        
        # Technical Metrics Dashboard
        create_technical_metrics_panel(df, analyzer)
        
        st.markdown("---")
        
        # Enhanced Candlestick Chart
        st.markdown("### 📊 Advanced Candlestick Chart")
        
        # Chart configuration
        col1, col2 = st.columns([3, 1])
        with col2:
            show_volume = st.checkbox("Show Volume", value=True)
            show_indicators = st.checkbox("Show Indicators", value=True)
            if st.button("🔄 Refresh Chart", use_container_width=True):
                st.rerun()
        
        fig = create_enhanced_candlestick_chart(df, analyzer)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Trading Signals
        st.markdown("### 📈 Advanced Trading Signals")
        
        signals = analyzer.get_enhanced_signals()
        
        # Display signals in a grid
        cols = st.columns(2)
        for i, (signal_type, category, description) in enumerate(signals):
            with cols[i % 2]:
                with st.container():
                    st.markdown(f"""
                    <div style='background-color: #2d3436; padding: 15px; border-radius: 10px; border-left: 4px solid #00b894; margin: 5px 0;'>
                        <div style='font-size: 16px; font-weight: bold; color: #00b894;'>{signal_type}</div>
                        <div style='font-size: 12px; color: #b2bec3;'>Category: {category}</div>
                        <div style='font-size: 14px; color: #dfe6e9; margin-top: 5px;'>{description}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Analysis Summary
        st.markdown("### 🎯 Quick Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Trend Analysis
            if latest['EMA12'] > latest['EMA26'] and latest['Close'] > latest['EMA12']:
                trend = "🟢 Strong Uptrend"
            elif latest['EMA12'] < latest['EMA26'] and latest['Close'] < latest['EMA12']:
                trend = "🔴 Strong Downtrend"
            else:
                trend = "🟡 Sideways"
            st.metric("Primary Trend", trend)
        
        with col2:
            # Momentum Analysis
            if latest['RSI'] < 30:
                momentum = "🟢 Oversold"
            elif latest['RSI'] > 70:
                momentum = "🔴 Overbought"
            else:
                momentum = "🟡 Neutral"
            st.metric("Momentum", momentum)
        
        with col3:
            # Volume Analysis
            if latest['Volume_Ratio'] > 1.5:
                volume_signal = "🟢 High Interest"
            elif latest['Volume_Ratio'] < 0.7:
                volume_signal = "🔴 Low Interest"
            else:
                volume_signal = "🟡 Normal"
            st.metric("Volume Signal", volume_signal)
    
    # ========================================
    # TAB 2: ENHANCED FUNDAMENTAL ANALYSIS
    # ========================================
    with tab2:
        st.markdown("## 💼 Advanced Fundamental Analysis")
        
        with st.spinner("Loading comprehensive fundamental data..."):
            try:
                # Initialize advanced fundamental analyzer
                fundamental_analyzer = AdvancedFundamentalAnalyzer(stock)
                
                # Company Overview
                st.markdown("### 🏢 Company Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sector", info.get('sector', 'N/A'))
                with col2:
                    st.metric("Industry", info.get('industry', 'N/A'))
                with col3:
                    st.metric("Market Cap", fundamental_analyzer.valuation_metrics['Market Cap'])
                with col4:
                    st.metric("Employees", f"{info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "N/A")
                
                st.markdown("---")
                
                # Valuation Analysis
                st.markdown("### 💰 Valuation Analysis")
                
                # Valuation Grades
                valuation_grades = fundamental_analyzer.get_valuation_grade()
                if valuation_grades:
                    st.markdown("#### 📊 Valuation Grades")
                    grade_cols = st.columns(len(valuation_grades))
                    for i, (grade, description) in enumerate(valuation_grades):
                        with grade_cols[i]:
                            st.metric(description, grade)
                
                # Valuation Metrics
                st.markdown("#### 📈 Valuation Metrics")
                val_cols = st.columns(4)
                val_metrics = list(fundamental_analyzer.valuation_metrics.items())[:8]
                for i, (key, value) in enumerate(val_metrics):
                    with val_cols[i % 4]:
                        if value is not None:
                            if isinstance(value, (int, float)):
                                if key in ['Trailing P/E', 'Forward P/E', 'PEG Ratio', 'Price/Sales', 'Price/Book']:
                                    st.metric(key, f"{value:.2f}")
                                else:
                                    st.metric(key, value)
                            else:
                                st.metric(key, value)
                
                st.markdown("---")
                
                # Profitability Analysis
                st.markdown("### 📊 Profitability & Efficiency")
                
                prof_cols = st.columns(4)
                prof_metrics = list(fundamental_analyzer.profitability_metrics.items())
                for i, (key, value) in enumerate(prof_metrics):
                    with prof_cols[i % 4]:
                        if value is not None:
                            st.metric(key, f"{value:.2f}%")
                
                st.markdown("---")
                
                # Growth Analysis
                st.markdown("### 📈 Growth Metrics")
                
                growth_cols = st.columns(3)
                growth_metrics = list(fundamental_analyzer.growth_metrics.items())
                for i, (key, value) in enumerate(growth_metrics):
                    with growth_cols[i % 3]:
                        if value is not None:
                            st.metric(key, f"{value:+.2f}%")
                
                st.markdown("---")
                
                # Financial Health
                st.markdown("### 🏦 Financial Health")
                
                # Health Grades
                health_grades = fundamental_analyzer.get_financial_health_grade()
                if health_grades:
                    st.markdown("#### 📊 Health Assessment")
                    health_cols = st.columns(len(health_grades))
                    for i, (grade, description) in enumerate(health_grades):
                        with health_cols[i]:
                            st.metric(description, grade)
                
                # Health Metrics
                st.markdown("#### 🔧 Financial Metrics")
                health_metrics_cols = st.columns(4)
                health_metrics = list(fundamental_analyzer.health_metrics.items())
                for i, (key, value) in enumerate(health_metrics):
                    with health_metrics_cols[i % 4]:
                        if value is not None:
                            if isinstance(value, (int, float)):
                                st.metric(key, f"{value:.2f}")
                            else:
                                st.metric(key, value)
                
                st.markdown("---")
                
                # Dividend Information
                st.markdown("### 💵 Dividend Analysis")
                
                div_cols = st.columns(4)
                div_metrics = list(fundamental_analyzer.dividend_metrics.items())
                for i, (key, value) in enumerate(div_metrics):
                    with div_cols[i % 4]:
                        if value is not None:
                            if isinstance(value, (int, float)):
                                st.metric(key, f"{value:.2f}")
                            else:
                                st.metric(key, value)
                
                st.markdown("---")
                
                # Financial Statements with Visualizations
                st.markdown("### 📄 Financial Statements")
                
                # Income Statement
                try:
                    income_stmt = stock.income_stmt
                    if not income_stmt.empty:
                        with st.expander("📊 Income Statement Trend", expanded=True):
                            # Get recent years
                            recent_cols = [col for col in income_stmt.columns if col.year >= 2020][:4]
                            if len(recent_cols) > 1:
                                # Plot key metrics
                                metrics_to_plot = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
                                available_metrics = [m for m in metrics_to_plot if m in income_stmt.index]
                                
                                if available_metrics:
                                    fig_income = go.Figure()
                                    for metric in available_metrics:
                                        values = [income_stmt.loc[metric, col] for col in recent_cols]
                                        fig_income.add_trace(go.Scatter(
                                            x=[col.year for col in recent_cols],
                                            y=values,
                                            name=metric,
                                            mode='lines+markers'
                                        ))
                                    
                                    fig_income.update_layout(
                                        title="Income Statement Trend",
                                        template="plotly_dark",
                                        height=400
                                    )
                                    st.plotly_chart(fig_income, use_container_width=True)
                except:
                    pass
                
                # Additional fundamental data sections
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("📋 Key Statistics"):
                        try:
                            key_stats = {
                                "52 Week High": info.get('fiftyTwoWeekHigh'),
                                "52 Week Low": info.get('fiftyTwoWeekLow'),
                                "Beta": info.get('beta'),
                                "Short Ratio": info.get('shortRatio'),
                                "Float Shares": info.get('floatShares'),
                                "Shares Outstanding": info.get('sharesOutstanding'),
                            }
                            
                            for stat, value in key_stats.items():
                                if value is not None:
                                    if isinstance(value, (int, float)):
                                        if stat in ['52 Week High', '52 Week Low']:
                                            st.metric(stat, f"₹{value:.2f}")
                                        else:
                                            st.metric(stat, f"{value:.2f}")
                        except:
                            st.info("Key statistics not available")
                
                with col2:
                    with st.expander("🎯 Analyst Targets"):
                        try:
                            analyst_stats = {
                                "Target Mean": info.get('targetMeanPrice'),
                                "Target High": info.get('targetHighPrice'),
                                "Target Low": info.get('targetLowPrice'),
                                "Recommendation": info.get('recommendationKey', '').title(),
                                # "Number of Analysts": info.get('numberOfAnalystOpinions'),
                            }
                            
                            for stat, value in analyst_stats.items():
                                if value is not None:
                                    if isinstance(value, (int, float)):
                                        st.metric(stat, f"${value:.2f}")
                                    else:
                                        st.metric(stat, value)
                        except:
                            st.info("Analyst targets not available")
                
            except Exception as e:
                st.error(f"Error loading fundamental data: {str(e)}")
    
    # ========================================
    # TAB 3: MARKET NEWS - FIXED VERSION WITH FinBERT
    # ========================================
    with tab3:
        st.markdown("## 📰 Latest Market News & Sentiment")
        
        # AI Model Selection
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            use_ml = st.checkbox("🤖 Use AI (FinBERT) for Sentiment Analysis", value=ML_AVAILABLE, 
                               disabled=not ML_AVAILABLE,
                               help="Uses advanced AI model for more accurate sentiment analysis")
        with col2:
            num_articles = st.selectbox("Articles", [10, 15, 20], index=1)
        with col3:
            if st.button("🔄 Refresh News", use_container_width=True):
                st.session_state.news_refresh = True
        
        if not ML_AVAILABLE and use_ml:
            st.warning("🤖 FinBERT not available. Using basic sentiment analysis. Install with: `pip install transformers torch`")
            use_ml = False
        
        with st.spinner("Fetching latest news..."):
            try:
                # Initialize news fetcher
                news_fetcher = NewsFetcher(ticker)
                articles = news_fetcher.fetch_news()[:num_articles]
                
                if not articles:
                    st.warning("No news articles found for this stock.")
                else:
                    st.success(f"📰 Found {len(articles)} news articles")
                    
                    # Analyze sentiment
                    analyzed_articles = analyze_news_sentiment(articles, use_ml=use_ml)
                    
                    # Display sentiment summary
                    sentiments = [a['sentiment'] for a in analyzed_articles]
                    sentiment_counts = Counter(sentiments)
                    
                    total = len(analyzed_articles)
                    positive = sentiment_counts.get('POSITIVE', 0)
                    negative = sentiment_counts.get('NEGATIVE', 0)
                    neutral = sentiment_counts.get('NEUTRAL', 0)
                    
                    # Sentiment Overview
                    st.markdown("### 📊 Sentiment Overview")
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("🟢 Positive", f"{positive}/{total}", f"{(positive/total*100):.1f}%")
                    with cols[1]:
                        st.metric("🔴 Negative", f"{negative}/{total}", f"{(negative/total*100):.1f}%")
                    with cols[2]:
                        st.metric("🟡 Neutral", f"{neutral}/{total}", f"{(neutral/total*100):.1f}%")
                    with cols[3]:
                        model_used = "FinBERT AI" if use_ml else "Basic"
                        st.metric("🤖 Model", model_used)
                    
                    # Trading Signal
                    st.markdown("### 🎯 Trading Signal")
                    
                    if positive > negative and positive > neutral:
                        signal = "🟢 BULLISH"
                        reasoning = "Positive news sentiment dominates"
                        color = "green"
                    elif negative > positive and negative > neutral:
                        signal = "🔴 BEARISH" 
                        reasoning = "Negative news sentiment dominates"
                        color = "red"
                    else:
                        signal = "🟡 NEUTRAL"
                        reasoning = "Mixed or neutral sentiment"
                        color = "orange"
                    
                    st.markdown(f"""
                    <div style='background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color};'>
                        <h3 style='color: {color}; margin: 0;'>{signal}</h3>
                        <p style='margin: 5px 0 0 0; color: #666;'>{reasoning}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Display Articles
                    st.markdown("### 📝 News Articles")
                    
                    for i, article in enumerate(analyzed_articles):
                        sentiment_icon = {
                            'POSITIVE': '🟢',
                            'NEGATIVE': '🔴',
                            'NEUTRAL': '🟡'
                        }[article['sentiment']]
                        
                        confidence_color = "#00b894" if article['confidence'] > 0.7 else "#fdcb6e" if article['confidence'] > 0.5 else "#ff7675"
                        
                        with st.expander(f"{sentiment_icon} {article['title'][:80]}...", expanded=i < 2):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{article['title']}**")
                                st.caption(f"📰 Source: {article['source']} | 🕒 {article['published']}")
                            with col2:
                                st.markdown(f"**Sentiment:** {article['sentiment']}")
                                st.markdown(f"<span style='color: {confidence_color};'>Confidence: {article['confidence']:.1%}</span>", unsafe_allow_html=True)
                                st.markdown(f"*Model: {article['model']}*")
                            
                            st.link_button("📖 Read Full Article", article['url'])
                            
                            if i < len(analyzed_articles) - 1:
                                st.markdown("---")
            
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")
    
    # ========================================
    # TAB 4: ADDITIONAL CHARTS & DATA
    # ========================================
    with tab4:
        st.markdown("## 📊 Additional Charts & Data")
        
        try:
            # Add more technical indicators
            col1, col2 = st.columns(2)
            
            with col1:
                # Stochastic Oscillator
                if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
                    fig_stoch = go.Figure()
                    fig_stoch.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_K"], name="Stochastic %K", line=dict(color="#00b894")))
                    fig_stoch.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_D"], name="Stochastic %D", line=dict(color="#ff7675")))
                    fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
                    fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
                    fig_stoch.update_layout(title="Stochastic Oscillator", template="plotly_dark", height=300)
                    st.plotly_chart(fig_stoch, use_container_width=True)
                else:
                    st.info("Stochastic data not available")
            
            with col2:
                # OBV
                if 'OBV' in df.columns:
                    fig_obv = go.Figure()
                    fig_obv.add_trace(go.Scatter(x=df["Date"], y=df["OBV"], name="On-Balance Volume", line=dict(color="#6c5ce7")))
                    fig_obv.update_layout(title="On-Balance Volume (OBV)", template="plotly_dark", height=300)
                    st.plotly_chart(fig_obv, use_container_width=True)
                else:
                    st.info("OBV data not available")
            
            # Data table
            st.markdown("### 📋 Recent Data (Last 20 Days)")
            display_cols = [col for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD'] if col in df.columns]
            st.dataframe(df[display_cols].tail(20), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error displaying charts: {str(e)}")

# Quick test function
def test_finbert():
    """Test FinBERT with sample text"""
    if not ML_AVAILABLE:
        st.warning("FinBERT not available!")
        return
    
    test_texts = [
        "Company reports strong earnings growth and exceeding expectations",
        "Stock plunges as company misses revenue targets",
        "Board announces quarterly dividend payment",
        "Company faces regulatory investigation and potential fines"
    ]
    
    st.write("### 🧪 FinBERT Test Results:")
    
    for text in test_texts:
        result = finbert_analyzer.analyze_sentiment(text)
        icon = '🟢' if result['sentiment'] == 'POSITIVE' else '🔴' if result['sentiment'] == 'NEGATIVE' else '🟡'
        st.write(f"{icon} **{text}** → {result['sentiment']} ({result['confidence']:.1%})")