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


# ui/stock_analysis.py - FIXED VERSION

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from main import fetch_and_enhance
from data.news_fetcher import NewsFetcher
from datetime import datetime, timedelta
import logging  # ✅ FIXED: Add this import

# ✅ FIXED: Setup logger
logger = logging.getLogger(__name__)

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
            logger.error(f"Stock data error: {e}")
            return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Technical Analysis", "💼 Fundamental Analysis", "📰 Market News", "📊 Charts & Data"])
    
    # ========================================
    # TAB 1: TECHNICAL ANALYSIS
    # ========================================
    with tab1:
        st.markdown("## Technical Indicators & Metrics")
        
        # Fetch technical data
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        with st.spinner("Calculating technical indicators..."):
            try:
                df = fetch_and_enhance(ticker, start_date, end_date)
                latest = df.iloc[-1]
                st.success(f"✅ Loaded {len(df):,} trading days")
            except Exception as e:
                st.error(f"Error calculating indicators: {str(e)}")
                logger.error(f"Technical data error: {e}")
                return
        
        # Key Technical Metrics
        cols = st.columns(6)
        with cols[0]:
            st.metric("Close Price", f"₹{latest['Close']:.2f}")
        with cols[1]:
            rsi_color = "🟢" if latest['RSI'] < 30 else "🔴" if latest['RSI'] > 70 else "🟡"
            st.metric("RSI", f"{latest['RSI']:.1f} {rsi_color}")
        with cols[2]:
            macd_signal = "🟢 Bullish" if latest['MACD'] > latest['MACD_Signal'] else "🔴 Bearish"
            st.metric("MACD", f"{latest['MACD']:.2f}", macd_signal)
        with cols[3]:
            st.metric("VWAP", f"₹{latest['VWAP']:.2f}")
        with cols[4]:
            st.metric("Volume", f"{latest['Volume']:,.0f}")
        with cols[5]:
            change_pct = ((latest['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100
            st.metric("YoY Change", f"{change_pct:+.2f}%")
        
        st.markdown("---")
        
        # Technical Signals
        st.markdown("### 📊 Technical Signals")
        
        signals = []
        
        # RSI Signals
        if latest['RSI'] < 30:
            signals.append(("🟢 BUY Signal", "RSI Oversold", f"RSI at {latest['RSI']:.1f} indicates oversold conditions"))
        elif latest['RSI'] > 70:
            signals.append(("🔴 SELL Signal", "RSI Overbought", f"RSI at {latest['RSI']:.1f} indicates overbought conditions"))
        
        # MACD Signals
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > 0:
            signals.append(("🟢 BUY Signal", "MACD Bullish Crossover", "MACD crossed above signal line"))
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < 0:
            signals.append(("🔴 SELL Signal", "MACD Bearish Crossover", "MACD crossed below signal line"))
        
        # Bollinger Bands
        if latest['Close'] < latest['BB_Lower']:
            signals.append(("🟢 BUY Signal", "Price Below Lower BB", "Potential bounce from support"))
        elif latest['Close'] > latest['BB_Upper']:
            signals.append(("🔴 SELL Signal", "Price Above Upper BB", "Potential reversal from resistance"))
        
        # EMA Trend
        if latest['EMA12'] > latest['EMA26']:
            signals.append(("🟢 Bullish Trend", "EMA Crossover", "Short-term EMA above long-term EMA"))
        else:
            signals.append(("🔴 Bearish Trend", "EMA Crossover", "Short-term EMA below long-term EMA"))
        
        for signal_type, signal_name, description in signals:
            cols = st.columns([1, 2, 4])
            with cols[0]:
                st.markdown(f"**{signal_type}**")
            with cols[1]:
                st.markdown(f"`{signal_name}`")
            with cols[2]:
                st.markdown(description)
        
        st.markdown("---")
        
        # Candlestick Chart
        st.markdown("### 📈 Price Action Chart")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=("Price & Indicators", "RSI", "Volume"),
            vertical_spacing=0.05
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], line=dict(color="rgba(255,255,255,0.2)", dash='dash'), name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], line=dict(color="rgba(255,255,255,0.2)", dash='dash'), name="BB Lower", fill='tonexty', fillcolor='rgba(255,255,255,0.05)', showlegend=False), row=1, col=1)
        
        # EMAs
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA12"], line=dict(color="#00b894", width=2), name="EMA 12"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA26"], line=dict(color="#ff7675", width=2), name="EMA 26"), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], line=dict(color="#6c5ce7", width=2), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,118,117,0.5)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,184,148,0.5)", row=2, col=1)
        
        # Volume
        vol_colors = ['#00b894' if df["Close"].iloc[i] >= df["Open"].iloc[i] else '#ff7675' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], marker_color=vol_colors, name="Volume"), row=3, col=1)
        
        fig.update_layout(height=800, template="plotly_dark", showlegend=True, hovermode='x unified')
        fig.update_xaxes(rangeslider_visible=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # TAB 2: FUNDAMENTAL ANALYSIS
    # ========================================
    with tab2:
        st.markdown("## Fundamental Analysis")
        
        with st.spinner("Loading fundamental data..."):
            try:
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
                cashflow = stock.cashflow
                
                # Filter for recent years
                target_years = [str(year) for year in range(2022, 2026)]
                income_stmt = income_stmt[[col for col in income_stmt.columns if str(col.year) in target_years]] if not income_stmt.empty else income_stmt
                balance_sheet = balance_sheet[[col for col in balance_sheet.columns if str(col.year) in target_years]] if not balance_sheet.empty else balance_sheet
                cashflow = cashflow[[col for col in cashflow.columns if str(col.year) in target_years]] if not cashflow.empty else cashflow
                
                # Financial Ratios
                ratios = {
                    "P/E Ratio": info.get("trailingPE"),
                    "Forward P/E": info.get("forwardPE"),
                    "Price/Book": info.get("priceToBook"),
                    "Price/Sales": info.get("priceToSalesTrailing12Months"),
                    "ROE": info.get("returnOnEquity"),
                    "ROA": info.get("returnOnAssets"),
                    "Debt/Equity": info.get("debtToEquity") / 100 if info.get("debtToEquity") else None,
                    "Current Ratio": info.get("currentRatio"),
                    "Dividend Yield": info.get("dividendYield"),
                }
                
                st.markdown("### 💰 Key Financial Ratios")
                
                cols = st.columns(4)
                for idx, (key, value) in enumerate(ratios.items()):
                    if value is not None:
                        with cols[idx % 4]:
                            if key in ["ROE", "ROA", "Dividend Yield"]:
                                formatted_value = f"{value * 100:.2f}%"
                            else:
                                formatted_value = f"{value:.2f}"
                            st.metric(key, formatted_value)
                
                st.markdown("---")
                
                # Income Statement
                if not income_stmt.empty:
                    st.markdown("### 📊 Income Statement Highlights")
                    
                    latest_col = income_stmt.columns[0]
                    
                    cols = st.columns(3)
                    if "Total Revenue" in income_stmt.index:
                        with cols[0]:
                            st.metric("Total Revenue", f"₹{income_stmt.loc['Total Revenue', latest_col]:,.0f}")
                    if "Net Income" in income_stmt.index:
                        with cols[1]:
                            st.metric("Net Income", f"₹{income_stmt.loc['Net Income', latest_col]:,.0f}")
                    if "Diluted EPS" in income_stmt.index:
                        with cols[2]:
                            st.metric("EPS", f"₹{income_stmt.loc['Diluted EPS', latest_col]:.2f}")
                    
                    with st.expander("📄 View Full Income Statement"):
                        st.dataframe(income_stmt, use_container_width=True)
                
                # Balance Sheet
                if not balance_sheet.empty:
                    st.markdown("### 🏦 Balance Sheet Highlights")
                    
                    latest_bs = balance_sheet.iloc[:, 0]
                    
                    cols = st.columns(3)
                    if "Stockholders Equity" in balance_sheet.index:
                        with cols[0]:
                            st.metric("Total Equity", f"₹{latest_bs.get('Stockholders Equity'):,.0f}")
                    if "Total Debt" in balance_sheet.index:
                        with cols[1]:
                            st.metric("Total Debt", f"₹{latest_bs.get('Total Debt'):,.0f}")
                    if "Cash And Cash Equivalents" in balance_sheet.index:
                        with cols[2]:
                            st.metric("Cash", f"₹{latest_bs.get('Cash And Cash Equivalents'):,.0f}")
                    
                    with st.expander("📄 View Full Balance Sheet"):
                        st.dataframe(balance_sheet, use_container_width=True)
                
                # Cash Flow
                if not cashflow.empty:
                    st.markdown("### 💵 Cash Flow Highlights")
                    
                    latest_cf = cashflow.columns[0]
                    
                    cols = st.columns(3)
                    if "Operating Cash Flow" in cashflow.index:
                        with cols[0]:
                            st.metric("Operating Cash Flow", f"₹{cashflow.loc['Operating Cash Flow', latest_cf]:,.0f}")
                    if "Capital Expenditure" in cashflow.index:
                        with cols[1]:
                            st.metric("CapEx", f"₹{cashflow.loc['Capital Expenditure', latest_cf]:,.0f}")
                    if "Free Cash Flow" in cashflow.index:
                        with cols[2]:
                            st.metric("Free Cash Flow", f"₹{cashflow.loc['Free Cash Flow', latest_cf]:,.0f}")
                    
                    with st.expander("📄 View Full Cash Flow Statement"):
                        st.dataframe(cashflow, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading fundamental data: {str(e)}")
                logger.error(f"Fundamental data error: {e}")
    
    # ========================================
    # TAB 3: MARKET NEWS - FIXED VERSION
    # ========================================
    with tab3:
        st.markdown("## 📰 Latest Market News & Sentiment")
        
        with st.spinner("Fetching latest news..."):
            try:
                news_fetcher = NewsFetcher()
                
                # ✅ FIXED: Get sentiment summary
                sentiment_data = news_fetcher.get_sentiment_summary(ticker, company_name)
                
                # ✅ FIXED: Check all required keys exist
                sentiment = sentiment_data.get('sentiment', 'neutral')
                confidence = sentiment_data.get('confidence', 0)
                article_count = sentiment_data.get('article_count', 0)
                summary = sentiment_data.get('summary', 'No sentiment data')
                
                # Display sentiment
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment_emoji = "🟢" if sentiment == 'positive' else "🔴" if sentiment == 'negative' else "🟡"
                    st.metric("Overall Sentiment", f"{sentiment_emoji} {sentiment.upper()}")
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                with col3:
                    st.metric("Articles Analyzed", article_count)
                
                st.info(summary)
                
                st.markdown("---")
                
                # Get news articles
                news_articles = news_fetcher.get_stock_news(ticker, company_name, max_results=10)
                
                if news_articles:
                    st.markdown(f"### 📑 Latest News Articles ({len(news_articles)} found)")
                    
                    for idx, article in enumerate(news_articles, 1):
                        with st.expander(f"**{idx}. {article.get('title', 'No title')}**"):
                            st.markdown(f"**Published:** {article.get('published_date', 'Unknown')}")
                            st.markdown(f"**Source:** {article.get('source', 'Unknown')}")
                            st.markdown(f"**Relevance Score:** {article.get('score', 0):.2f}")
                            st.markdown(f"**Content:** {article.get('content', 'No content')}")
                            st.markdown(f"[Read Full Article]({article.get('url', '#')})")
                else:
                    st.warning("No news articles found for this stock.")
                    
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")
                logger.error(f"News fetching error: {e}")
    
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
            logger.error(f"Chart display error: {e}")

