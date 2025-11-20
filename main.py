
# # # main.py - COMPLETE FIXED VERSION

# # """
# # ProTrader AI - Main Trading System
# # Orchestrates all agents for trading analysis
# # """

# # import pandas as pd
# # import numpy as np
# # from datetime import datetime
# # from typing import Dict, List, Any
# # import logging

# # # Import data fetcher
# # from data.data_fetcher import fetch_data

# # # Import institutional aggregator
# # from agents.institutional_agents import MasterInstitutionalAggregator

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)


# # # ============================================
# # # TECHNICAL INDICATORS - Calculate Here
# # # ============================================

# # def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
# #     """Add technical indicators to DataFrame"""
    
# #     df = df.copy()
    
# #     if 'Date' in df.columns:
# #         df['Date'] = pd.to_datetime(df['Date'])
# #         df = df.set_index('Date')
    
# #     close = df["Close"]
# #     high = df["High"]
# #     low = df["Low"]
# #     vol = df["Volume"]

# #     # RSI
# #     delta = close.diff()
# #     gain = delta.where(delta > 0, 0)
# #     loss = -delta.where(delta < 0, 0)
# #     df["RSI"] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.finfo(float).eps))))

# #     # MACD
# #     ema12 = close.ewm(span=12, adjust=False).mean()
# #     ema26 = close.ewm(span=26, adjust=False).mean()
# #     df["MACD"] = ema12 - ema26
# #     df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
# #     df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

# #     # Bollinger Bands
# #     ma20 = close.rolling(20).mean()
# #     std20 = close.rolling(20).std()
# #     df["BB_Upper"] = ma20 + 2 * std20
# #     df["BB_Lower"] = ma20 - 2 * std20

# #     # Support / Resistance
# #     df["Support"] = low.rolling(20).min()
# #     df["Resistance"] = high.rolling(20).max()

# #     # EMA
# #     df["EMA12"] = ema12
# #     df["EMA26"] = ema26

# #     # VWAP
# #     typical = (high + low + close) / 3
# #     df["VWAP"] = (typical * vol).cumsum() / vol.cumsum()

# #     # Stochastic
# #     low14 = low.rolling(14).min()
# #     high14 = high.rolling(14).max()
# #     df["Stoch_K"] = 100 * (close - low14) / (high14 - low14)
# #     df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

# #     # OBV
# #     df["OBV"] = (np.sign(close.diff()) * vol).fillna(0).cumsum()

# #     # Aroon
# #     df["Aroon_Up"] = 100 * (25 - high.rolling(25).apply(lambda x: len(x) - x.argmax())) / 25
# #     df["Aroon_Down"] = 100 * (25 - low.rolling(25).apply(lambda x: len(x) - x.argmin())) / 25

# #     df = df.dropna().reset_index()
# #     return df


# # def fetch_and_enhance(ticker: str, start, end) -> pd.DataFrame:
# #     """Fetch data and add indicators"""
# #     df = fetch_data(ticker, start, end)
# #     return add_indicators(df)


# # # ============================================
# # # AGENT 1: TECHNICAL AGENT
# # # ============================================

# # class TechnicalAgent:
# #     """Analyzes technical indicators"""
# #     def __init__(self):
# #         self.name = "Technical Agent"
    
# #     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
# #         """Analyze technical indicators - returns BUY/SELL/HOLD"""
# #         latest = df.iloc[-1]
# #         score = 0
# #         signals = []
        
# #         try:
# #             # RSI Analysis
# #             rsi = latest.get('RSI', 50)
# #             if rsi < 30:
# #                 score += 2
# #                 signals.append("RSI oversold (bullish)")
# #             elif rsi > 70:
# #                 score -= 2
# #                 signals.append("RSI overbought (bearish)")
            
# #             # MACD Analysis
# #             macd = latest.get('MACD', 0)
# #             macd_signal = latest.get('MACD_Signal', 0)
# #             macd_hist = latest.get('MACD_Hist', 0)
            
# #             if macd > macd_signal and macd_hist > 0:
# #                 score += 2
# #                 signals.append("MACD bullish crossover")
# #             elif macd < macd_signal and macd_hist < 0:
# #                 score -= 2
# #                 signals.append("MACD bearish crossover")
            
# #             # Bollinger Bands
# #             close = latest.get('Close', 0)
# #             bb_lower = latest.get('BB_Lower', 0)
# #             bb_upper = latest.get('BB_Upper', 0)
            
# #             if close < bb_lower:
# #                 score += 1
# #                 signals.append("Price below lower BB")
# #             elif close > bb_upper:
# #                 score -= 1
# #                 signals.append("Price above upper BB")
            
# #             # EMA Trend
# #             ema12 = latest.get('EMA12', 0)
# #             ema26 = latest.get('EMA26', 0)
            
# #             if ema12 > ema26:
# #                 score += 1
# #                 signals.append("Short-term trend bullish")
# #             else:
# #                 score -= 1
# #                 signals.append("Short-term trend bearish")
            
# #             # Stochastic
# #             stoch_k = latest.get('Stoch_K', 50)
# #             if stoch_k < 20:
# #                 score += 1
# #                 signals.append("Stochastic oversold")
# #             elif stoch_k > 80:
# #                 score -= 1
# #                 signals.append("Stochastic overbought")
            
# #             # ✅ FIXED: Return ONLY "BUY", "SELL", "HOLD"
# #             if score >= 2:
# #                 action = "BUY"
# #             elif score <= -2:
# #                 action = "SELL"
# #             else:
# #                 action = "HOLD"
            
# #             confidence = min(abs(score) / 8 * 100, 100)
            
# #             return {
# #                 "agent": self.name,
# #                 "action": action,
# #                 "score": score,
# #                 "confidence": round(confidence, 1),
# #                 "signals": signals,
# #                 "key_metrics": {
# #                     "RSI": round(rsi, 2),
# #                     "MACD": round(macd, 2),
# #                     "Price_vs_VWAP": round((close - latest.get('VWAP', close)) / latest.get('VWAP', close) * 100, 2) if latest.get('VWAP') else 0
# #                 }
# #             }
        
# #         except Exception as e:
# #             logger.error(f"Error in technical analysis: {e}")
# #             return {
# #                 "agent": self.name,
# #                 "action": "HOLD",
# #                 "score": 0,
# #                 "confidence": 0,
# #                 "signals": [f"Error: {str(e)}"],
# #                 "key_metrics": {}
# #             }


# # # ============================================
# # # AGENT 2: SENTIMENT AGENT
# # # ============================================

# # class SentimentAgent:
# #     """Analyzes market sentiment"""
# #     def __init__(self):
# #         self.name = "Sentiment Agent"
    
# #     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
# #         """Analyze sentiment - returns BUY/SELL/HOLD"""
# #         try:
# #             avg_volume = df['Volume'].tail(20).mean()
# #             recent_volume = df['Volume'].tail(5).mean()
# #             volume_change = (recent_volume - avg_volume) / avg_volume * 100 if avg_volume > 0 else 0
            
# #             price_change_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100 if len(df) >= 6 else 0
# #             price_change_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100 if len(df) >= 21 else 0
            
# #             score = 0
# #             signals = []
            
# #             if volume_change > 20 and price_change_5d > 0:
# #                 score += 2
# #                 signals.append("Rising volume with price")
# #             elif volume_change > 20 and price_change_5d < 0:
# #                 score -= 2
# #                 signals.append("Rising volume with falling price")
            
# #             if price_change_20d > 10:
# #                 score += 1
# #                 signals.append("Strong uptrend")
# #             elif price_change_20d < -10:
# #                 score -= 1
# #                 signals.append("Strong downtrend")
            
# #             obv_change = 0
# #             if len(df) >= 21 and df['OBV'].iloc[-21] != 0:
# #                 obv_change = (df['OBV'].iloc[-1] - df['OBV'].iloc[-21]) / abs(df['OBV'].iloc[-21]) * 100
            
# #             if obv_change > 10:
# #                 score += 1
# #                 signals.append("Positive OBV trend")
# #             elif obv_change < -10:
# #                 score -= 1
# #                 signals.append("Negative OBV trend")
            
# #             # ✅ FIXED: Return ONLY "BUY", "SELL", "HOLD"
# #             if score >= 1:
# #                 action = "BUY"
# #             elif score <= -1:
# #                 action = "SELL"
# #             else:
# #                 action = "HOLD"
            
# #             confidence = min(abs(score) / 4 * 100, 100)
            
# #             return {
# #                 "agent": self.name,
# #                 "action": action,
# #                 "score": score,
# #                 "confidence": round(confidence, 1),
# #                 "signals": signals,
# #                 "key_metrics": {
# #                     "Volume_Change": round(volume_change, 2),
# #                     "Price_Change_5D": round(price_change_5d, 2),
# #                     "Price_Change_20D": round(price_change_20d, 2)
# #                 }
# #             }
        
# #         except Exception as e:
# #             logger.error(f"Error in sentiment analysis: {e}")
# #             return {
# #                 "agent": self.name,
# #                 "action": "HOLD",
# #                 "score": 0,
# #                 "confidence": 0,
# #                 "signals": [f"Error: {str(e)}"],
# #                 "key_metrics": {}
# #             }


# # # ============================================
# # # AGENT 3: RISK AGENT
# # # ============================================

# # class RiskAgent:
# #     """Assesses risk and volatility"""
# #     def __init__(self):
# #         self.name = "Risk Agent"
    
# #     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
# #         """Analyze risk"""
# #         try:
# #             returns = df['Close'].pct_change().dropna()
# #             volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
            
# #             cumulative = (1 + returns).cumprod()
# #             running_max = cumulative.expanding().max()
# #             drawdown = (cumulative - running_max) / running_max
# #             max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0
            
# #             latest = df.iloc[-1]
# #             support = latest.get('Support', latest['Close'])
# #             resistance = latest.get('Resistance', latest['Close'])
            
# #             distance_to_support = (latest['Close'] - support) / latest['Close'] * 100 if support > 0 else 100
# #             distance_to_resistance = (resistance - latest['Close']) / latest['Close'] * 100 if resistance > 0 else 100
            
# #             risk_score = 0
# #             signals = []
            
# #             if volatility < 20:
# #                 risk_score = 3
# #                 signals.append("Low volatility")
# #             elif volatility < 40:
# #                 risk_score = 2
# #                 signals.append("Moderate volatility")
# #             else:
# #                 risk_score = 1
# #                 signals.append("High volatility")
            
# #             if distance_to_support < 2:
# #                 signals.append("Near support")
# #                 risk_score += 1
# #             if distance_to_resistance < 2:
# #                 signals.append("Near resistance")
# #                 risk_score -= 1
            
# #             if volatility < 20:
# #                 position_size = "Normal (100%)"
# #             elif volatility < 40:
# #                 position_size = "Reduced (50%)"
# #             else:
# #                 position_size = "Minimal (25%)"
            
# #             return {
# #                 "agent": self.name,
# #                 "risk_level": "LOW" if risk_score >= 3 else "MEDIUM" if risk_score >= 2 else "HIGH",
# #                 "risk_score": risk_score,
# #                 "position_size": position_size,
# #                 "signals": signals,
# #                 "key_metrics": {
# #                     "Volatility": round(volatility, 2),
# #                     "Max_Drawdown": round(max_drawdown, 2),
# #                     "Distance_to_Support": round(distance_to_support, 2),
# #                     "Distance_to_Resistance": round(distance_to_resistance, 2)
# #                 }
# #             }
        
# #         except Exception as e:
# #             logger.error(f"Error in risk analysis: {e}")
# #             return {
# #                 "agent": self.name,
# #                 "risk_level": "MEDIUM",
# #                 "risk_score": 2,
# #                 "position_size": "Normal (100%)",
# #                 "signals": [f"Error: {str(e)}"],
# #                 "key_metrics": {}
# #             }


# # # ============================================
# # # AGENT 4: PORTFOLIO AGENT
# # # ============================================

# # class PortfolioAgent:
# #     """Manages portfolio allocation"""
# #     def __init__(self, portfolio_value: float = 1000000):
# #         self.name = "Portfolio Agent"
# #         self.portfolio_value = portfolio_value
    
# #     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
# #         """Calculate portfolio allocation"""
# #         try:
# #             latest = df.iloc[-1]
# #             current_price = latest['Close']
            
# #             risk_level = context.get('risk', {}).get('risk_level', 'MEDIUM') if context else 'MEDIUM'
            
# #             if risk_level == 'LOW':
# #                 allocation_pct = 10
# #             elif risk_level == 'MEDIUM':
# #                 allocation_pct = 5
# #             else:
# #                 allocation_pct = 2
            
# #             allocation_amount = self.portfolio_value * (allocation_pct / 100)
# #             suggested_quantity = int(allocation_amount / current_price) if current_price > 0 else 0
            
# #             signals = [
# #                 f"Allocation: {allocation_pct}% of portfolio",
# #                 f"Max position: ₹{allocation_amount:,.0f}",
# #                 f"Quantity: {suggested_quantity} shares"
# #             ]
            
# #             return {
# #                 "agent": self.name,
# #                 "allocation_pct": allocation_pct,
# #                 "allocation_amount": round(allocation_amount, 2),
# #                 "suggested_quantity": suggested_quantity,
# #                 "signals": signals,
# #                 "key_metrics": {
# #                     "Portfolio_Value": self.portfolio_value,
# #                     "Current_Price": round(current_price, 2)
# #                 }
# #             }
        
# #         except Exception as e:
# #             logger.error(f"Error in portfolio analysis: {e}")
# #             return {
# #                 "agent": self.name,
# #                 "allocation_pct": 5,
# #                 "allocation_amount": 0,
# #                 "suggested_quantity": 0,
# #                 "signals": [f"Error: {str(e)}"],
# #                 "key_metrics": {}
# #             }


# # # ============================================
# # # AGENT 5: MASTER AGENT
# # # ============================================

# # class MasterAgent:
# #     """Makes final trading decision"""
# #     def __init__(self):
# #         self.name = "Master Agent"
    
# #     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
# #         """Master decision - synthesizes all agents"""
# #         try:
# #             if not context:
# #                 return {
# #                     "agent": self.name,
# #                     "action": "HOLD",
# #                     "confidence": 0,
# #                     "reasoning": "No data",
# #                     "quantity": 0,
# #                     "risk_level": "MEDIUM",
# #                     "votes": {"BUY": 0, "SELL": 0, "HOLD": 5}
# #                 }
            
# #             technical = context.get('technical', {})
# #             sentiment = context.get('sentiment', {})
# #             risk = context.get('risk', {})
# #             institutional = context.get('institutional', {})
# #             portfolio = context.get('portfolio', {})
            
# #             votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
            
# #             # Get actions
# #             tech_action = technical.get('action', 'HOLD')
# #             sent_action = sentiment.get('action', 'HOLD')
            
# #             # Vote
# #             if tech_action in votes:
# #                 votes[tech_action] += 3
# #             else:
# #                 votes['HOLD'] += 3
            
# #             if sent_action in votes:
# #                 votes[sent_action] += 2
# #             else:
# #                 votes['HOLD'] += 2
            
# #             # Institutional signal
# #             inst_rec = institutional.get('recommendation', '')
# #             if "BUY" in inst_rec:
# #                 votes['BUY'] += 2.5
# #             elif "SELL" in inst_rec:
# #                 votes['SELL'] += 2.5
# #             else:
# #                 votes['HOLD'] += 2.5
            
# #             # Risk veto
# #             risk_level = risk.get('risk_level', 'MEDIUM')
# #             if risk_level == 'HIGH':
# #                 votes['BUY'] = max(0, votes['BUY'] - 2)
# #                 votes['HOLD'] += 3
            
# #             # Final decision
# #             final_action = max(votes, key=votes.get)
# #             total_votes = sum(votes.values())
# #             confidence = (votes[final_action] / total_votes * 100) if total_votes > 0 else 0
            
# #             reasoning = f"Tech: {tech_action} | Sentiment: {sent_action} | Inst: {institutional.get('recommendation', 'NEUTRAL')} | Risk: {risk_level}"
            
# #             return {
# #                 "agent": self.name,
# #                 "action": final_action,
# #                 "confidence": round(confidence, 1),
# #                 "reasoning": reasoning,
# #                 "votes": votes,
# #                 "risk_level": risk_level,
# #                 "quantity": portfolio.get('suggested_quantity', 0)
# #             }
        
# #         except Exception as e:
# #             logger.error(f"Error in master analysis: {e}")
# #             return {
# #                 "agent": self.name,
# #                 "action": "HOLD",
# #                 "confidence": 0,
# #                 "reasoning": f"Error: {str(e)}",
# #                 "quantity": 0,
# #                 "risk_level": "MEDIUM",
# #                 "votes": {"BUY": 0, "SELL": 0, "HOLD": 5}
# #             }


# # # ============================================
# # # TRADING SYSTEM
# # # ============================================

# # class TradingSystem:
# #     """Main trading system with all agents"""
    
# #     def __init__(self, portfolio_value: float = 1000000):
# #         self.agents = {
# #             'technical': TechnicalAgent(),
# #             'sentiment': SentimentAgent(),
# #             'risk': RiskAgent(),
# #             'portfolio': PortfolioAgent(portfolio_value),
# #             'master': MasterAgent()
# #         }
        
# #         # Institutional aggregator
# #         self.institutional_aggregator = MasterInstitutionalAggregator()
        
# #         self.analysis_log = []
    
# #     def run_analysis(self, ticker: str, start, end, 
# #                     fii_data=None, order_data=None, 
# #                     block_deals=None, holdings_data=None) -> Dict:
# #         """Run complete analysis"""
        
# #         logger.info(f"📊 Analyzing {ticker}...")
        
# #         try:
# #             # Fetch and enhance data
# #             df = fetch_and_enhance(ticker, start, end)
            
# #             if df is None or len(df) == 0:
# #                 logger.error("Failed to fetch data")
# #                 return {"error": "No data fetched", "ticker": ticker}
            
# #             # Run all agents
# #             context = {}
# #             context['technical'] = self.agents['technical'].analyze(df)
# #             context['sentiment'] = self.agents['sentiment'].analyze(df)
# #             context['risk'] = self.agents['risk'].analyze(df, context)
# #             context['portfolio'] = self.agents['portfolio'].analyze(df, context)
            
# #             # ✅ FIXED: Institutional analysis using .aggregate() method
# #             try:
# #                 institutional_signal = self.institutional_aggregator.aggregate(
# #                     price_data=df['Close'],
# #                     volume_data=df['Volume'],
# #                     fii_data=fii_data or {},
# #                     order_data=order_data or []
# #                 )
                
# #                 context['institutional'] = {
# #                     'final_score': institutional_signal.final_score,
# #                     'recommendation': institutional_signal.recommendation,
# #                     'confidence': institutional_signal.confidence
# #                 }
# #             except Exception as e:
# #                 logger.warning(f"Institutional analysis skipped: {e}")
# #                 context['institutional'] = {
# #                     'final_score': 50,
# #                     'recommendation': 'HOLD',
# #                     'confidence': 0
# #                 }
            
# #             # Master decision
# #             context['master'] = self.agents['master'].analyze(df, context)
            
# #             result = {
# #                 'ticker': ticker,
# #                 'timestamp': datetime.now(),
# #                 'data': df,
# #                 'recommendations': context,
# #                 'status': 'SUCCESS'
# #             }
            
# #             self.analysis_log.append(result)
# #             logger.info(f"✅ Analysis complete: {context['master']['action']}")
# #             return result
        
# #         except Exception as e:
# #             logger.error(f"Analysis error: {e}")
# #             return {
# #                 "error": str(e),
# #                 "ticker": ticker,
# #                 "status": "FAILED"
# #             }


# # # ============================================
# # # GLOBAL INSTANCE & HELPERS
# # # ============================================

# # _trading_system = None


# # def get_trading_system(portfolio_value: float = 1000000):
# #     """Get singleton instance"""
# #     global _trading_system
# #     if _trading_system is None:
# #         _trading_system = TradingSystem(portfolio_value)
# #     return _trading_system


# # def analyze_stock(ticker: str, start, end, portfolio_value: float = 1000000,
# #                  fii_data=None, order_data=None, block_deals=None, holdings_data=None):
# #     """Main analysis function"""
# #     system = get_trading_system(portfolio_value)
# #     return system.run_analysis(ticker, start, end, fii_data, order_data, block_deals, holdings_data)


# # def get_detailed_analysis(analysis_result):
# #     """Get detailed report"""
# #     return analysis_result


# # main.py - UPDATED WITH New News INTEGRATION

# """
# ProTrader AI - Main Trading System
# Orchestrates all agents for trading analysis
# """

# import os  # 🆕
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from typing import Dict, List, Any
# import logging

# # Import data fetcher
# from data.data_fetcher import fetch_data

# # Import institutional aggregator
# from agents.institutional_agents import MasterInstitutionalAggregator

# # 🆕 New News imports (safe-import to avoid breaking if files aren't present yet)
# try:
#     from data.new_news_fetcher import NewNewsFetcher
#     from agents.new_news_agent import NewNews
# except Exception:
#     NewNewsFetcher = None
#     NewNews = None

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # ============================================
# # TECHNICAL INDICATORS - Calculate Here
# # ============================================

# def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
#     """Add technical indicators to DataFrame"""
#     df = df.copy()
#     if 'Date' in df.columns:
#         df['Date'] = pd.to_datetime(df['Date'])
#         df = df.set_index('Date')
#     close = df["Close"]
#     high = df["High"]
#     low = df["Low"]
#     vol = df["Volume"]

#     # RSI
#     delta = close.diff()
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)
#     df["RSI"] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.finfo(float).eps))))

#     # MACD
#     ema12 = close.ewm(span=12, adjust=False).mean()
#     ema26 = close.ewm(span=26, adjust=False).mean()
#     df["MACD"] = ema12 - ema26
#     df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
#     df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

#     # Bollinger Bands
#     ma20 = close.rolling(20).mean()
#     std20 = close.rolling(20).std()
#     df["BB_Upper"] = ma20 + 2 * std20
#     df["BB_Lower"] = ma20 - 2 * std20

#     # Support / Resistance
#     df["Support"] = low.rolling(20).min()
#     df["Resistance"] = high.rolling(20).max()

#     # EMA
#     df["EMA12"] = ema12
#     df["EMA26"] = ema26

#     # VWAP
#     typical = (high + low + close) / 3
#     df["VWAP"] = (typical * vol).cumsum() / vol.cumsum()

#     # Stochastic
#     low14 = low.rolling(14).min()
#     high14 = high.rolling(14).max()
#     df["Stoch_K"] = 100 * (close - low14) / (high14 - low14)
#     df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

#     # OBV
#     df["OBV"] = (np.sign(close.diff()) * vol).fillna(0).cumsum()

#     # Aroon
#     df["Aroon_Up"] = 100 * (25 - high.rolling(25).apply(lambda x: len(x) - x.argmax())) / 25
#     df["Aroon_Down"] = 100 * (25 - low.rolling(25).apply(lambda x: len(x) - x.argmin())) / 25

#     df = df.dropna().reset_index()
#     return df


# def fetch_and_enhance(ticker: str, start, end) -> pd.DataFrame:
#     """Fetch data and add indicators"""
#     df = fetch_data(ticker, start, end)
#     return add_indicators(df)


# # ============================================
# # AGENT 1: TECHNICAL AGENT
# # ============================================

# class TechnicalAgent:
#     """Analyzes technical indicators"""
#     def __init__(self):
#         self.name = "Technical Agent"
    
#     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
#         """Analyze technical indicators - returns BUY/SELL/HOLD"""
#         latest = df.iloc[-1]
#         score = 0
#         signals = []
        
#         try:
#             # RSI Analysis
#             rsi = latest.get('RSI', 50)
#             if rsi < 30:
#                 score += 2
#                 signals.append("RSI oversold (bullish)")
#             elif rsi > 70:
#                 score -= 2
#                 signals.append("RSI overbought (bearish)")
            
#             # MACD Analysis
#             macd = latest.get('MACD', 0)
#             macd_signal = latest.get('MACD_Signal', 0)
#             macd_hist = latest.get('MACD_Hist', 0)
            
#             if macd > macd_signal and macd_hist > 0:
#                 score += 2
#                 signals.append("MACD bullish crossover")
#             elif macd < macd_signal and macd_hist < 0:
#                 score -= 2
#                 signals.append("MACD bearish crossover")
            
#             # Bollinger Bands
#             close = latest.get('Close', 0)
#             bb_lower = latest.get('BB_Lower', 0)
#             bb_upper = latest.get('BB_Upper', 0)
            
#             if close < bb_lower:
#                 score += 1
#                 signals.append("Price below lower BB")
#             elif close > bb_upper:
#                 score -= 1
#                 signals.append("Price above upper BB")
            
#             # EMA Trend
#             ema12 = latest.get('EMA12', 0)
#             ema26 = latest.get('EMA26', 0)
            
#             if ema12 > ema26:
#                 score += 1
#                 signals.append("Short-term trend bullish")
#             else:
#                 score -= 1
#                 signals.append("Short-term trend bearish")
            
#             # Stochastic
#             stoch_k = latest.get('Stoch_K', 50)
#             if stoch_k < 20:
#                 score += 1
#                 signals.append("Stochastic oversold")
#             elif stoch_k > 80:
#                 score -= 1
#                 signals.append("Stochastic overbought")
            
#             # ✅ FIXED: Return ONLY "BUY", "SELL", "HOLD"
#             if score >= 2:
#                 action = "BUY"
#             elif score <= -2:
#                 action = "SELL"
#             else:
#                 action = "HOLD"
            
#             confidence = min(abs(score) / 8 * 100, 100)
            
#             return {
#                 "agent": self.name,
#                 "action": action,
#                 "score": score,
#                 "confidence": round(confidence, 1),
#                 "signals": signals,
#                 "key_metrics": {
#                     "RSI": round(rsi, 2),
#                     "MACD": round(macd, 2),
#                     "Price_vs_VWAP": round((close - latest.get('VWAP', close)) / latest.get('VWAP', close) * 100, 2) if latest.get('VWAP') else 0
#                 }
#             }
        
#         except Exception as e:
#             logger.error(f"Error in technical analysis: {e}")
#             return {
#                 "agent": self.name,
#                 "action": "HOLD",
#                 "score": 0,
#                 "confidence": 0,
#                 "signals": [f"Error: {str(e)}"],
#                 "key_metrics": {}
#             }


# # ============================================
# # AGENT 2: SENTIMENT AGENT
# # ============================================

# class SentimentAgent:
#     """Analyzes market sentiment"""
#     def __init__(self):
#         self.name = "Sentiment Agent"
    
#     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
#         """Analyze sentiment - returns BUY/SELL/HOLD"""
#         try:
#             avg_volume = df['Volume'].tail(20).mean()
#             recent_volume = df['Volume'].tail(5).mean()
#             volume_change = (recent_volume - avg_volume) / avg_volume * 100 if avg_volume > 0 else 0
            
#             price_change_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100 if len(df) >= 6 else 0
#             price_change_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100 if len(df) >= 21 else 0
            
#             score = 0
#             signals = []
            
#             if volume_change > 20 and price_change_5d > 0:
#                 score += 2
#                 signals.append("Rising volume with price")
#             elif volume_change > 20 and price_change_5d < 0:
#                 score -= 2
#                 signals.append("Rising volume with falling price")
            
#             if price_change_20d > 10:
#                 score += 1
#                 signals.append("Strong uptrend")
#             elif price_change_20d < -10:
#                 score -= 1
#                 signals.append("Strong downtrend")
            
#             obv_change = 0
#             if len(df) >= 21 and df['OBV'].iloc[-21] != 0:
#                 obv_change = (df['OBV'].iloc[-1] - df['OBV'].iloc[-21]) / abs(df['OBV'].iloc[-21]) * 100
            
#             if obv_change > 10:
#                 score += 1
#                 signals.append("Positive OBV trend")
#             elif obv_change < -10:
#                 score -= 1
#                 signals.append("Negative OBV trend")
            
#             # ✅ FIXED: Return ONLY "BUY", "SELL", "HOLD"
#             if score >= 1:
#                 action = "BUY"
#             elif score <= -1:
#                 action = "SELL"
#             else:
#                 action = "HOLD"
            
#             confidence = min(abs(score) / 4 * 100, 100)
            
#             return {
#                 "agent": self.name,
#                 "action": action,
#                 "score": score,
#                 "confidence": round(confidence, 1),
#                 "signals": signals,
#                 "key_metrics": {
#                     "Volume_Change": round(volume_change, 2),
#                     "Price_Change_5D": round(price_change_5d, 2),
#                     "Price_Change_20D": round(price_change_20d, 2)
#                 }
#             }
        
#         except Exception as e:
#             logger.error(f"Error in sentiment analysis: {e}")
#             return {
#                 "agent": self.name,
#                 "action": "HOLD",
#                 "score": 0,
#                 "confidence": 0,
#                 "signals": [f"Error: {str(e)}"],
#                 "key_metrics": {}
#             }


# # ============================================
# # AGENT 3: RISK AGENT
# # ============================================

# class RiskAgent:
#     """Assesses risk and volatility"""
#     def __init__(self):
#         self.name = "Risk Agent"
    
#     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
#         """Analyze risk"""
#         try:
#             returns = df['Close'].pct_change().dropna()
#             volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
            
#             cumulative = (1 + returns).cumprod()
#             running_max = cumulative.expanding().max()
#             drawdown = (cumulative - running_max) / running_max
#             max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0
            
#             latest = df.iloc[-1]
#             support = latest.get('Support', latest['Close'])
#             resistance = latest.get('Resistance', latest['Close'])
            
#             distance_to_support = (latest['Close'] - support) / latest['Close'] * 100 if support > 0 else 100
#             distance_to_resistance = (resistance - latest['Close']) / latest['Close'] * 100 if resistance > 0 else 100
            
#             risk_score = 0
#             signals = []
            
#             if volatility < 20:
#                 risk_score = 3
#                 signals.append("Low volatility")
#             elif volatility < 40:
#                 risk_score = 2
#                 signals.append("Moderate volatility")
#             else:
#                 risk_score = 1
#                 signals.append("High volatility")
            
#             if distance_to_support < 2:
#                 signals.append("Near support")
#                 risk_score += 1
#             if distance_to_resistance < 2:
#                 signals.append("Near resistance")
#                 risk_score -= 1
            
#             if volatility < 20:
#                 position_size = "Normal (100%)"
#             elif volatility < 40:
#                 position_size = "Reduced (50%)"
#             else:
#                 position_size = "Minimal (25%)"
            
#             return {
#                 "agent": self.name,
#                 "risk_level": "LOW" if risk_score >= 3 else "MEDIUM" if risk_score >= 2 else "HIGH",
#                 "risk_score": risk_score,
#                 "position_size": position_size,
#                 "signals": signals,
#                 "key_metrics": {
#                     "Volatility": round(volatility, 2),
#                     "Max_Drawdown": round(max_drawdown, 2),
#                     "Distance_to_Support": round(distance_to_support, 2),
#                     "Distance_to_Resistance": round(distance_to_resistance, 2)
#                 }
#             }
        
#         except Exception as e:
#             logger.error(f"Error in risk analysis: {e}")
#             return {
#                 "agent": self.name,
#                 "risk_level": "MEDIUM",
#                 "risk_score": 2,
#                 "position_size": "Normal (100%)",
#                 "signals": [f"Error: {str(e)}"],
#                 "key_metrics": {}
#             }


# # ============================================
# # AGENT 4: PORTFOLIO AGENT
# # ============================================

# class PortfolioAgent:
#     """Manages portfolio allocation"""
#     def __init__(self, portfolio_value: float = 1000000):
#         self.name = "Portfolio Agent"
#         self.portfolio_value = portfolio_value
    
#     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
#         """Calculate portfolio allocation"""
#         try:
#             latest = df.iloc[-1]
#             current_price = latest['Close']
            
#             risk_level = context.get('risk', {}).get('risk_level', 'MEDIUM') if context else 'MEDIUM'
            
#             if risk_level == 'LOW':
#                 allocation_pct = 10
#             elif risk_level == 'MEDIUM':
#                 allocation_pct = 5
#             else:
#                 allocation_pct = 2
            
#             allocation_amount = self.portfolio_value * (allocation_pct / 100)
#             suggested_quantity = int(allocation_amount / current_price) if current_price > 0 else 0
            
#             signals = [
#                 f"Allocation: {allocation_pct}% of portfolio",
#                 f"Max position: ₹{allocation_amount:,.0f}",
#                 f"Quantity: {suggested_quantity} shares"
#             ]
            
#             return {
#                 "agent": self.name,
#                 "allocation_pct": allocation_pct,
#                 "allocation_amount": round(allocation_amount, 2),
#                 "suggested_quantity": suggested_quantity,
#                 "signals": signals,
#                 "key_metrics": {
#                     "Portfolio_Value": self.portfolio_value,
#                     "Current_Price": round(current_price, 2)
#                 }
#             }
        
#         except Exception as e:
#             logger.error(f"Error in portfolio analysis: {e}")
#             return {
#                 "agent": self.name,
#                 "allocation_pct": 5,
#                 "allocation_amount": 0,
#                 "suggested_quantity": 0,
#                 "signals": [f"Error: {str(e)}"],
#                 "key_metrics": {}
#             }


# # ============================================
# # AGENT 5: MASTER AGENT
# # ============================================

# class MasterAgent:
#     """Makes final trading decision"""
#     def __init__(self):
#         self.name = "Master Agent"
    
#     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
#         """Master decision - synthesizes all agents"""
#         try:
#             if not context:
#                 return {
#                     "agent": self.name,
#                     "action": "HOLD",
#                     "confidence": 0,
#                     "reasoning": "No data",
#                     "quantity": 0,
#                     "risk_level": "MEDIUM",
#                     "votes": {"BUY": 0, "SELL": 0, "HOLD": 5}
#                 }
            
#             technical = context.get('technical', {})
#             sentiment = context.get('sentiment', {})
#             risk = context.get('risk', {})
#             institutional = context.get('institutional', {})
#             portfolio = context.get('portfolio', {})
            
#             votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
            
#             # Get actions
#             tech_action = technical.get('action', 'HOLD')
#             sent_action = sentiment.get('action', 'HOLD')
            
#             # Vote
#             if tech_action in votes:
#                 votes[tech_action] += 3
#             else:
#                 votes['HOLD'] += 3
            
#             if sent_action in votes:
#                 votes[sent_action] += 2
#             else:
#                 votes['HOLD'] += 2
            
#             # Institutional signal
#             inst_rec = institutional.get('recommendation', '')
#             if "BUY" in inst_rec:
#                 votes['BUY'] += 2.5
#             elif "SELL" in inst_rec:
#                 votes['SELL'] += 2.5
#             else:
#                 votes['HOLD'] += 2.5
            
#             # Risk veto
#             risk_level = risk.get('risk_level', 'MEDIUM')
#             if risk_level == 'HIGH':
#                 votes['BUY'] = max(0, votes['BUY'] - 2)
#                 votes['HOLD'] += 3
            
#             # Final decision
#             final_action = max(votes, key=votes.get)
#             total_votes = sum(votes.values())
#             confidence = (votes[final_action] / total_votes * 100) if total_votes > 0 else 0
            
#             reasoning = f"Tech: {tech_action} | Sentiment: {sent_action} | Inst: {institutional.get('recommendation', 'NEUTRAL')} | Risk: {risk_level}"
            
#             return {
#                 "agent": self.name,
#                 "action": final_action,
#                 "confidence": round(confidence, 1),
#                 "reasoning": reasoning,
#                 "votes": votes,
#                 "risk_level": risk_level,
#                 "quantity": portfolio.get('suggested_quantity', 0)
#             }
        
#         except Exception as e:
#             logger.error(f"Error in master analysis: {e}")
#             return {
#                 "agent": self.name,
#                 "action": "HOLD",
#                 "confidence": 0,
#                 "reasoning": f"Error: {str(e)}",
#                 "quantity": 0,
#                 "risk_level": "MEDIUM",
#                 "votes": {"BUY": 0, "SELL": 0, "HOLD": 5}
#             }


# # ============================================
# # TRADING SYSTEM
# # ============================================

# class TradingSystem:
#     """Main trading system with all agents"""
    
#     def __init__(self, portfolio_value: float = 1000000):
#         self.agents = {
#             'technical': TechnicalAgent(),
#             'sentiment': SentimentAgent(),
#             'risk': RiskAgent(),
#             'portfolio': PortfolioAgent(portfolio_value),
#             'master': MasterAgent()
#         }
        
#         # Institutional aggregator
#         self.institutional_aggregator = MasterInstitutionalAggregator()

#         # 🆕 New News: initialize fetcher + agent if available
#         self.new_news_fetcher = NewNewsFetcher() if NewNewsFetcher else None
#         self.new_news_agent = NewNews(groq_api_key=os.getenv("GROQ_API_KEY")) if NewNews else None
#         if not self.new_news_fetcher:
#             logger.info("NewNewsFetcher not available. Skipping New News integration.")
#         if not self.new_news_agent:
#             logger.info("NewNews agent not available or GROQ not installed. Chat/summary disabled.")

#         self.analysis_log = []
    
#     def run_analysis(self, ticker: str, start, end, 
#                     fii_data=None, order_data=None, 
#                     block_deals=None, holdings_data=None) -> Dict:
#         """Run complete analysis"""
        
#         logger.info(f"📊 Analyzing {ticker}...")
        
#         try:
#             # Fetch and enhance data
#             df = fetch_and_enhance(ticker, start, end)
            
#             if df is None or len(df) == 0:
#                 logger.error("Failed to fetch data")
#                 return {"error": "No data fetched", "ticker": ticker}
            
#             # Run all agents
#             context = {}
#             context['technical'] = self.agents['technical'].analyze(df)
#             context['sentiment'] = self.agents['sentiment'].analyze(df)
#             context['risk'] = self.agents['risk'].analyze(df, context)
#             context['portfolio'] = self.agents['portfolio'].analyze(df, context)
            
#             # ✅ Institutional analysis
#             try:
#                 institutional_signal = self.institutional_aggregator.aggregate(
#                     price_data=df['Close'],
#                     volume_data=df['Volume'],
#                     fii_data=fii_data or {},
#                     order_data=order_data or []
#                 )
#                 context['institutional'] = {
#                     'final_score': institutional_signal.final_score,
#                     'recommendation': institutional_signal.recommendation,
#                     'confidence': institutional_signal.confidence
#                 }
#             except Exception as e:
#                 logger.warning(f"Institutional analysis skipped: {e}")
#                 context['institutional'] = {
#                     'final_score': 50,
#                     'recommendation': 'HOLD',
#                     'confidence': 0
#                 }

#             # 🆕 New News: fetch + analyze (no LLM by default)
#             new_news_section = {"articles": [], "analysis": {}, "groq_ready": False}
#             try:
#                 if self.new_news_fetcher:
#                     articles = self.new_news_fetcher.get_news(
#                         company_name=ticker,  # replace with real company name if available
#                         ticker=ticker,
#                         max_results=16,
#                         days=7,
#                         dedupe=True
#                     )
#                     new_news_section["articles"] = articles

#                     if self.new_news_agent and articles:
#                         self.new_news_agent.set_context(ticker=ticker, company_name=ticker, news_data=articles)
#                         analysis = self.new_news_agent.analyze(ticker=ticker, news_data=articles)
#                         new_news_section["analysis"] = analysis
#                         new_news_section["groq_ready"] = bool(getattr(self.new_news_agent, "client", None))
#             except Exception as e:
#                 logger.warning(f"New News fetch/analyze skipped: {e}")

#             # Master decision
#             context['master'] = self.agents['master'].analyze(df, context)
            
#             result = {
#                 'ticker': ticker,
#                 'timestamp': datetime.now(),
#                 'data': df,
#                 'recommendations': context,
#                 'new_news': new_news_section,  # 🆕 include in result
#                 'status': 'SUCCESS'
#             }
            
#             self.analysis_log.append(result)
#             logger.info(f"✅ Analysis complete: {context['master']['action']}")
#             return result
        
#         except Exception as e:
#             logger.error(f"Analysis error: {e}")
#             return {
#                 "error": str(e),
#                 "ticker": ticker,
#                 "status": "FAILED"
#             }


# # ============================================
# # GLOBAL INSTANCE & HELPERS
# # ============================================

# _trading_system = None


# def get_trading_system(portfolio_value: float = 1000000):
#     """Get singleton instance"""
#     global _trading_system
#     if _trading_system is None:
#         _trading_system = TradingSystem(portfolio_value)
#     return _trading_system


# def analyze_stock(ticker: str, start, end, portfolio_value: float = 1000000,
#                  fii_data=None, order_data=None, block_deals=None, holdings_data=None):
#     """Main analysis function"""
#     system = get_trading_system(portfolio_value)
#     return system.run_analysis(ticker, start, end, fii_data, order_data, block_deals, holdings_data)


# def get_detailed_analysis(analysis_result):
#     """Get detailed report"""
#     return analysis_result


# main.py - UPDATED WITH New News INTEGRATION

# """
# ProTrader AI - Main Trading System
# Orchestrates all agents for trading analysis
# """

# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from typing import Dict, List, Any
# import logging

# # Import data fetcher
# from data.data_fetcher import fetch_data

# # Import institutional aggregator
# from agents.institutional_agents import MasterInstitutionalAggregator

# # Optional: New News (fetcher + agent)
# try:
#     from data.new_news_fetcher import NewNewsFetcher
#     from agents.new_news_agent import NewNews
# except Exception:
#     NewNewsFetcher = None
#     NewNews = None

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # ============================================
# # TECHNICAL INDICATORS - Calculate Here
# # ============================================

# def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
#     """Add technical indicators to DataFrame"""
#     df = df.copy()
#     if 'Date' in df.columns:
#         df['Date'] = pd.to_datetime(df['Date'])
#         df = df.set_index('Date')
    
#     close = df["Close"]
#     high = df["High"]
#     low = df["Low"]
#     vol = df["Volume"]

#     # RSI
#     delta = close.diff()
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)
#     df["RSI"] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.finfo(float).eps))))

#     # MACD
#     ema12 = close.ewm(span=12, adjust=False).mean()
#     ema26 = close.ewm(span=26, adjust=False).mean()
#     df["MACD"] = ema12 - ema26
#     df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
#     df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

#     # Bollinger Bands
#     ma20 = close.rolling(20).mean()
#     std20 = close.rolling(20).std()
#     df["BB_Upper"] = ma20 + 2 * std20
#     df["BB_Lower"] = ma20 - 2 * std20

#     # Support / Resistance
#     df["Support"] = low.rolling(20).min()
#     df["Resistance"] = high.rolling(20).max()

#     # EMA
#     df["EMA12"] = ema12
#     df["EMA26"] = ema26

#     # VWAP
#     typical = (high + low + close) / 3
#     df["VWAP"] = (typical * vol).cumsum() / vol.cumsum()

#     # Stochastic
#     low14 = low.rolling(14).min()
#     high14 = high.rolling(14).max()
#     df["Stoch_K"] = 100 * (close - low14) / (high14 - low14)
#     df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

#     # OBV
#     df["OBV"] = (np.sign(close.diff()) * vol).fillna(0).cumsum()

#     # Aroon
#     df["Aroon_Up"] = 100 * (25 - high.rolling(25).apply(lambda x: len(x) - x.argmax())) / 25
#     df["Aroon_Down"] = 100 * (25 - low.rolling(25).apply(lambda x: len(x) - x.argmin())) / 25

#     df = df.dropna().reset_index()
#     return df


# def fetch_and_enhance(ticker: str, start, end) -> pd.DataFrame:
#     """Fetch data and add indicators"""
#     df = fetch_data(ticker, start, end)
#     return add_indicators(df)


# # ============================================
# # AGENT 1: TECHNICAL AGENT
# # ============================================

# class TechnicalAgent:
#     """Analyzes technical indicators"""
#     def __init__(self):
#         self.name = "Technical Agent"
    
#     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
#         """Analyze technical indicators - returns BUY/SELL/HOLD"""
#         latest = df.iloc[-1]
#         score = 0
#         signals = []
        
#         try:
#             # RSI Analysis
#             rsi = latest.get('RSI', 50)
#             if rsi < 30:
#                 score += 2
#                 signals.append("RSI oversold (bullish)")
#             elif rsi > 70:
#                 score -= 2
#                 signals.append("RSI overbought (bearish)")
            
#             # MACD Analysis
#             macd = latest.get('MACD', 0)
#             macd_signal = latest.get('MACD_Signal', 0)
#             macd_hist = latest.get('MACD_Hist', 0)
            
#             if macd > macd_signal and macd_hist > 0:
#                 score += 2
#                 signals.append("MACD bullish crossover")
#             elif macd < macd_signal and macd_hist < 0:
#                 score -= 2
#                 signals.append("MACD bearish crossover")
            
#             # Bollinger Bands
#             close = latest.get('Close', 0)
#             bb_lower = latest.get('BB_Lower', 0)
#             bb_upper = latest.get('BB_Upper', 0)
            
#             if close < bb_lower:
#                 score += 1
#                 signals.append("Price below lower BB")
#             elif close > bb_upper:
#                 score -= 1
#                 signals.append("Price above upper BB")
            
#             # EMA Trend
#             ema12 = latest.get('EMA12', 0)
#             ema26 = latest.get('EMA26', 0)
            
#             if ema12 > ema26:
#                 score += 1
#                 signals.append("Short-term trend bullish")
#             else:
#                 score -= 1
#                 signals.append("Short-term trend bearish")
            
#             # Stochastic
#             stoch_k = latest.get('Stoch_K', 50)
#             if stoch_k < 20:
#                 score += 1
#                 signals.append("Stochastic oversold")
#             elif stoch_k > 80:
#                 score -= 1
#                 signals.append("Stochastic overbought")
            
#             # ✅ FIXED: Return ONLY "BUY", "SELL", "HOLD"
#             if score >= 2:
#                 action = "BUY"
#             elif score <= -2:
#                 action = "SELL"
#             else:
#                 action = "HOLD"
            
#             confidence = min(abs(score) / 8 * 100, 100)
            
#             return {
#                 "agent": self.name,
#                 "action": action,
#                 "score": score,
#                 "confidence": round(confidence, 1),
#                 "signals": signals,
#                 "key_metrics": {
#                     "RSI": round(rsi, 2),
#                     "MACD": round(macd, 2),
#                     "Price_vs_VWAP": round((close - latest.get('VWAP', close)) / latest.get('VWAP', close) * 100, 2) if latest.get('VWAP') else 0
#                 }
#             }
        
#         except Exception as e:
#             logger.error(f"Error in technical analysis: {e}")
#             return {
#                 "agent": self.name,
#                 "action": "HOLD",
#                 "score": 0,
#                 "confidence": 0,
#                 "signals": [f"Error: {str(e)}"],
#                 "key_metrics": {}
#             }


# # ============================================
# # AGENT 2: SENTIMENT AGENT
# # ============================================

# class SentimentAgent:
#     """Analyzes market sentiment"""
#     def __init__(self):
#         self.name = "Sentiment Agent"
    
#     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
#         """Analyze sentiment - returns BUY/SELL/HOLD"""
#         try:
#             avg_volume = df['Volume'].tail(20).mean()
#             recent_volume = df['Volume'].tail(5).mean()
#             volume_change = (recent_volume - avg_volume) / avg_volume * 100 if avg_volume > 0 else 0
            
#             price_change_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100 if len(df) >= 6 else 0
#             price_change_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100 if len(df) >= 21 else 0
            
#             score = 0
#             signals = []
            
#             if volume_change > 20 and price_change_5d > 0:
#                 score += 2
#                 signals.append("Rising volume with price")
#             elif volume_change > 20 and price_change_5d < 0:
#                 score -= 2
#                 signals.append("Rising volume with falling price")
            
#             if price_change_20d > 10:
#                 score += 1
#                 signals.append("Strong uptrend")
#             elif price_change_20d < -10:
#                 score -= 1
#                 signals.append("Strong downtrend")
            
#             obv_change = 0
#             if len(df) >= 21 and df['OBV'].iloc[-21] != 0:
#                 obv_change = (df['OBV'].iloc[-1] - df['OBV'].iloc[-21]) / abs(df['OBV'].iloc[-21]) * 100
            
#             if obv_change > 10:
#                 score += 1
#                 signals.append("Positive OBV trend")
#             elif obv_change < -10:
#                 score -= 1
#                 signals.append("Negative OBV trend")
            
#             # ✅ FIXED: Return ONLY "BUY", "SELL", "HOLD"
#             if score >= 1:
#                 action = "BUY"
#             elif score <= -1:
#                 action = "SELL"
#             else:
#                 action = "HOLD"
            
#             confidence = min(abs(score) / 4 * 100, 100)
            
#             return {
#                 "agent": self.name,
#                 "action": action,
#                 "score": score,
#                 "confidence": round(confidence, 1),
#                 "signals": signals,
#                 "key_metrics": {
#                     "Volume_Change": round(volume_change, 2),
#                     "Price_Change_5D": round(price_change_5d, 2),
#                     "Price_Change_20D": round(price_change_20d, 2)
#                 }
#             }
        
#         except Exception as e:
#             logger.error(f"Error in sentiment analysis: {e}")
#             return {
#                 "agent": self.name,
#                 "action": "HOLD",
#                 "score": 0,
#                 "confidence": 0,
#                 "signals": [f"Error: {str(e)}"],
#                 "key_metrics": {}
#             }


# # ============================================
# # AGENT 3: RISK AGENT
# # ============================================

# class RiskAgent:
#     """Assesses risk and volatility"""
#     def __init__(self):
#         self.name = "Risk Agent"
    
#     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
#         """Analyze risk"""
#         try:
#             returns = df['Close'].pct_change().dropna()
#             volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
            
#             cumulative = (1 + returns).cumprod()
#             running_max = cumulative.expanding().max()
#             drawdown = (cumulative - running_max) / running_max
#             max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0
            
#             latest = df.iloc[-1]
#             support = latest.get('Support', latest['Close'])
#             resistance = latest.get('Resistance', latest['Close'])
            
#             distance_to_support = (latest['Close'] - support) / latest['Close'] * 100 if support > 0 else 100
#             distance_to_resistance = (resistance - latest['Close']) / latest['Close'] * 100 if resistance > 0 else 100
            
#             risk_score = 0
#             signals = []
            
#             if volatility < 20:
#                 risk_score = 3
#                 signals.append("Low volatility")
#             elif volatility < 40:
#                 risk_score = 2
#                 signals.append("Moderate volatility")
#             else:
#                 risk_score = 1
#                 signals.append("High volatility")
            
#             if distance_to_support < 2:
#                 signals.append("Near support")
#                 risk_score += 1
#             if distance_to_resistance < 2:
#                 signals.append("Near resistance")
#                 risk_score -= 1
            
#             if volatility < 20:
#                 position_size = "Normal (100%)"
#             elif volatility < 40:
#                 position_size = "Reduced (50%)"
#             else:
#                 position_size = "Minimal (25%)"
            
#             return {
#                 "agent": self.name,
#                 "risk_level": "LOW" if risk_score >= 3 else "MEDIUM" if risk_score >= 2 else "HIGH",
#                 "risk_score": risk_score,
#                 "position_size": position_size,
#                 "signals": signals,
#                 "key_metrics": {
#                     "Volatility": round(volatility, 2),
#                     "Max_Drawdown": round(max_drawdown, 2),
#                     "Distance_to_Support": round(distance_to_support, 2),
#                     "Distance_to_Resistance": round(distance_to_resistance, 2)
#                 }
#             }
        
#         except Exception as e:
#             logger.error(f"Error in risk analysis: {e}")
#             return {
#                 "agent": self.name,
#                 "risk_level": "MEDIUM",
#                 "risk_score": 2,
#                 "position_size": "Normal (100%)",
#                 "signals": [f"Error: {str(e)}"],
#                 "key_metrics": {}
#             }


# # ============================================
# # AGENT 4: PORTFOLIO AGENT
# # ============================================

# class PortfolioAgent:
#     """Manages portfolio allocation"""
#     def __init__(self, portfolio_value: float = 1000000):
#         self.name = "Portfolio Agent"
#         self.portfolio_value = portfolio_value
    
#     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
#         """Calculate portfolio allocation"""
#         try:
#             latest = df.iloc[-1]
#             current_price = latest['Close']
            
#             risk_level = context.get('risk', {}).get('risk_level', 'MEDIUM') if context else 'MEDIUM'
            
#             if risk_level == 'LOW':
#                 allocation_pct = 10
#             elif risk_level == 'MEDIUM':
#                 allocation_pct = 5
#             else:
#                 allocation_pct = 2
            
#             allocation_amount = self.portfolio_value * (allocation_pct / 100)
#             suggested_quantity = int(allocation_amount / current_price) if current_price > 0 else 0
            
#             signals = [
#                 f"Allocation: {allocation_pct}% of portfolio",
#                 f"Max position: ₹{allocation_amount:,.0f}",
#                 f"Quantity: {suggested_quantity} shares"
#             ]
            
#             return {
#                 "agent": self.name,
#                 "allocation_pct": allocation_pct,
#                 "allocation_amount": round(allocation_amount, 2),
#                 "suggested_quantity": suggested_quantity,
#                 "signals": signals,
#                 "key_metrics": {
#                     "Portfolio_Value": self.portfolio_value,
#                     "Current_Price": round(current_price, 2)
#                 }
#             }
        
#         except Exception as e:
#             logger.error(f"Error in portfolio analysis: {e}")
#             return {
#                 "agent": self.name,
#                 "allocation_pct": 5,
#                 "allocation_amount": 0,
#                 "suggested_quantity": 0,
#                 "signals": [f"Error: {str(e)}"],
#                 "key_metrics": {}
#             }


# # ============================================
# # AGENT 5: MASTER AGENT
# # ============================================

# class MasterAgent:
#     """Makes final trading decision"""
#     def __init__(self):
#         self.name = "Master Agent"
    
#     def analyze(self, df: pd.DataFrame, context: Dict = None) -> Dict[str, Any]:
#         """Master decision - synthesizes all agents"""
#         try:
#             if not context:
#                 return {
#                     "agent": self.name,
#                     "action": "HOLD",
#                     "confidence": 0,
#                     "reasoning": "No data",
#                     "quantity": 0,
#                     "risk_level": "MEDIUM",
#                     "votes": {"BUY": 0, "SELL": 0, "HOLD": 5}
#                 }
            
#             technical = context.get('technical', {})
#             sentiment = context.get('sentiment', {})
#             risk = context.get('risk', {})
#             institutional = context.get('institutional', {})
#             portfolio = context.get('portfolio', {})
            
#             votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
            
#             # Get actions
#             tech_action = technical.get('action', 'HOLD')
#             sent_action = sentiment.get('action', 'HOLD')
            
#             # Vote
#             if tech_action in votes:
#                 votes[tech_action] += 3
#             else:
#                 votes['HOLD'] += 3
            
#             if sent_action in votes:
#                 votes[sent_action] += 2
#             else:
#                 votes['HOLD'] += 2
            
#             # Institutional signal
#             inst_rec = institutional.get('recommendation', '')
#             if "BUY" in inst_rec:
#                 votes['BUY'] += 2.5
#             elif "SELL" in inst_rec:
#                 votes['SELL'] += 2.5
#             else:
#                 votes['HOLD'] += 2.5
            
#             # Risk veto
#             risk_level = risk.get('risk_level', 'MEDIUM')
#             if risk_level == 'HIGH':
#                 votes['BUY'] = max(0, votes['BUY'] - 2)
#                 votes['HOLD'] += 3
            
#             # Final decision
#             final_action = max(votes, key=votes.get)
#             total_votes = sum(votes.values())
#             confidence = (votes[final_action] / total_votes * 100) if total_votes > 0 else 0
            
#             reasoning = f"Tech: {tech_action} | Sentiment: {sent_action} | Inst: {institutional.get('recommendation', 'NEUTRAL')} | Risk: {risk_level}"
            
#             return {
#                 "agent": self.name,
#                 "action": final_action,
#                 "confidence": round(confidence, 1),
#                 "reasoning": reasoning,
#                 "votes": votes,
#                 "risk_level": risk_level,
#                 "quantity": portfolio.get('suggested_quantity', 0)
#             }
        
#         except Exception as e:
#             logger.error(f"Error in master analysis: {e}")
#             return {
#                 "agent": self.name,
#                 "action": "HOLD",
#                 "confidence": 0,
#                 "reasoning": f"Error: {str(e)}",
#                 "quantity": 0,
#                 "risk_level": "MEDIUM",
#                 "votes": {"BUY": 0, "SELL": 0, "HOLD": 5}
#             }


# # ============================================
# # TRADING SYSTEM
# # ============================================

# class TradingSystem:
#     """Main trading system with all agents"""
    
#     def __init__(self, portfolio_value: float = 1000000):
#         self.agents = {
#             'technical': TechnicalAgent(),
#             'sentiment': SentimentAgent(),
#             'risk': RiskAgent(),
#             'portfolio': PortfolioAgent(portfolio_value),
#             'master': MasterAgent()
#         }
        
#         # Institutional aggregator
#         self.institutional_aggregator = MasterInstitutionalAggregator()

#         # New News integration (optional)
#         self.new_news_fetcher = NewNewsFetcher() if NewNewsFetcher else None
#         self.new_news_agent = NewNews(groq_api_key=os.getenv("GROQ_API_KEY")) if NewNews else None
#         if not self.new_news_fetcher:
#             logger.info("NewNewsFetcher not available. Skipping New News integration.")
#         if not self.new_news_agent:
#             logger.info("NewNews agent not available or GROQ not installed. Chat/summary disabled.")
        
#         self.analysis_log = []
    
#     def run_analysis(self, ticker: str, start, end, 
#                     fii_data=None, order_data=None, 
#                     block_deals=None, holdings_data=None) -> Dict:
#         """Run complete analysis"""
        
#         logger.info(f"📊 Analyzing {ticker}...")
        
#         try:
#             # Fetch and enhance data
#             df = fetch_and_enhance(ticker, start, end)
            
#             if df is None or len(df) == 0:
#                 logger.error("Failed to fetch data")
#                 return {"error": "No data fetched", "ticker": ticker}
            
#             # Run all agents
#             context = {}
#             context['technical'] = self.agents['technical'].analyze(df)
#             context['sentiment'] = self.agents['sentiment'].analyze(df)
#             context['risk'] = self.agents['risk'].analyze(df, context)
#             context['portfolio'] = self.agents['portfolio'].analyze(df, context)
            
#             # Institutional analysis
#             try:
#                 institutional_signal = self.institutional_aggregator.aggregate(
#                     price_data=df['Close'],
#                     volume_data=df['Volume'],
#                     fii_data=fii_data or {},
#                     order_data=order_data or []
#                 )
                
#                 context['institutional'] = {
#                     'final_score': institutional_signal.final_score,
#                     'recommendation': institutional_signal.recommendation,
#                     'confidence': institutional_signal.confidence
#                 }
#             except Exception as e:
#                 logger.warning(f"Institutional analysis skipped: {e}")
#                 context['institutional'] = {
#                     'final_score': 50,
#                     'recommendation': 'HOLD',
#                     'confidence': 0
#                 }
            
#             # New News: fetch + analyze
#             new_news_section = {"articles": [], "analysis": {}, "groq_ready": False}
#             try:
#                 if self.new_news_fetcher:
#                     articles = self.new_news_fetcher.get_news(
#                         company_name=ticker,  # replace with actual company name if available
#                         ticker=ticker,
#                         max_results=16,
#                         days=7,
#                         dedupe=True
#                     )
#                     new_news_section["articles"] = articles

#                     if self.new_news_agent and articles:
#                         self.new_news_agent.set_context(ticker=ticker, company_name=ticker, news_data=articles)
#                         analysis = self.new_news_agent.analyze(ticker=ticker, news_data=articles)
#                         new_news_section["analysis"] = analysis
#                         new_news_section["groq_ready"] = bool(getattr(self.new_news_agent, "client", None))
#             except Exception as e:
#                 logger.warning(f"New News fetch/analyze skipped: {e}")
            
#             # Master decision
#             context['master'] = self.agents['master'].analyze(df, context)
            
#             result = {
#                 'ticker': ticker,
#                 'timestamp': datetime.now(),
#                 'data': df,
#                 'recommendations': context,
#                 'new_news': new_news_section,  # include New News in result
#                 'status': 'SUCCESS'
#             }
            
#             self.analysis_log.append(result)
#             logger.info(f"✅ Analysis complete: {context['master']['action']}")
#             return result
        
#         except Exception as e:
#             logger.error(f"Analysis error: {e}")
#             return {
#                 "error": str(e),
#                 "ticker": ticker,
#                 "status": "FAILED"
#             }


# # ============================================
# # GLOBAL INSTANCE & HELPERS
# # ============================================

# _trading_system = None


# def get_trading_system(portfolio_value: float = 1000000):
#     """Get singleton instance"""
#     global _trading_system
#     if _trading_system is None:
#         _trading_system = TradingSystem(portfolio_value)
#     return _trading_system


# def analyze_stock(ticker: str, start, end, portfolio_value: float = 1000000,
#                  fii_data=None, order_data=None, block_deals=None, holdings_data=None):
#     """Main analysis function"""
#     system = get_trading_system(portfolio_value)
#     return system.run_analysis(ticker, start, end, fii_data, order_data, block_deals, holdings_data)


# def get_detailed_analysis(analysis_result):
#     """Get detailed report"""
#     return analysis_result



# main.py
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import numpy as np

from data.data_fetcher import fetch_data
from agents.institutional_agents import MasterInstitutionalAggregator

# Optional New News
try:
    from data.new_news_fetcher import NewNewsFetcher
    from trading_bot.agents.news_agent import NewNews
except Exception:
    NewNewsFetcher = None
    NewNews = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------
# indicators helper (kept short)
# ------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    df["RSI"] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.finfo(float).eps))))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_Upper"] = ma20 + 2 * std20
    df["BB_Lower"] = ma20 - 2 * std20

    # Support / Resistance
    df["Support"] = low.rolling(20).min()
    df["Resistance"] = high.rolling(20).max()

    # EMA
    df["EMA12"] = ema12
    df["EMA26"] = ema26

    # VWAP
    typical = (high + low + close) / 3
    df["VWAP"] = (typical * vol).cumsum() / vol.cumsum()

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["Stoch_K"] = 100 * (close - low14) / (high14 - low14)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # OBV
    df["OBV"] = (np.sign(close.diff()) * vol).fillna(0).cumsum()

    df = df.dropna().reset_index()
    return df

def fetch_and_enhance(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = fetch_data(ticker, start, end)
    if df is None or df.empty:
        return pd.DataFrame()
    return add_indicators(df)

# ------------------------------
# Agents wiring
# ------------------------------
class TradingSystem:
    def __init__(self, portfolio_value: float = 1_000_000):
        # instantiate agents (logic classes)
        from agents.technical_agent import TechnicalAnalysisAgent
        from agents.sentiment_agent import SentimentAgent
        from agents.risk_agent import RiskManagementAgent
        from agents.portfolio_agent import PortfolioManagerAgent
        from agents.master_agent import MasterAgent

        self.agents = {
            'technical': TechnicalAnalysisAgent(),
            'sentiment': SentimentAgent(),
            'risk': RiskManagementAgent(),
            'portfolio': PortfolioManagerAgent(portfolio_value),
            'master': MasterAgent()
        }

        self.institutional_aggregator = MasterInstitutionalAggregator()
        self.new_news_fetcher = NewNewsFetcher() if NewNewsFetcher else None
        self.new_news_agent = NewNews(groq_api_key=os.getenv("GROQ_API_KEY")) if NewNews else None
        self.analysis_log = []

    def run_analysis(self, ticker: str, start: datetime, end: datetime, **kwargs) -> Dict[str, Any]:
        logger.info(f"Running analysis for {ticker}")
        df = fetch_and_enhance(ticker, start, end)
        if df is None or df.empty:
            return {"status": "FAILED", "error": "No data"}

        ctx = {}
        ctx['technical'] = self.agents['technical'].analyze(ticker, start, end)
        ctx['sentiment'] = self.agents['sentiment'].analyze(df)
        latest_close = df.iloc[-1]['Close']
        ctx['risk'] = self.agents['risk'].evaluate(ticker, df, latest_close, ctx['technical'].get('confidence', 50))
        ctx['portfolio'] = self.agents['portfolio'].analyze(df, ctx)
        # institutional aggregator
        try:
            inst = self.institutional_aggregator.aggregate(
                price_data=df['Close'],
                volume_data=df['Volume'],
                fii_data=kwargs.get('fii_data', {}),
                order_data=kwargs.get('order_data', [])
            )
            ctx['institutional'] = {
                'final_score': getattr(inst, 'final_score', 50),
                'recommendation': getattr(inst, 'recommendation', 'HOLD'),
                'confidence': getattr(inst, 'confidence', 50)
            }
        except Exception:
            ctx['institutional'] = {'final_score': 50, 'recommendation': 'HOLD', 'confidence': 50}

        # New News (optional)
        new_news_section = {'articles': [], 'analysis': {}}
        try:
            if self.new_news_fetcher:
                articles = self.new_news_fetcher.get_news(company_name=ticker, ticker=ticker, max_results=16, days=7)
                new_news_section['articles'] = articles
                if self.new_news_agent and articles:
                    self.new_news_agent.set_context(ticker=ticker, company_name=ticker, news_data=articles)
                    new_news_section['analysis'] = self.new_news_agent.analyze(ticker=ticker, news_data=articles)
        except Exception:
            pass

        ctx['master'] = self.agents['master'].synthesize(
            ticker=ticker,
            technical_result=ctx['technical'],
            sentiment_result=ctx['sentiment'],
            risk_metrics=ctx['risk'],
            portfolio_metrics=ctx['portfolio'],
            current_price=latest_close
        )

        result = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'data': df,
            'recommendations': ctx,
            'new_news': new_news_section,
            'status': 'SUCCESS'
        }
        self.analysis_log.append(result)
        return result

# Singleton helpers
_trading_system = None

def get_trading_system(portfolio_value: float = 1_000_000):
    global _trading_system
    if _trading_system is None:
        _trading_system = TradingSystem(portfolio_value)
    return _trading_system

def analyze_stock(ticker: str, start: datetime, end: datetime, **kwargs):
    system = get_trading_system(kwargs.get('portfolio_value', 1_000_000))
    return system.run_analysis(ticker, start, end, **kwargs)

def get_detailed_analysis(analysis_result: Dict[str, Any]):
    return analysis_result
