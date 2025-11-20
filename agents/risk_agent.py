# from langchain_groq import ChatGroq
# from models.schemas import TechnicalAnalysisOutput
# import os
# from dotenv import load_dotenv

# load_dotenv()
# llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

# class RiskManagementAgent:
#     def evaluate(self, state: dict, portfolio: dict) -> dict:
#         prompt = f"Evaluate risk for {state['ticker']} (e.g., NTPC.NS). Indicators: {state['technical']}. Sentiment: {state['sentiment']}. Portfolio: {portfolio}. Suggest position size (0-10%) and stop-loss."
#         response = llm.invoke(prompt)
#         return {"position_size": 0.05, "stop_loss": 0.1}  # Placeholder




# import pandas as pd
# import numpy as np
# from typing import Dict


# class RiskManagementAgent:
#     """Autonomous risk management - NO LLM DEPENDENCY"""
    
#     def __init__(self, max_drawdown: float = 0.05, risk_free_rate: float = 0.06):
#         self.max_drawdown = max_drawdown  # 5% max drawdown
#         self.risk_free_rate = risk_free_rate
#         self.max_position_size = 0.10  # 10% max
    
#     def evaluate(self, 
#                  ticker: str,
#                  df: pd.DataFrame,
#                  current_price: float,
#                  technical_confidence: float,
#                  sentiment_confidence: float = 50) -> Dict:
#         """
#         Autonomous risk evaluation
        
#         Args:
#             ticker: Stock ticker
#             df: Price dataframe with OHLCV
#             current_price: Current price
#             technical_confidence: Confidence from technical agent
#             sentiment_confidence: Confidence from sentiment agent
        
#         Returns:
#             Risk metrics dictionary
#         """
        
#         # Calculate volatility
#         volatility = self._calculate_volatility(df)
        
#         # Calculate risk level
#         risk_level = self._determine_risk_level(volatility)
        
#         # Calculate position size (adjusted for volatility and confidence)
#         position_size = self._calculate_position_size(
#             volatility,
#             technical_confidence,
#             sentiment_confidence
#         )
        
#         # Calculate stop-loss (ATR-based)
#         stop_loss_pct = self._calculate_stop_loss(df, volatility)
        
#         # Calculate take-profit (risk-reward ratio)
#         take_profit_pct = stop_loss_pct * 2  # 1:2 risk-reward
        
#         # Calculate support and resistance
#         support = df["Low"].rolling(20).min().iloc[-1]
#         resistance = df["High"].rolling(20).max().iloc[-1]
        
#         return {
#             "ticker": ticker,
#             "volatility": volatility,
#             "risk_level": risk_level,
#             "position_size": position_size,
#             "stop_loss_pct": stop_loss_pct,
#             "stop_loss_price": current_price * (1 - stop_loss_pct),
#             "take_profit_pct": take_profit_pct,
#             "take_profit_price": current_price * (1 + take_profit_pct),
#             "support": support,
#             "resistance": resistance,
#             "sharpe_ratio": self._calculate_sharpe(df),
#             "max_risk": self.max_drawdown
#         }
    
#     def _calculate_volatility(self, df: pd.DataFrame) -> float:
#         """
#         Calculate volatility using daily returns
        
#         Formula: std_dev of returns * sqrt(252) for annualized
#         """
#         close = df["Close"]
#         returns = close.pct_change().dropna()
        
#         # Daily volatility
#         daily_volatility = returns.std()
        
#         # Annualized volatility
#         annual_volatility = daily_volatility * np.sqrt(252)
        
#         return annual_volatility
    
#     def _determine_risk_level(self, volatility: float) -> str:
#         """Determine risk level based on volatility"""
#         if volatility < 0.15:
#             return "LOW"
#         elif volatility < 0.25:
#             return "MEDIUM"
#         elif volatility < 0.35:
#             return "HIGH"
#         else:
#             return "VERY_HIGH"
    
#     def _calculate_position_size(self, 
#                                   volatility: float,
#                                   technical_confidence: float,
#                                   sentiment_confidence: float) -> float:
#         """
#         Calculate position size adjusted for volatility and confidence
        
#         Logic:
#         - Higher volatility = smaller position
#         - Higher confidence = larger position
#         - Consensus (tech + sentiment aligned) = boost
#         """
#         # Base position size (inverse of volatility)
#         base_position = self.max_position_size / (1 + volatility * 10)
        
#         # Adjust for technical confidence
#         tech_multiplier = 0.5 + (technical_confidence / 100) * 0.5  # 0.5-1.0
        
#         # Adjust for sentiment (if aligned)
#         if abs(technical_confidence - sentiment_confidence) < 20:
#             consensus_bonus = 1.1
#         else:
#             consensus_bonus = 1.0
        
#         position_size = base_position * tech_multiplier * consensus_bonus
        
#         return min(self.max_position_size, position_size)
    
#     def _calculate_stop_loss(self, df: pd.DataFrame, volatility: float) -> float:
#         """
#         Calculate stop-loss as % of price
        
#         Method: Average True Range (ATR)
#         - ATR is volatility indicator
#         - Stop-loss = 1.5x ATR
#         """
#         high = df["High"]
#         low = df["Low"]
#         close = df["Close"]
        
#         # Calculate True Range
#         tr1 = high - low
#         tr2 = abs(high - close.shift(1))
#         tr3 = abs(low - close.shift(1))
        
#         tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
#         atr = tr.rolling(14).mean().iloc[-1]
        
#         # Stop-loss as % of current price
#         current_price = close.iloc[-1]
#         atr_pct = (atr / current_price) * 1.5  # 1.5x ATR
        
#         # Ensure minimum and maximum stops
#         min_stop = 0.01  # 1% minimum
#         max_stop = 0.10  # 10% maximum
        
#         return np.clip(atr_pct, min_stop, max_stop)
    
#     def _calculate_sharpe(self, df: pd.DataFrame) -> float:
#         """Calculate Sharpe Ratio"""
#         close = df["Close"]
#         returns = close.pct_change().dropna()
        
#         if returns.std() == 0:
#             return 0
        
#         sharpe = (returns.mean() * 252 - self.risk_free_rate) / (returns.std() * np.sqrt(252))
#         return sharpe
    
#     def assess_position_risk(self, 
#                             entry_price: float,
#                             stop_loss_price: float,
#                             position_size_qty: int,
#                             portfolio_value: float) -> Dict:
#         """Assess risk of a specific position"""
#         max_loss_per_share = abs(entry_price - stop_loss_price)
#         max_loss_amount = max_loss_per_share * position_size_qty
#         max_loss_pct = (max_loss_amount / portfolio_value) * 100
        
#         return {
#             "max_loss_per_share": max_loss_per_share,
#             "max_loss_amount": max_loss_amount,
#             "max_loss_pct": max_loss_pct,
#             "risk_acceptable": max_loss_pct <= 2  # Max 2% portfolio risk per trade
#         }

# trading_bot/agents/risk_agent.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
from dataclasses import dataclass

@dataclass
class RiskResult:
    ticker: str
    volatility: float
    risk_level: str
    position_size: float
    stop_loss_pct: float
    stop_loss_price: float
    take_profit_pct: float
    take_profit_price: float
    support: float
    resistance: float
    sharpe_ratio: float
    max_risk: float

class RiskManagementAgent:
    """
    Pure logic risk agent. Provides .evaluate(...) method.
    """

    def __init__(self, max_drawdown: float = 0.05, risk_free_rate: float = 0.02):
        self.max_drawdown = max_drawdown
        self.risk_free_rate = risk_free_rate
        self.max_position_size = 0.10

    def _to_df(self, df_like):
        if isinstance(df_like, pd.DataFrame):
            return df_like.copy()
        if isinstance(df_like, dict) and "df" in df_like and isinstance(df_like["df"], pd.DataFrame):
            return df_like["df"].copy()
        raise ValueError("RiskManagementAgent: expected DataFrame or {'df': DataFrame}")

    def evaluate(self,
                 ticker: str,
                 df,
                 current_price: float,
                 technical_confidence: float = 50.0,
                 sentiment_confidence: float = 50.0) -> Dict[str, Any]:
        """
        Evaluate risk metrics. Returns a dict.
        df can be DataFrame or {'df': DataFrame}
        """
        try:
            df = self._to_df(df)
        except Exception as e:
            # Return conservative defaults when df not available
            return {
                "ticker": ticker,
                "error": "no_price_data",
                "volatility": None,
                "risk_level": "UNKNOWN",
                "position_size": 0.01,
                "stop_loss_pct": 0.05,
                "stop_loss_price": None,
                "take_profit_pct": 0.10,
                "take_profit_price": None,
                "support": None,
                "resistance": None,
                "sharpe_ratio": None,
                "max_risk": self.max_drawdown
            }

        # Ensure numeric
        close = df["Close"].astype(float)
        returns = close.pct_change().dropna()
        if len(returns) == 0:
            volatility = 0.0
        else:
            daily_vol = returns.std()
            volatility = float(daily_vol * np.sqrt(252))

        # Risk level thresholds
        if volatility < 0.15:
            risk_level = "LOW"
        elif volatility < 0.25:
            risk_level = "MEDIUM"
        elif volatility < 0.35:
            risk_level = "HIGH"
        else:
            risk_level = "VERY_HIGH"

        # position sizing: lower size for higher volatility; boost with confidence
        base_position = self.max_position_size / (1 + volatility * 10)
        tech_mult = 0.5 + (float(technical_confidence) / 100) * 0.5
        consensus_bonus = 1.0
        position_size = min(self.max_position_size, base_position * tech_mult * consensus_bonus)

        # ATR-based stop loss
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).dropna()
        if len(tr) >= 14:
            atr = tr.rolling(14).mean().iloc[-1]
        else:
            atr = tr.mean() if len(tr) > 0 else 0.01 * current_price

        stop_loss_pct = np.clip((atr / current_price) * 1.5, 0.01, 0.10)
        take_profit_pct = stop_loss_pct * 2

        support = float(df["Low"].rolling(20).min().iloc[-1]) if "Low" in df.columns else None
        resistance = float(df["High"].rolling(20).max().iloc[-1]) if "High" in df.columns else None

        # Sharpe (simple)
        if returns.std() == 0 or np.isnan(returns.std()):
            sharpe = 0.0
        else:
            sharpe = float((returns.mean() * 252 - self.risk_free_rate) / (returns.std() * np.sqrt(252)))

        return {
            "ticker": ticker,
            "volatility": round(volatility, 4),
            "risk_level": risk_level,
            "position_size": round(position_size, 4),
            "stop_loss_pct": round(stop_loss_pct, 4),
            "stop_loss_price": round(current_price * (1 - stop_loss_pct), 4),
            "take_profit_pct": round(take_profit_pct, 4),
            "take_profit_price": round(current_price * (1 + take_profit_pct), 4),
            "support": support,
            "resistance": resistance,
            "sharpe_ratio": round(sharpe, 4),
            "max_risk": self.max_drawdown
        }
