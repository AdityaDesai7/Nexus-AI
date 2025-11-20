# # """
# # Technical Analysis Agent using LangChain
# # """

# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_groq import ChatGroq
# # from models.schemas import TechnicalAnalysisOutput
# # import os
# # from dotenv import load_dotenv
# # from datetime import datetime, timedelta
# # import pandas as pd
# # import numpy as np

# # # Import ONLY from data_fetcher (no circular import!)
# # from data.data_fetcher import fetch_data

# # load_dotenv()


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


# # class TechnicalAnalysisAgent:
# #     def __init__(self):
# #         self.llm = ChatGroq(
# #             groq_api_key=os.getenv("GROQ_API_KEY"), 
# #             model_name="llama-3.1-8b-instant"
# #         )
# #         self.prompt = ChatPromptTemplate.from_template(
# #             """
# #             You are a technical analysis expert. Based on the following data for {ticker}:
# #             - RSI: {rsi}
# #             - MACD: {macd}, MACD Signal: {macd_signal}
# #             - Bollinger Bands: Upper={bollinger_upper}, Lower={bollinger_lower}
# #             - Support: {support}, Resistance: {resistance}
# #             Provide a brief trading recommendation (Buy, Sell, or Hold) with a one-sentence explanation.
# #             """
# #         )

# #     def analyze(self, ticker: str) -> TechnicalAnalysisOutput:
# #         """Analyze technical indicators for a ticker"""
        
# #         # Fetch and enhance (NO circular import!)
# #         end = datetime.now()
# #         start = end - timedelta(days=180)
# #         df = fetch_data(ticker, start, end)
# #         df = add_indicators(df)
        
# #         # Extract latest values
# #         latest = df.iloc[-1]
        
# #         indicators = {
# #             "ticker": ticker,
# #             "rsi": round(latest['RSI'], 2),
# #             "macd": round(latest['MACD'], 2),
# #             "macd_signal": round(latest['MACD_Signal'], 2),
# #             "bollinger_upper": round(latest['BB_Upper'], 2),
# #             "bollinger_lower": round(latest['BB_Lower'], 2),
# #             "support": round(latest['Support'], 2),
# #             "resistance": round(latest['Resistance'], 2),
# #         }
        
# #         # Get LLM recommendation
# #         chain = self.prompt | self.llm
# #         recommendation = chain.invoke(indicators).content.strip()
        
# #         return TechnicalAnalysisOutput(
# #             ticker=ticker,
# #             rsi=indicators["rsi"],
# #             macd=indicators["macd"],
# #             macd_signal=indicators["macd_signal"],
# #             bollinger_upper=indicators["bollinger_upper"],
# #             bollinger_lower=indicators["bollinger_lower"],
# #             support=indicators["support"],
# #             resistance=indicators["resistance"],
# #             recommendation=recommendation,
# #         )


# """
# Technical Analysis Agent with Built-in Logic (No LLM Dependency)
# Implements: MACD, Bollinger Bands, and LOFS Strategy
# """

# from dataclasses import dataclass
# from models.schemas import TechnicalAnalysisOutput
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from typing import Dict, Tuple, List
# from data.data_fetcher import fetch_data


# @dataclass
# class SignalConfig:
#     """Configuration for trading signals"""
#     macd_fast_period: int = 12
#     macd_slow_period: int = 26
#     macd_signal_period: int = 9
#     bb_period: int = 20
#     bb_std_mult_normal: float = 2.0
#     bb_std_mult_tight: float = 1.5
#     lofs_buy_threshold: float = 0.02
#     lofs_sell_threshold: float = 0.03
#     rsi_period: int = 14


# class TechnicalSignalGenerator:
#     """Generate trading signals from technical indicators"""
    
#     def __init__(self, config: SignalConfig = None):
#         self.config = config or SignalConfig()
#         self.signals_log = []
    
#     def calculate_macd(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
#         """Calculate MACD using exponential moving averages
        
#         Returns:
#             macd_line, signal_line, histogram
#         """
#         close = df["Close"]
        
#         # Fast EMA (12 periods)
#         ema_fast = close.ewm(span=self.config.macd_fast_period, adjust=False).mean()
        
#         # Slow EMA (26 periods)
#         ema_slow = close.ewm(span=self.config.macd_slow_period, adjust=False).mean()
        
#         # MACD Line
#         macd_line = ema_fast - ema_slow
        
#         # Signal Line (9-period EMA of MACD)
#         signal_line = macd_line.ewm(span=self.config.macd_signal_period, adjust=False).mean()
        
#         # Histogram
#         histogram = macd_line - signal_line
        
#         return macd_line, signal_line, histogram
    
#     def detect_macd_crossover(self, macd_hist: pd.Series) -> pd.Series:
#         """Detect MACD crossovers
        
#         Returns:
#             1 = BUY (histogram < 0 and turning positive)
#             -1 = SELL (histogram > 0 and turning negative)
#             0 = NO SIGNAL
#         """
#         signals = pd.Series(0, index=macd_hist.index)
        
#         # Previous histogram value
#         prev_hist = macd_hist.shift(1)
        
#         # Product of current and previous: negative means crossover
#         crossover = macd_hist * prev_hist
        
#         # BUY: MACD crosses above signal (histogram < 0 and going up)
#         buy_signal = (crossover < 0) & (macd_hist > prev_hist) & (macd_hist < 0)
        
#         # SELL: MACD crosses below signal (histogram > 0 and going down)
#         sell_signal = (crossover < 0) & (macd_hist < prev_hist) & (macd_hist > 0)
        
#         signals[buy_signal] = 1
#         signals[sell_signal] = -1
        
#         return signals
    
#     def calculate_bollinger_bands(self, df: pd.DataFrame, std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
#         """Calculate Bollinger Bands
        
#         Returns:
#             upper_band, middle_band, lower_band
#         """
#         close = df["Close"]
        
#         # Middle band (20-period SMA)
#         middle = close.rolling(window=self.config.bb_period).mean()
        
#         # Standard deviation
#         std = close.rolling(window=self.config.bb_period).std()
        
#         # Bands
#         upper = middle + (std_mult * std)
#         lower = middle - (std_mult * std)
        
#         return upper, middle, lower
    
#     def detect_bb_bounce(self, df: pd.DataFrame, std_mult: float = 2.0) -> pd.Series:
#         """Detect Bollinger Band bounces
        
#         Returns:
#             1 = BUY (price crosses above lower band)
#             -1 = SELL (price crosses below upper band)
#             0 = NO SIGNAL
#         """
#         close = df["Close"]
#         upper, middle, lower = self.calculate_bollinger_bands(df, std_mult)
        
#         signals = pd.Series(0, index=df.index)
        
#         # Distance from bands
#         dist_from_lower = close - lower
#         dist_from_upper = upper - close
        
#         # BUY: Price crosses from below to above lower band
#         buy_cross = (dist_from_lower.shift(1) < 0) & (dist_from_lower > 0)
        
#         # SELL: Price crosses from above to below upper band
#         sell_cross = (dist_from_upper.shift(1) < 0) & (dist_from_upper > 0)
        
#         signals[buy_cross] = 1
#         signals[sell_cross] = -1
        
#         return signals
    
#     def calculate_lofs_signals(self, df: pd.DataFrame, max_lookback: int = 60) -> pd.Series:
#         """Loss Following Strategy - Average down on dips, sell on rises
        
#         Returns:
#             1 = BUY (price dropped enough to buy more)
#             -1 = SELL (price rose enough to take profit)
#             0 = NO SIGNAL
#         """
#         close = df["Close"].values
#         signals = pd.Series(0, index=df.index)
        
#         highest = np.zeros(len(close))
#         signal_array = np.zeros(len(close))
        
#         for i in range(1, len(close)):
#             # Find highest price in lookback period
#             lookback_start = max(0, i - max_lookback)
#             highest[i] = np.max(close[lookback_start:i+1])
            
#             # BUY if price drops buy_threshold% from highest
#             if close[i] < highest[i] * (1 - self.config.lofs_buy_threshold):
#                 signal_array[i] = 1
            
#             # SELL if price rises sell_threshold% from previous price
#             elif i > 1 and close[i] > close[i-1] * (1 + self.config.lofs_sell_threshold):
#                 signal_array[i] = -1
        
#         signals[:] = signal_array
#         return signals
    
#     def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
#         """Calculate Relative Strength Index"""
#         close = df["Close"]
#         delta = close.diff()
        
#         gain = delta.where(delta > 0, 0)
#         loss = -delta.where(delta < 0, 0)
        
#         avg_gain = gain.rolling(window=period).mean()
#         avg_loss = loss.rolling(window=period).mean()
        
#         rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
#         rsi = 100 - (100 / (1 + rs))
        
#         return rsi
    
#     def generate_combined_signal(self, df: pd.DataFrame) -> Dict:
#         """Combine all signals to generate final recommendation
        
#         Returns confidence-weighted signal
#         """
#         # Calculate all signals
#         macd_line, macd_signal, macd_hist = self.calculate_macd(df)
#         macd_signal_values = self.detect_macd_crossover(macd_hist)
        
#         bb_signal = self.detect_bb_bounce(df, std_mult=2.0)
#         lofs_signal = self.calculate_lofs_signals(df)
#         rsi = self.calculate_rsi(df)
        
#         # Get latest values
#         latest_idx = len(df) - 1
        
#         macd_sig = macd_signal_values.iloc[latest_idx]
#         bb_sig = bb_signal.iloc[latest_idx]
#         lofs_sig = lofs_signal.iloc[latest_idx]
#         rsi_val = rsi.iloc[latest_idx]
        
#         # Signal aggregation with weights
#         signal_weights = {
#             'macd': 0.35,
#             'bb': 0.35,
#             'lofs': 0.20,
#             'rsi': 0.10
#         }
        
#         # Calculate weighted signal
#         weighted_signal = (
#             macd_sig * signal_weights['macd'] +
#             bb_sig * signal_weights['bb'] +
#             lofs_sig * signal_weights['lofs'] +
#             (1 if rsi_val < 30 else -1 if rsi_val > 70 else 0) * signal_weights['rsi']
#         )
        
#         # Determine action and confidence
#         if weighted_signal > 0.3:
#             action = "BUY"
#             confidence = min(90, abs(weighted_signal) * 100)
#         elif weighted_signal < -0.3:
#             action = "SELL"
#             confidence = min(90, abs(weighted_signal) * 100)
#         else:
#             action = "HOLD"
#             confidence = 50
        
#         # Get latest indicators for output
#         upper, middle, lower = self.calculate_bollinger_bands(df)
        
#         signals_triggered = []
#         if macd_sig != 0:
#             signals_triggered.append(f"MACD: {'BUY' if macd_sig > 0 else 'SELL'}")
#         if bb_sig != 0:
#             signals_triggered.append(f"BB: {'BUY' if bb_sig > 0 else 'SELL'}")
#         if lofs_sig != 0:
#             signals_triggered.append(f"LOFS: {'BUY' if lofs_sig > 0 else 'SELL'}")
#         if rsi_val < 30:
#             signals_triggered.append(f"RSI: Oversold ({rsi_val:.1f})")
#         elif rsi_val > 70:
#             signals_triggered.append(f"RSI: Overbought ({rsi_val:.1f})")
        
#         return {
#             'action': action,
#             'confidence': confidence,
#             'weighted_signal': weighted_signal,
#             'signals_triggered': signals_triggered if signals_triggered else ["No signals triggered"],
#             'macd_value': macd_line.iloc[-1],
#             'macd_signal': macd_signal.iloc[-1],
#             'macd_hist': macd_hist.iloc[-1],
#             'bb_upper': upper.iloc[-1],
#             'bb_middle': middle.iloc[-1],
#             'bb_lower': lower.iloc[-1],
#             'rsi': rsi_val,
#             'macd_signal_component': macd_sig,
#             'bb_signal_component': bb_sig,
#             'lofs_signal_component': lofs_sig
#         }


# class TechnicalAnalysisAgent:
#     """Autonomous technical analysis agent - NO LLM DEPENDENCY"""
    
#     def __init__(self, config: SignalConfig = None):
#         self.signal_generator = TechnicalSignalGenerator(config)
    
#     def analyze(self, ticker: str, start_date: datetime = None, end_date: datetime = None) -> TechnicalAnalysisOutput:
#         """Analyze technical indicators and generate autonomous recommendation
        
#         NO LLM DEPENDENCY - Pure logic-based decision making
#         """
        
#         # Set default date range if not provided
#         if end_date is None:
#             end_date = datetime.now()
#         if start_date is None:
#             start_date = end_date - timedelta(days=180)
        
#         # Fetch and prepare data
#         df = fetch_data(ticker, start_date, end_date)
#         df = self._add_indicators(df)
        
#         # Generate signal
#         signal_result = self.signal_generator.generate_combined_signal(df)
        
#         # Extract latest values
#         latest = df.iloc[-1]
        
#         # Build recommendation
#         recommendation = self._build_recommendation(signal_result)
        
#         return TechnicalAnalysisOutput(
#             ticker=ticker,
#             rsi=round(signal_result['rsi'], 2),
#             macd=round(signal_result['macd_value'], 4),
#             macd_signal=round(signal_result['macd_signal'], 4),
#             bollinger_upper=round(signal_result['bb_upper'], 2),
#             bollinger_lower=round(signal_result['bb_lower'], 2),
#             support=round(latest.get('Support', 0), 2),
#             resistance=round(latest.get('Resistance', 0), 2),
#             recommendation=recommendation
#         )
    
#     def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Add support/resistance indicators"""
#         df = df.copy()
        
#         high = df["High"]
#         low = df["Low"]
        
#         # Support / Resistance
#         df["Support"] = low.rolling(20).min()
#         df["Resistance"] = high.rolling(20).max()
        
#         return df
    
#     def _build_recommendation(self, signal_result: Dict) -> str:
#         """Build human-readable recommendation"""
#         action = signal_result['action']
#         confidence = signal_result['confidence']
#         signals = signal_result['signals_triggered']
        
#         recommendation = f"{action} (Confidence: {confidence:.1f}%) - "
#         recommendation += f"Signals: {', '.join(signals)}"
        
#         return recommendation

# trading_bot/agents/technical_agent.py
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any
from data.data_fetcher import fetch_data

@dataclass
class TechnicalAnalysisOutput:
    ticker: str
    rsi: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    support: float
    resistance: float
    recommendation: str

class TechnicalAnalysisAgent:
    """
    Pure-logic technical agent. analyze(...) returns a dict or TechnicalAnalysisOutput-like object.
    """

    def __init__(self):
        self.lookback_days = 180

    def _ensure_df(self, df_or_ticker, start: datetime, end: datetime) -> pd.DataFrame:
        if isinstance(df_or_ticker, pd.DataFrame):
            df = df_or_ticker.copy()
        else:
            df = fetch_data(df_or_ticker, start, end)
        # Ensure columns and types
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.zeros(len(df)))

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean().replace(0, np.finfo(float).eps)
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

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

        # Support/resistance
        df["Support"] = low.rolling(20).min()
        df["Resistance"] = high.rolling(20).max()

        df = df.dropna().reset_index(drop=True)
        return df

    def analyze(self, ticker: str, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=self.lookback_days)

        df = self._ensure_df(ticker, start_date, end_date)
        if df is None or len(df) == 0:
            return {"error": "no_data"}

        df_ind = self._add_indicators(df)
        if df_ind.empty:
            return {"error": "insufficient_data"}

        latest = df_ind.iloc[-1]
        # Build recommendation (simple rule-based)
        rsi = float(latest["RSI"])
        macd = float(latest["MACD"])
        macd_signal = float(latest["MACD_Signal"])
        bb_upper = float(latest["BB_Upper"])
        bb_lower = float(latest["BB_Lower"])
        support = float(latest["Support"])
        resistance = float(latest["Resistance"])
        close = float(latest["Close"])

        signals = []
        if rsi < 30:
            signals.append("RSI oversold")
        if macd > macd_signal:
            signals.append("MACD bullish")
        if close < bb_lower:
            signals.append("Price below lower BB")

        # LOFS-like simple rule
        recommendation = "HOLD"
        confidence = 50.0
        if rsi < 30 and macd > macd_signal:
            recommendation = "BUY"
            confidence = 60.0
        elif rsi > 70 and macd < macd_signal:
            recommendation = "SELL"
            confidence = 60.0
        elif macd > macd_signal and close > bb_lower:
            recommendation = "BUY"
            confidence = 55.0

        rec_str = f"{recommendation} (Confidence: {confidence:.1f}%) - Signals: {', '.join(signals) if signals else 'None'}"

        return {
            "ticker": ticker,
            "action": recommendation,
            "confidence": confidence,
            "signals": signals,
            "rsi": round(rsi, 2),
            "macd": round(macd, 6),
            "macd_signal": round(macd_signal, 6),
            "macd_hist": round(float(latest["MACD_Hist"]), 6),
            "bollinger_upper": round(bb_upper, 2),
            "bollinger_lower": round(bb_lower, 2),
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "latest_close": round(close, 2),
            "raw_index": df_ind.index[-1] if hasattr(df_ind, "index") else None
        }
