
# """
# =============================================================================
# INSTITUTIONAL AGENTS - COMPLETE AUTONOMOUS TRADING SYSTEM v3.0
# =============================================================================
# 9 Advanced Institutional Detection Agents + Master Aggregator
# REALISTIC SCORING WITH VARIATION
# Author: ProTrader AI
# Date: November 4, 2025
# =============================================================================
# """

# # agents/institutional_agents.py - v4.0 REALISTIC LOGIC

# """
# INSTITUTIONAL AGENTS v4.0 - REALISTIC TRADING LOGIC
# Based on real market consensus and conflicting signal handling
# """

# import pandas as pd
# import numpy as np
# from typing import Dict, List
# from dataclasses import dataclass, field
# from datetime import datetime
# import warnings
# import logging

# warnings.filterwarnings('ignore')
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# @dataclass
# class AgentSignal:
#     agent_name: str
#     score: float
#     confidence: float
#     action: str
#     reason: str
#     details: Dict = field(default_factory=dict)
#     timestamp: datetime = field(default_factory=datetime.now)


# @dataclass
# class AggregatedSignal:
#     final_score: float
#     recommendation: str
#     confidence: float
#     agent_votes: Dict = field(default_factory=dict)
#     reasoning: str = ""
#     agent_breakdown: Dict = field(default_factory=dict)
#     timestamp: datetime = field(default_factory=datetime.now)


# # ============================================
# # ALL 9 AGENTS - REALISTIC SCORING
# # ============================================

# class FFIMomentumAgent:
#     def __init__(self):
#         self.name = "FII Momentum"
    
#     def analyze(self, fii_flow_data: Dict, price_data: pd.Series) -> AgentSignal:
#         try:
#             if not fii_flow_data or price_data is None or len(price_data) < 5:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Insufficient data", {})
            
#             # Real FII analysis - more realistic
#             fii_ratio = fii_flow_data.get('today', 100) / max(fii_flow_data.get('30_day_avg', 100), 1)
            
#             # Realistic range: -2 to +2 (not 0.3 to 5.0)
#             fii_ratio = np.clip(fii_ratio, 0.3, 2.5)
            
#             # Real thresholds
#             if fii_ratio > 2.0:
#                 return AgentSignal(self.name, 40, 60, "BUY", f"Strong FII inflow", {'fii_ratio': round(fii_ratio, 2)})
#             elif fii_ratio > 1.5:
#                 return AgentSignal(self.name, 20, 50, "SLIGHT_BUY", f"Moderate FII", {'fii_ratio': round(fii_ratio, 2)})
#             elif fii_ratio < 0.5:
#                 return AgentSignal(self.name, -40, 60, "SELL", f"Strong FII outflow", {'fii_ratio': round(fii_ratio, 2)})
#             elif fii_ratio < 0.8:
#                 return AgentSignal(self.name, -20, 50, "SLIGHT_SELL", f"Moderate outflow", {'fii_ratio': round(fii_ratio, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 35, "NEUTRAL", "FII neutral", {'fii_ratio': round(fii_ratio, 2)})
#         except Exception as e:
#             logger.error(f"FII error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class ExecutionBreakdownDetector:
#     def __init__(self):
#         self.name = "Execution Pattern"
    
#     def analyze(self, order_data: List[Dict], price_data: pd.Series) -> AgentSignal:
#         try:
#             if not order_data or len(order_data) < 10:
#                 return AgentSignal(self.name, 0, 25, "NEUTRAL", "Insufficient orders", {})
            
#             # Real execution analysis
#             recent = order_data[-50:]
#             buy_cnt = sum(1 for o in recent if o.get('direction') == 'BUY')
#             sell_cnt = sum(1 for o in recent if o.get('direction') == 'SELL')
            
#             total = buy_cnt + sell_cnt
#             if total == 0:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "No clear pattern", {})
            
#             buy_ratio = buy_cnt / total
            
#             if buy_ratio > 0.7:
#                 return AgentSignal(self.name, 35, 65, "BUY", f"Strong buy execution", {'ratio': round(buy_ratio, 2)})
#             elif buy_ratio > 0.6:
#                 return AgentSignal(self.name, 15, 50, "SLIGHT_BUY", f"Moderate buy", {'ratio': round(buy_ratio, 2)})
#             elif buy_ratio < 0.3:
#                 return AgentSignal(self.name, -35, 65, "SELL", f"Strong sell execution", {'ratio': round(buy_ratio, 2)})
#             elif buy_ratio < 0.4:
#                 return AgentSignal(self.name, -15, 50, "SLIGHT_SELL", f"Moderate sell", {'ratio': round(buy_ratio, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 40, "NEUTRAL", "Balanced", {'ratio': round(buy_ratio, 2)})
#         except Exception as e:
#             logger.error(f"Execution error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class VolumePatternRecognition:
#     def __init__(self):
#         self.name = "Volume Pattern"
    
#     def analyze(self, volume_data: pd.Series, price_data: pd.Series) -> AgentSignal:
#         try:
#             if len(volume_data) < 10 or len(price_data) < 10:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 10+ days", {})
            
#             # Real volume analysis - realistic expectations
#             vol_recent = volume_data.tail(5).mean()
#             vol_past = volume_data.tail(20).mean()
#             vol_ratio = vol_recent / vol_past if vol_past > 0 else 1
            
#             price_trend = (price_data.iloc[-1] / price_data.iloc[-10]) - 1
            
#             # Realistic: volume spikes are rare
#             if vol_ratio > 1.5 and price_trend > 0.02:
#                 return AgentSignal(self.name, 30, 55, "BUY", "Volume+Price up", {'vol_ratio': round(vol_ratio, 2)})
#             elif vol_ratio > 1.3 and price_trend > 0.01:
#                 return AgentSignal(self.name, 15, 45, "SLIGHT_BUY", "Mild volume increase", {'vol_ratio': round(vol_ratio, 2)})
#             elif vol_ratio > 1.5 and price_trend < -0.02:
#                 return AgentSignal(self.name, -30, 55, "SELL", "Volume+Price down", {'vol_ratio': round(vol_ratio, 2)})
#             elif vol_ratio > 1.3 and price_trend < -0.01:
#                 return AgentSignal(self.name, -15, 45, "SLIGHT_SELL", "Mild volume, down", {'vol_ratio': round(vol_ratio, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 35, "NEUTRAL", "Normal volume", {'vol_ratio': round(vol_ratio, 2)})
#         except Exception as e:
#             logger.error(f"Volume error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class IFICalculator:
#     def __init__(self):
#         self.name = "IFI Score"
    
#     def analyze(self, volume_data: pd.Series, price_data: pd.Series) -> AgentSignal:
#         try:
#             if len(volume_data) < 5 or len(price_data) < 5:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 5+ days", {})
            
#             # Real IFI calculation
#             today_vol = volume_data.iloc[-1]
#             avg_vol = volume_data.tail(30).mean()
#             price_change = ((price_data.iloc[-1] / price_data.iloc[-5]) - 1) * 100
            
#             ifi = (today_vol / avg_vol) * (abs(price_change) / 100) if avg_vol > 0 else 0
#             ifi = np.clip(ifi, 0.5, 5.0)  # Realistic range
            
#             # High IFI is RARE - it signals institutional activity
#             if ifi > 2.5 and price_change > 0.5:
#                 return AgentSignal(self.name, 50, 75, "STRONG_BUY", f"Very high IFI={ifi:.2f}", {'ifi': round(ifi, 2)})
#             elif ifi > 1.8 and price_change > 0.3:
#                 return AgentSignal(self.name, 25, 60, "BUY", f"High IFI={ifi:.2f}", {'ifi': round(ifi, 2)})
#             elif ifi > 2.5 and price_change < -0.5:
#                 return AgentSignal(self.name, -50, 75, "STRONG_SELL", f"Very high IFI (down)", {'ifi': round(ifi, 2)})
#             elif ifi > 1.8 and price_change < -0.3:
#                 return AgentSignal(self.name, -25, 60, "SELL", f"High IFI (down)", {'ifi': round(ifi, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 30, "NEUTRAL", f"Normal IFI={ifi:.2f}", {'ifi': round(ifi, 2)})
#         except Exception as e:
#             logger.error(f"IFI error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class AccumulationDetector:
#     def __init__(self):
#         self.name = "Accumulation"
    
#     def analyze(self, price_data: pd.Series, volume_data: pd.Series) -> AgentSignal:
#         try:
#             if len(price_data) < 20 or len(volume_data) < 20:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 20+ days", {})
            
#             # Real accumulation - signs are subtle
#             price_range = (price_data.tail(20).max() - price_data.tail(20).min()) / price_data.tail(20).min()
#             vol_avg = volume_data.tail(20).mean()
#             recent_vol = volume_data.tail(5).mean()
#             vol_increase = recent_vol / vol_avg if vol_avg > 0 else 1
            
#             price_near_low = price_data.iloc[-1] / price_data.tail(20).min()
            
#             # Realistic accumulation signs
#             if price_near_low < 1.05 and vol_increase > 1.2 and price_range < 0.1:
#                 return AgentSignal(self.name, 45, 70, "ACCUMULATING", "Signs of accumulation", {'vol_inc': round(vol_increase, 2)})
#             elif vol_increase > 1.3 and price_near_low < 1.08:
#                 return AgentSignal(self.name, 20, 55, "POSSIBLE_ACCUM", "Potential accumulation", {'vol_inc': round(vol_increase, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 35, "NORMAL", "No accumulation signs", {'vol_inc': round(vol_increase, 2)})
#         except Exception as e:
#             logger.error(f"Accumulation error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class LiquidityDetector:
#     def __init__(self):
#         self.name = "Liquidity"
    
#     def analyze(self, volume_data: pd.Series) -> AgentSignal:
#         try:
#             if len(volume_data) < 20:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 20+ days", {})
            
#             vol_ratio = volume_data.tail(5).mean() / volume_data.tail(30).mean()
            
#             if vol_ratio > 1.4:
#                 return AgentSignal(self.name, 20, 60, "HIGH_LIQ", f"Good liquidity", {'ratio': round(vol_ratio, 2)})
#             elif vol_ratio < 0.8:
#                 return AgentSignal(self.name, -15, 50, "LOW_LIQ", f"Low liquidity concern", {'ratio': round(vol_ratio, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 40, "NORMAL_LIQ", "Normal liquidity", {'ratio': round(vol_ratio, 2)})
#         except Exception as e:
#             logger.error(f"Liquidity error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class SmartMoneyTracker:
#     def __init__(self):
#         self.name = "Smart Money"
    
#     def analyze(self, price_data: pd.Series, volume_data: pd.Series) -> AgentSignal:
#         try:
#             if len(price_data) < 20 or len(volume_data) < 20:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 20+ days", {})
            
#             price_trend_10 = (price_data.iloc[-1] / price_data.iloc[-10]) - 1
#             vol_trend = volume_data.tail(10).mean() / volume_data.tail(30).mean()
            
#             # Real smart money pattern - subtle
#             if vol_trend > 1.2 and -0.01 < price_trend_10 < 0.01:
#                 return AgentSignal(self.name, 35, 65, "POSSIBLE_SMART", "Quiet accumulation pattern", {'vol': round(vol_trend, 2)})
#             elif vol_trend > 1.3 and price_trend_10 < -0.03:
#                 return AgentSignal(self.name, -35, 65, "POSSIBLE_DIST", "Quiet distribution", {'vol': round(vol_trend, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 40, "NEUTRAL", "No smart money pattern", {'vol': round(vol_trend, 2)})
#         except Exception as e:
#             logger.error(f"Smart money error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class BlockOrderTracker:
#     def __init__(self):
#         self.name = "Block Orders"
    
#     def analyze(self, order_data: List[Dict]) -> AgentSignal:
#         try:
#             if not order_data or len(order_data) < 10:
#                 return AgentSignal(self.name, 0, 25, "NEUTRAL", "No block data", {})
            
#             blocks = [o for o in order_data[-100:] if o.get('quantity', 0) > 50000]
#             if not blocks:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "No blocks", {})
            
#             buy_blocks = len([o for o in blocks if o.get('direction') == 'BUY'])
#             sell_blocks = len([o for o in blocks if o.get('direction') == 'SELL'])
            
#             if buy_blocks > sell_blocks * 1.5:
#                 return AgentSignal(self.name, 30, 60, "INSTITUTIONAL_BUY", f"{buy_blocks} buy blocks", {'ratio': round(buy_blocks/max(sell_blocks,1), 2)})
#             elif sell_blocks > buy_blocks * 1.5:
#                 return AgentSignal(self.name, -30, 60, "INSTITUTIONAL_SELL", f"{sell_blocks} sell blocks", {'ratio': round(sell_blocks/max(buy_blocks,1), 2)})
#             else:
#                 return AgentSignal(self.name, 0, 45, "NEUTRAL", "Balanced blocks", {'buy': buy_blocks, 'sell': sell_blocks})
#         except Exception as e:
#             logger.error(f"Block order error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class BreakoutDetector:
#     def __init__(self):
#         self.name = "Breakout"
    
#     def analyze(self, price_data: pd.Series, volume_data: pd.Series) -> AgentSignal:
#         try:
#             if len(price_data) < 30 or len(volume_data) < 30:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 30+ days", {})
            
#             high_30 = price_data.tail(30).max()
#             low_30 = price_data.tail(30).min()
#             curr = price_data.iloc[-1]
#             vol_conf = volume_data.iloc[-1] > volume_data.tail(30).mean() * 1.2
            
#             if curr >= high_30 * 0.98 and vol_conf:
#                 return AgentSignal(self.name, 50, 75, "BULLISH_BREAKOUT", "Resistance breakout", {'level': round(high_30, 2)})
#             elif curr >= high_30 * 0.98:
#                 return AgentSignal(self.name, 25, 55, "POTENTIAL_BREAKOUT_UP", "At resistance", {'level': round(high_30, 2)})
#             elif curr <= low_30 * 1.02 and vol_conf:
#                 return AgentSignal(self.name, -50, 75, "BEARISH_BREAKOUT", "Support breakout", {'level': round(low_30, 2)})
#             elif curr <= low_30 * 1.02:
#                 return AgentSignal(self.name, -25, 55, "POTENTIAL_BREAKOUT_DOWN", "At support", {'level': round(low_30, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 30, "NO_BREAKOUT", "In consolidation", {})
#         except Exception as e:
#             logger.error(f"Breakout error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# # ============================================
# # MASTER AGGREGATOR - REALISTIC CONSENSUS LOGIC
# # ============================================

# class MasterInstitutionalAggregator:
#     def __init__(self):
#         self.agents = {
#             'fii': FFIMomentumAgent(),
#             'execution': ExecutionBreakdownDetector(),
#             'volume': VolumePatternRecognition(),
#             'ifi': IFICalculator(),
#             'accumulation': AccumulationDetector(),
#             'liquidity': LiquidityDetector(),
#             'smart_money': SmartMoneyTracker(),
#             'block_orders': BlockOrderTracker(),
#             'breakout': BreakoutDetector(),
#         }
    
#     def aggregate(self, fii_data: Dict = None, order_data: List[Dict] = None,
#                   price_data: pd.Series = None, volume_data: pd.Series = None) -> AggregatedSignal:
#         """✅ REALISTIC: Proper consensus-based logic"""
        
#         signals = {}
        
#         try:
#             # Run all agents
#             if fii_data and price_data is not None:
#                 signals['fii'] = self.agents['fii'].analyze(fii_data, price_data)
            
#             if order_data and price_data is not None:
#                 signals['execution'] = self.agents['execution'].analyze(order_data, price_data)
            
#             if volume_data is not None and price_data is not None:
#                 signals['volume'] = self.agents['volume'].analyze(volume_data, price_data)
#                 signals['ifi'] = self.agents['ifi'].analyze(volume_data, price_data)
#                 signals['accumulation'] = self.agents['accumulation'].analyze(price_data, volume_data)
#                 signals['liquidity'] = self.agents['liquidity'].analyze(volume_data)
#                 signals['smart_money'] = self.agents['smart_money'].analyze(price_data, volume_data)
#                 signals['breakout'] = self.agents['breakout'].analyze(price_data, volume_data)
            
#             if order_data:
#                 signals['block_orders'] = self.agents['block_orders'].analyze(order_data)
            
#             if not signals:
#                 return AggregatedSignal(50, "HOLD", 0, {}, "No data", {})
            
#             # ✅ REALISTIC: Count consensus (NOT average)
#             buy_signals = []
#             sell_signals = []
#             neutral_signals = []
#             buy_score = sell_score = 0
            
#             for agent_name, signal in signals.items():
#                 if not signal:
#                     continue
                
#                 if "BUY" in signal.action.upper():
#                     buy_signals.append((agent_name, signal))
#                     buy_score += signal.score
#                 elif "SELL" in signal.action.upper():
#                     sell_signals.append((agent_name, signal))
#                     sell_score += signal.score
#                 else:
#                     neutral_signals.append((agent_name, signal))
            
#             total_agents = len(signals)
#             buy_count = len(buy_signals)
#             sell_count = len(sell_signals)
#             neutral_count = len(neutral_signals)
            
#             # ✅ CONSENSUS RULES
#             if buy_count >= 7:  # 7+ agents agree
#                 recommendation = "STRONG BUY"
#                 confidence = min(85, 70 + (buy_count - 6) * 5)
#             elif buy_count >= 5:  # 5-6 agree
#                 recommendation = "BUY"
#                 confidence = min(75, 55 + (buy_count - 4) * 5)
#             elif sell_count >= 7:
#                 recommendation = "STRONG SELL"
#                 confidence = min(85, 70 + (sell_count - 6) * 5)
#             elif sell_count >= 5:
#                 recommendation = "SELL"
#                 confidence = min(75, 55 + (sell_count - 4) * 5)
#             elif buy_count >= 4 and sell_count <= 2:
#                 recommendation = "SLIGHT BUY"
#                 confidence = 50
#             elif sell_count >= 4 and buy_count <= 2:
#                 recommendation = "SLIGHT SELL"
#                 confidence = 50
#             else:  # Conflicting signals
#                 recommendation = "HOLD"
#                 confidence = max(25, 50 - abs(buy_count - sell_count) * 5)
            
#             # Calculate score
#             if buy_count > sell_count:
#                 final_score = 50 + (buy_count / total_agents) * 30
#             elif sell_count > buy_count:
#                 final_score = 50 - (sell_count / total_agents) * 30
#             else:
#                 final_score = 50
            
#             final_score = np.clip(final_score, 0, 100)
            
#             reasoning = f"Consensus: {buy_count} BUY | {sell_count} SELL | {neutral_count} NEUTRAL => {recommendation} ({confidence:.0f}%)"
            
#             agent_votes = {name: signal.score for name, signal in signals.items()}
        
#         except Exception as e:
#             logger.error(f"Aggregation error: {e}")
#             return AggregatedSignal(50, "HOLD", 0, {}, f"Error: {str(e)}", {})
        
#         return AggregatedSignal(
#             final_score=round(final_score, 1),
#             recommendation=recommendation,
#             confidence=round(confidence, 1),
#             agent_votes=agent_votes,
#             reasoning=reasoning,
#             agent_breakdown=signals,
#             timestamp=datetime.now()
#         )


# logger.info("✅ Institutional Agents v4.0 - REALISTIC CONSENSUS LOGIC")

# agents/institutional_agents.py - v5.0 WITH REAL DATA INTEGRATION

# """
# INSTITUTIONAL AGENTS v5.0 - REAL DATA + REALISTIC LOGIC
# Uses actual NSE, Groww, and Moneycontrol data
# """

# import pandas as pd
# import numpy as np
# from typing import Dict, List
# from dataclasses import dataclass, field
# from datetime import datetime
# import warnings
# import logging


# warnings.filterwarnings('ignore')
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # ============================================
# # DATA CLASSES
# # ============================================

# @dataclass
# class AgentSignal:
#     agent_name: str
#     score: float
#     confidence: float
#     action: str
#     reason: str
#     details: Dict = field(default_factory=dict)
#     timestamp: datetime = field(default_factory=datetime.now)


# @dataclass
# class AggregatedSignal:
#     final_score: float
#     recommendation: str
#     confidence: float
#     agent_votes: Dict = field(default_factory=dict)
#     reasoning: str = ""
#     agent_breakdown: Dict = field(default_factory=dict)
#     timestamp: datetime = field(default_factory=datetime.now)


# # ============================================
# # 9 AGENTS WITH REAL DATA
# # ============================================

# class FFIMomentumAgent:
#     """Agent 1: Uses REAL FII data from NSE"""
#     def __init__(self):
#         self.name = "FII Momentum"
    
#     def analyze(self, fii_data: Dict, price_data: pd.Series) -> AgentSignal:
#         try:
#             if not fii_data or price_data is None or len(price_data) < 5:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Insufficient data", {})
            
#             # ✅ REAL DATA: Use actual FII flows from NSE
#             fii_today = fii_data.get('today', 0)
#             fii_avg = fii_data.get('30_day_avg', 1)
            
#             fii_ratio = fii_today / fii_avg if fii_avg != 0 else 1
#             fii_ratio = np.clip(fii_ratio, 0.3, 2.5)
            
#             logger.info(f"FII Agent - Today: ₹{fii_today:.0f}Cr, Avg: ₹{fii_avg:.0f}Cr, Ratio: {fii_ratio:.2f}x")
            
#             if fii_ratio > 2.0:
#                 return AgentSignal(self.name, 40, 75, "BUY", f"🔥 Strong FII INFLOW ₹{fii_today:.0f}Cr", {'fii_ratio': round(fii_ratio, 2), 'fii_today': fii_today})
#             elif fii_ratio > 1.5:
#                 return AgentSignal(self.name, 20, 60, "SLIGHT_BUY", f"✅ Moderate FII inflow", {'fii_ratio': round(fii_ratio, 2), 'fii_today': fii_today})
#             elif fii_ratio < 0.5:
#                 return AgentSignal(self.name, -40, 75, "SELL", f"⚠️ Strong FII OUTFLOW", {'fii_ratio': round(fii_ratio, 2), 'fii_today': fii_today})
#             elif fii_ratio < 0.8:
#                 return AgentSignal(self.name, -20, 60, "SLIGHT_SELL", f"⬇️ Moderate FII selling", {'fii_ratio': round(fii_ratio, 2), 'fii_today': fii_today})
#             else:
#                 return AgentSignal(self.name, 0, 40, "NEUTRAL", "FII BALANCED", {'fii_ratio': round(fii_ratio, 2)})
#         except Exception as e:
#             logger.error(f"FII error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class ExecutionBreakdownDetector:
#     """Agent 2: Uses REAL NSE bulk/block deals"""
#     def __init__(self):
#         self.name = "Execution Pattern"
    
#     def analyze(self, order_data: List[Dict], price_data: pd.Series) -> AgentSignal:
#         try:
#             if not order_data or len(order_data) < 1:
#                 return AgentSignal(self.name, 0, 25, "NEUTRAL", "No order data available", {})
            
#             # ✅ REAL DATA: Analyze actual NSE orders
#             recent = order_data[-50:] if len(order_data) >= 50 else order_data
            
#             buy_orders = [o for o in recent if o.get('direction', '').upper() == 'BUY']
#             sell_orders = [o for o in recent if o.get('direction', '').upper() == 'SELL']
            
#             total_orders = len(buy_orders) + len(sell_orders)
#             if total_orders == 0:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "No execution data", {})
            
#             buy_ratio = len(buy_orders) / total_orders
#             total_buy_qty = sum(o.get('quantity', 0) for o in buy_orders)
#             total_sell_qty = sum(o.get('quantity', 0) for o in sell_orders)
            
#             logger.info(f"Execution Agent - Buys: {len(buy_orders)}, Sells: {len(sell_orders)}, Ratio: {buy_ratio:.2f}")
            
#             if buy_ratio > 0.7:
#                 return AgentSignal(self.name, 35, 70, "BUY", f"🔥 Strong BUY execution - {len(buy_orders)} bulk orders", {'ratio': round(buy_ratio, 2), 'qty': total_buy_qty})
#             elif buy_ratio > 0.6:
#                 return AgentSignal(self.name, 15, 55, "SLIGHT_BUY", f"✅ Moderate buy execution", {'ratio': round(buy_ratio, 2)})
#             elif buy_ratio < 0.3:
#                 return AgentSignal(self.name, -35, 70, "SELL", f"⚠️ Strong SELL execution - {len(sell_orders)} blocks", {'ratio': round(buy_ratio, 2), 'qty': total_sell_qty})
#             elif buy_ratio < 0.4:
#                 return AgentSignal(self.name, -15, 55, "SLIGHT_SELL", f"⬇️ Moderate sell execution", {'ratio': round(buy_ratio, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 45, "NEUTRAL", "Balanced order flow", {'ratio': round(buy_ratio, 2)})
#         except Exception as e:
#             logger.error(f"Execution error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class VolumePatternRecognition:
#     """Agent 3: Uses real volume anomalies"""
#     def __init__(self):
#         self.name = "Volume Pattern"
    
#     def analyze(self, volume_data: pd.Series, price_data: pd.Series) -> AgentSignal:
#         try:
#             if volume_data is None or len(volume_data) < 10 or price_data is None or len(price_data) < 10:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Insufficient data", {})
            
#             # ✅ REAL DATA: Analyze actual volume from market data
#             vol_recent = volume_data.tail(5).mean()
#             vol_past = volume_data.tail(20).mean()
#             vol_ratio = vol_recent / vol_past if vol_past > 0 else 1
            
#             price_trend = (price_data.iloc[-1] / price_data.iloc[-10]) - 1 if len(price_data) >= 10 else 0
            
#             logger.info(f"Volume Agent - Ratio: {vol_ratio:.2f}x, Price Trend: {price_trend*100:.2f}%")
            
#             if vol_ratio > 1.5 and price_trend > 0.02:
#                 return AgentSignal(self.name, 35, 65, "BUY", f"🔥 Volume SPIKE with price up {price_trend*100:.1f}%", {'vol_ratio': round(vol_ratio, 2)})
#             elif vol_ratio > 1.3 and price_trend > 0.01:
#                 return AgentSignal(self.name, 15, 50, "SLIGHT_BUY", f"✅ Volume increase + uptrend", {'vol_ratio': round(vol_ratio, 2)})
#             elif vol_ratio > 1.5 and price_trend < -0.02:
#                 return AgentSignal(self.name, -35, 65, "SELL", f"⚠️ Volume spike with price down {price_trend*100:.1f}%", {'vol_ratio': round(vol_ratio, 2)})
#             elif vol_ratio > 1.3 and price_trend < -0.01:
#                 return AgentSignal(self.name, -15, 50, "SLIGHT_SELL", f"⬇️ Volume increase + downtrend", {'vol_ratio': round(vol_ratio, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 40, "NEUTRAL", "Normal volume patterns", {'vol_ratio': round(vol_ratio, 2)})
#         except Exception as e:
#             logger.error(f"Volume error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class IFICalculator:
#     """Agent 4: Institutional Footprint Indicator"""
#     def __init__(self):
#         self.name = "IFI Score"
    
#     def analyze(self, volume_data: pd.Series, price_data: pd.Series) -> AgentSignal:
#         try:
#             if volume_data is None or len(volume_data) < 5 or price_data is None or len(price_data) < 5:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 5+ days data", {})
            
#             # ✅ REAL DATA: Calculate IFI from actual market data
#             today_vol = volume_data.iloc[-1]
#             avg_vol = volume_data.tail(30).mean()
#             price_change = ((price_data.iloc[-1] / price_data.iloc[-5]) - 1) * 100
            
#             ifi = (today_vol / avg_vol) * (abs(price_change) / 100) if avg_vol > 0 else 0
#             ifi = np.clip(ifi, 0.5, 5.0)
            
#             logger.info(f"IFI Agent - IFI Score: {ifi:.2f}, Price Change: {price_change:.2f}%")
            
#             if ifi > 2.5 and price_change > 0.5:
#                 return AgentSignal(self.name, 50, 80, "STRONG_BUY", f"🔥 EXTREME institutional activity IFI={ifi:.2f}", {'ifi': round(ifi, 2)})
#             elif ifi > 1.8 and price_change > 0.3:
#                 return AgentSignal(self.name, 30, 65, "BUY", f"✅ High IFI={ifi:.2f} (Buying)", {'ifi': round(ifi, 2)})
#             elif ifi > 2.5 and price_change < -0.5:
#                 return AgentSignal(self.name, -50, 80, "STRONG_SELL", f"⚠️ EXTREME institutional selling IFI={ifi:.2f}", {'ifi': round(ifi, 2)})
#             elif ifi > 1.8 and price_change < -0.3:
#                 return AgentSignal(self.name, -30, 65, "SELL", f"⬇️ High IFI={ifi:.2f} (Selling)", {'ifi': round(ifi, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 35, "NEUTRAL", f"Normal IFI={ifi:.2f}", {'ifi': round(ifi, 2)})
#         except Exception as e:
#             logger.error(f"IFI error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class AccumulationDetector:
#     """Agent 5: Detects quiet institutional buying"""
#     def __init__(self):
#         self.name = "Accumulation"
    
#     def analyze(self, price_data: pd.Series, volume_data: pd.Series) -> AgentSignal:
#         try:
#             if price_data is None or volume_data is None or len(price_data) < 20 or len(volume_data) < 20:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 20+ days data", {})
            
#             # ✅ REAL DATA: Analyze actual price/volume patterns
#             recent_price = price_data.tail(20)
#             recent_vol = volume_data.tail(20)
            
#             price_range = (recent_price.max() - recent_price.min()) / recent_price.min()
#             vol_avg = recent_vol.mean()
#             vol_increase = vol_avg / volume_data.tail(60).mean() if len(volume_data) >= 60 else 1
            
#             price_near_low = price_data.iloc[-1] / recent_price.min()
            
#             logger.info(f"Accumulation Agent - Price/Low: {price_near_low:.2f}, Vol Increase: {vol_increase:.2f}x")
            
#             if price_near_low < 1.05 and vol_increase > 1.2 and price_range < 0.1:
#                 return AgentSignal(self.name, 50, 75, "ACCUMULATING", f"🔥 ACCUMULATION phase detected! Near 20-day low", {'vol_inc': round(vol_increase, 2)})
#             elif vol_increase > 1.3 and price_near_low < 1.08:
#                 return AgentSignal(self.name, 25, 60, "POSSIBLE_ACCUM", f"✅ Possible accumulation phase", {'vol_inc': round(vol_increase, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 40, "NORMAL", "No accumulation signals", {'vol_inc': round(vol_increase, 2)})
#         except Exception as e:
#             logger.error(f"Accumulation error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class LiquidityDetector:
#     """Agent 6: Market liquidity conditions"""
#     def __init__(self):
#         self.name = "Liquidity"
    
#     def analyze(self, volume_data: pd.Series) -> AgentSignal:
#         try:
#             if volume_data is None or len(volume_data) < 20:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 20+ days data", {})
            
#             # ✅ REAL DATA: Check actual trading volumes
#             vol_recent = volume_data.tail(5).mean()
#             vol_avg_30 = volume_data.tail(30).mean()
#             vol_ratio = vol_recent / vol_avg_30 if vol_avg_30 > 0 else 1
            
#             logger.info(f"Liquidity Agent - Recent Vol: {vol_recent:,.0f}, 30-day Avg: {vol_avg_30:,.0f}, Ratio: {vol_ratio:.2f}x")
            
#             if vol_ratio > 1.4:
#                 return AgentSignal(self.name, 20, 65, "HIGH_LIQ", f"✅ EXCELLENT liquidity - Easy to trade {vol_ratio:.2f}x", {'ratio': round(vol_ratio, 2)})
#             elif vol_ratio < 0.8:
#                 return AgentSignal(self.name, -20, 60, "LOW_LIQ", f"⚠️ LOW liquidity - {vol_ratio:.2f}x (trading difficult)", {'ratio': round(vol_ratio, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 45, "NORMAL_LIQ", "Normal liquidity conditions", {'ratio': round(vol_ratio, 2)})
#         except Exception as e:
#             logger.error(f"Liquidity error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class SmartMoneyTracker:
#     """Agent 7: Tracks smart money patterns"""
#     def __init__(self):
#         self.name = "Smart Money"
    
#     def analyze(self, price_data: pd.Series, volume_data: pd.Series) -> AgentSignal:
#         try:
#             if price_data is None or volume_data is None or len(price_data) < 20 or len(volume_data) < 20:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 20+ days data", {})
            
#             # ✅ REAL DATA: Detect smart money patterns
#             price_trend_10 = (price_data.iloc[-1] / price_data.iloc[-10]) - 1
#             vol_recent = volume_data.tail(10).mean()
#             vol_past = volume_data.tail(30).mean()
#             vol_trend = vol_recent / vol_past if vol_past > 0 else 1
            
#             logger.info(f"Smart Money Agent - Price Trend (10d): {price_trend_10*100:.2f}%, Vol Trend: {vol_trend:.2f}x")
            
#             if vol_trend > 1.2 and -0.01 < price_trend_10 < 0.01:
#                 return AgentSignal(self.name, 40, 70, "POSSIBLE_SMART", f"🔥 QUIET accumulation pattern - High vol, flat price!", {'vol': round(vol_trend, 2)})
#             elif vol_trend > 1.3 and price_trend_10 < -0.03:
#                 return AgentSignal(self.name, -40, 70, "POSSIBLE_DIST", f"⚠️ QUIET distribution - High vol, falling price", {'vol': round(vol_trend, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 45, "NEUTRAL", "No clear smart money pattern", {'vol': round(vol_trend, 2)})
#         except Exception as e:
#             logger.error(f"Smart money error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class BlockOrderTracker:
#     """Agent 8: Tracks large institutional block orders"""
#     def __init__(self):
#         self.name = "Block Orders"
    
#     def analyze(self, order_data: List[Dict]) -> AgentSignal:
#         try:
#             if not order_data:
#                 return AgentSignal(self.name, 0, 25, "NEUTRAL", "No block order data", {})
            
#             # ✅ REAL DATA: Analyze actual block orders
#             blocks = [o for o in order_data if o.get('quantity', 0) > 50000]
            
#             if not blocks:
#                 return AgentSignal(self.name, 0, 25, "NEUTRAL", "No large blocks today", {})
            
#             buy_blocks = [o for o in blocks if o.get('direction', '').upper() == 'BUY']
#             sell_blocks = [o for o in blocks if o.get('direction', '').upper() == 'SELL']
            
#             buy_qty = sum(o.get('quantity', 0) for o in buy_blocks)
#             sell_qty = sum(o.get('quantity', 0) for o in sell_blocks)
            
#             logger.info(f"Block Orders - Buy blocks: {len(buy_blocks)} ({buy_qty:,.0f} shares), Sell: {len(sell_blocks)} ({sell_qty:,.0f} shares)")
            
#             if len(buy_blocks) > len(sell_blocks) * 1.5:
#                 return AgentSignal(self.name, 40, 70, "INSTITUTIONAL_BUY", f"🔥 INSTITUTIONAL BUYING - {len(buy_blocks)} bulk blocks", {'qty': buy_qty})
#             elif len(sell_blocks) > len(buy_blocks) * 1.5:
#                 return AgentSignal(self.name, -40, 70, "INSTITUTIONAL_SELL", f"⚠️ INSTITUTIONAL SELLING - {len(sell_blocks)} bulk blocks", {'qty': sell_qty})
#             else:
#                 return AgentSignal(self.name, 0, 50, "NEUTRAL", f"Mixed: {len(buy_blocks)} buy, {len(sell_blocks)} sell blocks", {'buy': len(buy_blocks), 'sell': len(sell_blocks)})
#         except Exception as e:
#             logger.error(f"Block order error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# class BreakoutDetector:
#     """Agent 9: Detects price breakouts"""
#     def __init__(self):
#         self.name = "Breakout"
    
#     def analyze(self, price_data: pd.Series, volume_data: pd.Series) -> AgentSignal:
#         try:
#             if price_data is None or volume_data is None or len(price_data) < 30 or len(volume_data) < 30:
#                 return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 30+ days data", {})
            
#             # ✅ REAL DATA: Detect actual breakouts
#             high_30 = price_data.tail(30).max()
#             low_30 = price_data.tail(30).min()
#             current = price_data.iloc[-1]
            
#             vol_current = volume_data.iloc[-1]
#             vol_avg_30 = volume_data.tail(30).mean()
#             vol_confirm = vol_current > vol_avg_30 * 1.2
            
#             logger.info(f"Breakout Agent - Current: {current:.2f}, 30-day High: {high_30:.2f}, Low: {low_30:.2f}, Vol Confirm: {vol_confirm}")
            
#             if current >= high_30 * 0.98 and vol_confirm:
#                 return AgentSignal(self.name, 55, 80, "BULLISH_BREAKOUT", f"🔥 BULLISH BREAKOUT above ₹{high_30:.2f}!", {'level': round(high_30, 2)})
#             elif current >= high_30 * 0.98:
#                 return AgentSignal(self.name, 28, 60, "POTENTIAL_BREAKOUT_UP", f"⚠️ Testing resistance at ₹{high_30:.2f}", {'level': round(high_30, 2)})
#             elif current <= low_30 * 1.02 and vol_confirm:
#                 return AgentSignal(self.name, -55, 80, "BEARISH_BREAKOUT", f"⬇️ BEARISH BREAKDOWN below ₹{low_30:.2f}!", {'level': round(low_30, 2)})
#             elif current <= low_30 * 1.02:
#                 return AgentSignal(self.name, -28, 60, "POTENTIAL_BREAKOUT_DOWN", f"⚠️ Testing support at ₹{low_30:.2f}", {'level': round(low_30, 2)})
#             else:
#                 return AgentSignal(self.name, 0, 35, "NO_BREAKOUT", "Price in consolidation zone", {})
#         except Exception as e:
#             logger.error(f"Breakout error: {e}")
#             return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# # ============================================
# # MASTER AGGREGATOR - CONSENSUS LOGIC
# # ============================================

# class MasterInstitutionalAggregator:
#     """Aggregates all 9 agents with real data"""
    
#     def __init__(self):
#         self.agents = {
#             'fii': FFIMomentumAgent(),
#             'execution': ExecutionBreakdownDetector(),
#             'volume': VolumePatternRecognition(),
#             'ifi': IFICalculator(),
#             'accumulation': AccumulationDetector(),
#             'liquidity': LiquidityDetector(),
#             'smart_money': SmartMoneyTracker(),
#             'block_orders': BlockOrderTracker(),
#             'breakout': BreakoutDetector(),
#         }
    
#     def aggregate(self, fii_data: Dict = None, order_data: List[Dict] = None,
#                   price_data: pd.Series = None, volume_data: pd.Series = None) -> AggregatedSignal:
#         """Aggregate all 9 agents with REAL DATA"""
        
#         signals = {}
        
#         try:
#             # Run all 9 agents
#             if fii_data is not None and price_data is not None and len(price_data) > 0:
#                 signals['fii'] = self.agents['fii'].analyze(fii_data, price_data)
            
#             if order_data is not None and price_data is not None and len(price_data) > 0:
#                 signals['execution'] = self.agents['execution'].analyze(order_data, price_data)
            
#             if volume_data is not None and price_data is not None and len(volume_data) > 0 and len(price_data) > 0:
#                 signals['volume'] = self.agents['volume'].analyze(volume_data, price_data)
#                 signals['ifi'] = self.agents['ifi'].analyze(volume_data, price_data)
#                 signals['accumulation'] = self.agents['accumulation'].analyze(price_data, volume_data)
#                 signals['liquidity'] = self.agents['liquidity'].analyze(volume_data)
#                 signals['smart_money'] = self.agents['smart_money'].analyze(price_data, volume_data)
#                 signals['breakout'] = self.agents['breakout'].analyze(price_data, volume_data)
            
#             if order_data is not None:
#                 signals['block_orders'] = self.agents['block_orders'].analyze(order_data)
            
#             if not signals:
#                 return AggregatedSignal(50, "HOLD", 0, {}, "No data available", {})
            
#             # ✅ CONSENSUS VOTING
#             buy_signals = []
#             sell_signals = []
#             neutral_signals = []
            
#             for agent_name, signal in signals.items():
#                 if signal is None:
#                     continue
                
#                 action_upper = signal.action.upper()
                
#                 if "BUY" in action_upper:
#                     buy_signals.append((agent_name, signal))
#                 elif "SELL" in action_upper:
#                     sell_signals.append((agent_name, signal))
#                 else:
#                     neutral_signals.append((agent_name, signal))
            
#             total_agents = len(signals)
#             buy_count = len(buy_signals)
#             sell_count = len(sell_signals)
#             neutral_count = len(neutral_signals)
            
#             # Consensus rules
#             if buy_count >= 7:
#                 recommendation = "STRONG BUY 🔥"
#                 confidence = min(90, 75 + (buy_count - 6) * 5)
#             elif buy_count >= 5:
#                 recommendation = "BUY ✅"
#                 confidence = min(80, 60 + (buy_count - 4) * 5)
#             elif sell_count >= 7:
#                 recommendation = "STRONG SELL ⚠️"
#                 confidence = min(90, 75 + (sell_count - 6) * 5)
#             elif sell_count >= 5:
#                 recommendation = "SELL ⬇️"
#                 confidence = min(80, 60 + (sell_count - 4) * 5)
#             elif buy_count >= 4 and sell_count <= 2:
#                 recommendation = "SLIGHT BUY"
#                 confidence = 55
#             elif sell_count >= 4 and buy_count <= 2:
#                 recommendation = "SLIGHT SELL"
#                 confidence = 55
#             else:
#                 recommendation = "HOLD ⏸️"
#                 confidence = max(30, 50 - abs(buy_count - sell_count) * 8)
            
#             # Calculate final score
#             if buy_count > sell_count:
#                 final_score = 50 + (buy_count / total_agents) * 40
#             elif sell_count > buy_count:
#                 final_score = 50 - (sell_count / total_agents) * 40
#             else:
#                 final_score = 50
            
#             final_score = np.clip(final_score, 0, 100)
            
#             reasoning = f"📊 Consensus: {buy_count}🟢 BUY | {sell_count}🔴 SELL | {neutral_count}⚪ NEUTRAL → {recommendation}"
            
#             agent_votes = {name: signal.score for name, signal in signals.items()}
            
#             logger.info(f"✅ Master Result: {recommendation} (Score: {final_score:.1f}/100, Confidence: {confidence:.0f}%)")
        
#         except Exception as e:
#             logger.error(f"Aggregation error: {e}")
#             import traceback
#             logger.error(traceback.format_exc())
#             return AggregatedSignal(50, "HOLD ⏸️", 0, {}, f"Error: {str(e)}", {})
        
#         return AggregatedSignal(
#             final_score=round(final_score, 1),
#             recommendation=recommendation,
#             confidence=round(confidence, 1),
#             agent_votes=agent_votes,
#             reasoning=reasoning,
#             agent_breakdown=signals,
#             timestamp=datetime.now()
#         )
# agents/institutional_agents.py - v5.0 WITH REAL DATA INTEGRATION (CIRCULAR IMPORT FIXED)

"""
INSTITUTIONAL AGENTS v5.0 - REAL DATA + REALISTIC LOGIC
Uses actual NSE, Groww, and Moneycontrol data
NO CIRCULAR IMPORTS
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import logging


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class AgentSignal:
    agent_name: str
    score: float
    confidence: float
    action: str
    reason: str
    details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedSignal:
    final_score: float
    recommendation: str
    confidence: float
    agent_votes: Dict = field(default_factory=dict)
    reasoning: str = ""
    agent_breakdown: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================
# 9 AGENTS WITH REAL DATA
# ============================================

class FFIMomentumAgent:
    """Agent 1: Uses REAL FII data from NSE"""
    def __init__(self):
        self.name = "FII Momentum"
    
    def analyze(self, fii_data: Dict, price_data: pd.Series) -> AgentSignal:
        try:
            if not fii_data or price_data is None or len(price_data) < 5:
                return AgentSignal(self.name, 0, 20, "NEUTRAL", "Insufficient data", {})
            
            # ✅ REAL DATA: Use actual FII flows from NSE
            fii_today = fii_data.get('today', 0)
            fii_avg = fii_data.get('30_day_avg', 1)
            
            fii_ratio = fii_today / fii_avg if fii_avg != 0 else 1
            fii_ratio = np.clip(fii_ratio, 0.3, 2.5)
            
            logger.info(f"FII Agent - Today: ₹{fii_today:.0f}Cr, Avg: ₹{fii_avg:.0f}Cr, Ratio: {fii_ratio:.2f}x")
            
            if fii_ratio > 2.0:
                return AgentSignal(self.name, 40, 75, "BUY", f"🔥 Strong FII INFLOW ₹{fii_today:.0f}Cr", {'fii_ratio': round(fii_ratio, 2), 'fii_today': fii_today})
            elif fii_ratio > 1.5:
                return AgentSignal(self.name, 20, 60, "SLIGHT_BUY", f"✅ Moderate FII inflow", {'fii_ratio': round(fii_ratio, 2), 'fii_today': fii_today})
            elif fii_ratio < 0.5:
                return AgentSignal(self.name, -40, 75, "SELL", f"⚠️ Strong FII OUTFLOW", {'fii_ratio': round(fii_ratio, 2), 'fii_today': fii_today})
            elif fii_ratio < 0.8:
                return AgentSignal(self.name, -20, 60, "SLIGHT_SELL", f"⬇️ Moderate FII selling", {'fii_ratio': round(fii_ratio, 2), 'fii_today': fii_today})
            else:
                return AgentSignal(self.name, 0, 40, "NEUTRAL", "FII BALANCED", {'fii_ratio': round(fii_ratio, 2)})
        except Exception as e:
            logger.error(f"FII error: {e}")
            return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


class ExecutionBreakdownDetector:
    """Agent 2: Uses REAL NSE bulk/block deals"""
    def __init__(self):
        self.name = "Execution Pattern"
    
    def analyze(self, order_data: List[Dict], price_data: pd.Series) -> AgentSignal:
        try:
            if not order_data or len(order_data) < 1:
                return AgentSignal(self.name, 0, 25, "NEUTRAL", "No order data available", {})
            
            # ✅ REAL DATA: Analyze actual NSE orders
            recent = order_data[-50:] if len(order_data) >= 50 else order_data
            
            buy_orders = [o for o in recent if o.get('direction', '').upper() == 'BUY']
            sell_orders = [o for o in recent if o.get('direction', '').upper() == 'SELL']
            
            total_orders = len(buy_orders) + len(sell_orders)
            if total_orders == 0:
                return AgentSignal(self.name, 0, 20, "NEUTRAL", "No execution data", {})
            
            buy_ratio = len(buy_orders) / total_orders
            total_buy_qty = sum(o.get('quantity', 0) for o in buy_orders)
            total_sell_qty = sum(o.get('quantity', 0) for o in sell_orders)
            
            logger.info(f"Execution Agent - Buys: {len(buy_orders)}, Sells: {len(sell_orders)}, Ratio: {buy_ratio:.2f}")
            
            if buy_ratio > 0.7:
                return AgentSignal(self.name, 35, 70, "BUY", f"🔥 Strong BUY execution - {len(buy_orders)} bulk orders", {'ratio': round(buy_ratio, 2), 'qty': total_buy_qty})
            elif buy_ratio > 0.6:
                return AgentSignal(self.name, 15, 55, "SLIGHT_BUY", f"✅ Moderate buy execution", {'ratio': round(buy_ratio, 2)})
            elif buy_ratio < 0.3:
                return AgentSignal(self.name, -35, 70, "SELL", f"⚠️ Strong SELL execution - {len(sell_orders)} blocks", {'ratio': round(buy_ratio, 2), 'qty': total_sell_qty})
            elif buy_ratio < 0.4:
                return AgentSignal(self.name, -15, 55, "SLIGHT_SELL", f"⬇️ Moderate sell execution", {'ratio': round(buy_ratio, 2)})
            else:
                return AgentSignal(self.name, 0, 45, "NEUTRAL", "Balanced order flow", {'ratio': round(buy_ratio, 2)})
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


class VolumePatternRecognition:
    """Agent 3: Uses real volume anomalies"""
    def __init__(self):
        self.name = "Volume Pattern"
    
    def analyze(self, volume_data: pd.Series, price_data: pd.Series) -> AgentSignal:
        try:
            if volume_data is None or len(volume_data) < 10 or price_data is None or len(price_data) < 10:
                return AgentSignal(self.name, 0, 20, "NEUTRAL", "Insufficient data", {})
            
            # ✅ REAL DATA: Analyze actual volume from market data
            vol_recent = volume_data.tail(5).mean()
            vol_past = volume_data.tail(20).mean()
            vol_ratio = vol_recent / vol_past if vol_past > 0 else 1
            
            price_trend = (price_data.iloc[-1] / price_data.iloc[-10]) - 1 if len(price_data) >= 10 else 0
            
            logger.info(f"Volume Agent - Ratio: {vol_ratio:.2f}x, Price Trend: {price_trend*100:.2f}%")
            
            if vol_ratio > 1.5 and price_trend > 0.02:
                return AgentSignal(self.name, 35, 65, "BUY", f"🔥 Volume SPIKE with price up {price_trend*100:.1f}%", {'vol_ratio': round(vol_ratio, 2)})
            elif vol_ratio > 1.3 and price_trend > 0.01:
                return AgentSignal(self.name, 15, 50, "SLIGHT_BUY", f"✅ Volume increase + uptrend", {'vol_ratio': round(vol_ratio, 2)})
            elif vol_ratio > 1.5 and price_trend < -0.02:
                return AgentSignal(self.name, -35, 65, "SELL", f"⚠️ Volume spike with price down {price_trend*100:.1f}%", {'vol_ratio': round(vol_ratio, 2)})
            elif vol_ratio > 1.3 and price_trend < -0.01:
                return AgentSignal(self.name, -15, 50, "SLIGHT_SELL", f"⬇️ Volume increase + downtrend", {'vol_ratio': round(vol_ratio, 2)})
            else:
                return AgentSignal(self.name, 0, 40, "NEUTRAL", "Normal volume patterns", {'vol_ratio': round(vol_ratio, 2)})
        except Exception as e:
            logger.error(f"Volume error: {e}")
            return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


class IFICalculator:
    """Agent 4: Institutional Footprint Indicator"""
    def __init__(self):
        self.name = "IFI Score"
    
    def analyze(self, volume_data: pd.Series, price_data: pd.Series) -> AgentSignal:
        try:
            if volume_data is None or len(volume_data) < 5 or price_data is None or len(price_data) < 5:
                return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 5+ days data", {})
            
            # ✅ REAL DATA: Calculate IFI from actual market data
            today_vol = volume_data.iloc[-1]
            avg_vol = volume_data.tail(30).mean()
            price_change = ((price_data.iloc[-1] / price_data.iloc[-5]) - 1) * 100
            
            ifi = (today_vol / avg_vol) * (abs(price_change) / 100) if avg_vol > 0 else 0
            ifi = np.clip(ifi, 0.5, 5.0)
            
            logger.info(f"IFI Agent - IFI Score: {ifi:.2f}, Price Change: {price_change:.2f}%")
            
            if ifi > 2.5 and price_change > 0.5:
                return AgentSignal(self.name, 50, 80, "STRONG_BUY", f"🔥 EXTREME institutional activity IFI={ifi:.2f}", {'ifi': round(ifi, 2)})
            elif ifi > 1.8 and price_change > 0.3:
                return AgentSignal(self.name, 30, 65, "BUY", f"✅ High IFI={ifi:.2f} (Buying)", {'ifi': round(ifi, 2)})
            elif ifi > 2.5 and price_change < -0.5:
                return AgentSignal(self.name, -50, 80, "STRONG_SELL", f"⚠️ EXTREME institutional selling IFI={ifi:.2f}", {'ifi': round(ifi, 2)})
            elif ifi > 1.8 and price_change < -0.3:
                return AgentSignal(self.name, -30, 65, "SELL", f"⬇️ High IFI={ifi:.2f} (Selling)", {'ifi': round(ifi, 2)})
            else:
                return AgentSignal(self.name, 0, 35, "NEUTRAL", f"Normal IFI={ifi:.2f}", {'ifi': round(ifi, 2)})
        except Exception as e:
            logger.error(f"IFI error: {e}")
            return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


class AccumulationDetector:
    """Agent 5: Detects quiet institutional buying"""
    def __init__(self):
        self.name = "Accumulation"
    
    def analyze(self, price_data: pd.Series, volume_data: pd.Series) -> AgentSignal:
        try:
            if price_data is None or volume_data is None or len(price_data) < 20 or len(volume_data) < 20:
                return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 20+ days data", {})
            
            # ✅ REAL DATA: Analyze actual price/volume patterns
            recent_price = price_data.tail(20)
            recent_vol = volume_data.tail(20)
            
            price_range = (recent_price.max() - recent_price.min()) / recent_price.min()
            vol_avg = recent_vol.mean()
            vol_increase = vol_avg / volume_data.tail(60).mean() if len(volume_data) >= 60 else 1
            
            price_near_low = price_data.iloc[-1] / recent_price.min()
            
            logger.info(f"Accumulation Agent - Price/Low: {price_near_low:.2f}, Vol Increase: {vol_increase:.2f}x")
            
            if price_near_low < 1.05 and vol_increase > 1.2 and price_range < 0.1:
                return AgentSignal(self.name, 50, 75, "ACCUMULATING", f"🔥 ACCUMULATION phase detected! Near 20-day low", {'vol_inc': round(vol_increase, 2)})
            elif vol_increase > 1.3 and price_near_low < 1.08:
                return AgentSignal(self.name, 25, 60, "POSSIBLE_ACCUM", f"✅ Possible accumulation phase", {'vol_inc': round(vol_increase, 2)})
            else:
                return AgentSignal(self.name, 0, 40, "NORMAL", "No accumulation signals", {'vol_inc': round(vol_increase, 2)})
        except Exception as e:
            logger.error(f"Accumulation error: {e}")
            return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


class LiquidityDetector:
    """Agent 6: Market liquidity conditions"""
    def __init__(self):
        self.name = "Liquidity"
    
    def analyze(self, volume_data: pd.Series) -> AgentSignal:
        try:
            if volume_data is None or len(volume_data) < 20:
                return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 20+ days data", {})
            
            # ✅ REAL DATA: Check actual trading volumes
            vol_recent = volume_data.tail(5).mean()
            vol_avg_30 = volume_data.tail(30).mean()
            vol_ratio = vol_recent / vol_avg_30 if vol_avg_30 > 0 else 1
            
            logger.info(f"Liquidity Agent - Recent Vol: {vol_recent:,.0f}, 30-day Avg: {vol_avg_30:,.0f}, Ratio: {vol_ratio:.2f}x")
            
            if vol_ratio > 1.4:
                return AgentSignal(self.name, 20, 65, "HIGH_LIQ", f"✅ EXCELLENT liquidity - Easy to trade {vol_ratio:.2f}x", {'ratio': round(vol_ratio, 2)})
            elif vol_ratio < 0.8:
                return AgentSignal(self.name, -20, 60, "LOW_LIQ", f"⚠️ LOW liquidity - {vol_ratio:.2f}x (trading difficult)", {'ratio': round(vol_ratio, 2)})
            else:
                return AgentSignal(self.name, 0, 45, "NORMAL_LIQ", "Normal liquidity conditions", {'ratio': round(vol_ratio, 2)})
        except Exception as e:
            logger.error(f"Liquidity error: {e}")
            return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


class SmartMoneyTracker:
    """Agent 7: Tracks smart money patterns"""
    def __init__(self):
        self.name = "Smart Money"
    
    def analyze(self, price_data: pd.Series, volume_data: pd.Series) -> AgentSignal:
        try:
            if price_data is None or volume_data is None or len(price_data) < 20 or len(volume_data) < 20:
                return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 20+ days data", {})
            
            # ✅ REAL DATA: Detect smart money patterns
            price_trend_10 = (price_data.iloc[-1] / price_data.iloc[-10]) - 1
            vol_recent = volume_data.tail(10).mean()
            vol_past = volume_data.tail(30).mean()
            vol_trend = vol_recent / vol_past if vol_past > 0 else 1
            
            logger.info(f"Smart Money Agent - Price Trend (10d): {price_trend_10*100:.2f}%, Vol Trend: {vol_trend:.2f}x")
            
            if vol_trend > 1.2 and -0.01 < price_trend_10 < 0.01:
                return AgentSignal(self.name, 40, 70, "POSSIBLE_SMART", f"🔥 QUIET accumulation pattern - High vol, flat price!", {'vol': round(vol_trend, 2)})
            elif vol_trend > 1.3 and price_trend_10 < -0.03:
                return AgentSignal(self.name, -40, 70, "POSSIBLE_DIST", f"⚠️ QUIET distribution - High vol, falling price", {'vol': round(vol_trend, 2)})
            else:
                return AgentSignal(self.name, 0, 45, "NEUTRAL", "No clear smart money pattern", {'vol': round(vol_trend, 2)})
        except Exception as e:
            logger.error(f"Smart money error: {e}")
            return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


class BlockOrderTracker:
    """Agent 8: Tracks large institutional block orders"""
    def __init__(self):
        self.name = "Block Orders"
    
    def analyze(self, order_data: List[Dict]) -> AgentSignal:
        try:
            if not order_data:
                return AgentSignal(self.name, 0, 25, "NEUTRAL", "No block order data", {})
            
            # ✅ REAL DATA: Analyze actual block orders
            blocks = [o for o in order_data if o.get('quantity', 0) > 50000]
            
            if not blocks:
                return AgentSignal(self.name, 0, 25, "NEUTRAL", "No large blocks today", {})
            
            buy_blocks = [o for o in blocks if o.get('direction', '').upper() == 'BUY']
            sell_blocks = [o for o in blocks if o.get('direction', '').upper() == 'SELL']
            
            buy_qty = sum(o.get('quantity', 0) for o in buy_blocks)
            sell_qty = sum(o.get('quantity', 0) for o in sell_blocks)
            
            logger.info(f"Block Orders - Buy blocks: {len(buy_blocks)} ({buy_qty:,.0f} shares), Sell: {len(sell_blocks)} ({sell_qty:,.0f} shares)")
            
            if len(buy_blocks) > len(sell_blocks) * 1.5:
                return AgentSignal(self.name, 40, 70, "INSTITUTIONAL_BUY", f"🔥 INSTITUTIONAL BUYING - {len(buy_blocks)} bulk blocks", {'qty': buy_qty})
            elif len(sell_blocks) > len(buy_blocks) * 1.5:
                return AgentSignal(self.name, -40, 70, "INSTITUTIONAL_SELL", f"⚠️ INSTITUTIONAL SELLING - {len(sell_blocks)} bulk blocks", {'qty': sell_qty})
            else:
                return AgentSignal(self.name, 0, 50, "NEUTRAL", f"Mixed: {len(buy_blocks)} buy, {len(sell_blocks)} sell blocks", {'buy': len(buy_blocks), 'sell': len(sell_blocks)})
        except Exception as e:
            logger.error(f"Block order error: {e}")
            return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


class BreakoutDetector:
    """Agent 9: Detects price breakouts"""
    def __init__(self):
        self.name = "Breakout"
    
    def analyze(self, price_data: pd.Series, volume_data: pd.Series) -> AgentSignal:
        try:
            if price_data is None or volume_data is None or len(price_data) < 30 or len(volume_data) < 30:
                return AgentSignal(self.name, 0, 20, "NEUTRAL", "Need 30+ days data", {})
            
            # ✅ REAL DATA: Detect actual breakouts
            high_30 = price_data.tail(30).max()
            low_30 = price_data.tail(30).min()
            current = price_data.iloc[-1]
            
            vol_current = volume_data.iloc[-1]
            vol_avg_30 = volume_data.tail(30).mean()
            vol_confirm = vol_current > vol_avg_30 * 1.2
            
            logger.info(f"Breakout Agent - Current: {current:.2f}, 30-day High: {high_30:.2f}, Low: {low_30:.2f}, Vol Confirm: {vol_confirm}")
            
            if current >= high_30 * 0.98 and vol_confirm:
                return AgentSignal(self.name, 55, 80, "BULLISH_BREAKOUT", f"🔥 BULLISH BREAKOUT above ₹{high_30:.2f}!", {'level': round(high_30, 2)})
            elif current >= high_30 * 0.98:
                return AgentSignal(self.name, 28, 60, "POTENTIAL_BREAKOUT_UP", f"⚠️ Testing resistance at ₹{high_30:.2f}", {'level': round(high_30, 2)})
            elif current <= low_30 * 1.02 and vol_confirm:
                return AgentSignal(self.name, -55, 80, "BEARISH_BREAKOUT", f"⬇️ BEARISH BREAKDOWN below ₹{low_30:.2f}!", {'level': round(low_30, 2)})
            elif current <= low_30 * 1.02:
                return AgentSignal(self.name, -28, 60, "POTENTIAL_BREAKOUT_DOWN", f"⚠️ Testing support at ₹{low_30:.2f}", {'level': round(low_30, 2)})
            else:
                return AgentSignal(self.name, 0, 35, "NO_BREAKOUT", "Price in consolidation zone", {})
        except Exception as e:
            logger.error(f"Breakout error: {e}")
            return AgentSignal(self.name, 0, 0, "ERROR", str(e), {})


# ============================================
# MASTER AGGREGATOR - CONSENSUS LOGIC
# ============================================

class MasterInstitutionalAggregator:
    """Aggregates all 9 agents with real data"""
    
    def __init__(self):
        self.agents = {
            'fii': FFIMomentumAgent(),
            'execution': ExecutionBreakdownDetector(),
            'volume': VolumePatternRecognition(),
            'ifi': IFICalculator(),
            'accumulation': AccumulationDetector(),
            'liquidity': LiquidityDetector(),
            'smart_money': SmartMoneyTracker(),
            'block_orders': BlockOrderTracker(),
            'breakout': BreakoutDetector(),
        }
    
    def aggregate(self, fii_data: Dict = None, order_data: List[Dict] = None,
                  price_data: pd.Series = None, volume_data: pd.Series = None) -> AggregatedSignal:
        """Aggregate all 9 agents with REAL DATA"""
        
        signals = {}
        
        try:
            # Run all 9 agents
            if fii_data is not None and price_data is not None and len(price_data) > 0:
                signals['fii'] = self.agents['fii'].analyze(fii_data, price_data)
            
            if order_data is not None and price_data is not None and len(price_data) > 0:
                signals['execution'] = self.agents['execution'].analyze(order_data, price_data)
            
            if volume_data is not None and price_data is not None and len(volume_data) > 0 and len(price_data) > 0:
                signals['volume'] = self.agents['volume'].analyze(volume_data, price_data)
                signals['ifi'] = self.agents['ifi'].analyze(volume_data, price_data)
                signals['accumulation'] = self.agents['accumulation'].analyze(price_data, volume_data)
                signals['liquidity'] = self.agents['liquidity'].analyze(volume_data)
                signals['smart_money'] = self.agents['smart_money'].analyze(price_data, volume_data)
                signals['breakout'] = self.agents['breakout'].analyze(price_data, volume_data)
            
            if order_data is not None:
                signals['block_orders'] = self.agents['block_orders'].analyze(order_data)
            
            if not signals:
                return AggregatedSignal(50, "HOLD", 0, {}, "No data available", {})
            
            # ✅ CONSENSUS VOTING
            buy_signals = []
            sell_signals = []
            neutral_signals = []
            
            for agent_name, signal in signals.items():
                if signal is None:
                    continue
                
                action_upper = signal.action.upper()
                
                if "BUY" in action_upper:
                    buy_signals.append((agent_name, signal))
                elif "SELL" in action_upper:
                    sell_signals.append((agent_name, signal))
                else:
                    neutral_signals.append((agent_name, signal))
            
            total_agents = len(signals)
            buy_count = len(buy_signals)
            sell_count = len(sell_signals)
            neutral_count = len(neutral_signals)
            
            # Consensus rules
            if buy_count >= 7:
                recommendation = "STRONG BUY 🔥"
                confidence = min(90, 75 + (buy_count - 6) * 5)
            elif buy_count >= 5:
                recommendation = "BUY ✅"
                confidence = min(80, 60 + (buy_count - 4) * 5)
            elif sell_count >= 7:
                recommendation = "STRONG SELL ⚠️"
                confidence = min(90, 75 + (sell_count - 6) * 5)
            elif sell_count >= 5:
                recommendation = "SELL ⬇️"
                confidence = min(80, 60 + (sell_count - 4) * 5)
            elif buy_count >= 4 and sell_count <= 2:
                recommendation = "SLIGHT BUY"
                confidence = 55
            elif sell_count >= 4 and buy_count <= 2:
                recommendation = "SLIGHT SELL"
                confidence = 55
            else:
                recommendation = "HOLD ⏸️"
                confidence = max(30, 50 - abs(buy_count - sell_count) * 8)
            
            # Calculate final score
            if buy_count > sell_count:
                final_score = 50 + (buy_count / total_agents) * 40
            elif sell_count > buy_count:
                final_score = 50 - (sell_count / total_agents) * 40
            else:
                final_score = 50
            
            final_score = np.clip(final_score, 0, 100)
            
            reasoning = f"📊 Consensus: {buy_count}🟢 BUY | {sell_count}🔴 SELL | {neutral_count}⚪ NEUTRAL → {recommendation}"
            
            agent_votes = {name: signal.score for name, signal in signals.items()}
            
            logger.info(f"✅ Master Result: {recommendation} (Score: {final_score:.1f}/100, Confidence: {confidence:.0f}%)")
        
        except Exception as e:
            logger.error(f"Aggregation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return AggregatedSignal(50, "HOLD ⏸️", 0, {}, f"Error: {str(e)}", {})
        
        return AggregatedSignal(
            final_score=round(final_score, 1),
            recommendation=recommendation,
            confidence=round(confidence, 1),
            agent_votes=agent_votes,
            reasoning=reasoning,
            agent_breakdown=signals,
            timestamp=datetime.now()
        )


logger.info("✅ Institutional Agents v5.0 - REAL DATA INTEGRATION - NO CIRCULAR IMPORTS")

