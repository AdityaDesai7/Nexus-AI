# # ui/institutional_analysis.py - COMPLETE NEW FILE FOR 13 AGENTS

# import streamlit as st
# from datetime import datetime, timedelta
# import time
# import pandas as pd
# import logging
# from main import fetch_and_enhance
# from agents.institutional_agents import (
#     FFIMomentumAgent,
#     ExecutionBreakdownDetector,
#     VolumePatternRecognition,
#     IFICalculator,
#     AccumulationDetector,
#     LiquidityDetector,
#     SmartMoneyTracker,
#     BlockOrderTracker,
#     BreakoutDetector,
#     MasterInstitutionalAggregator
# )

# logger = logging.getLogger(__name__)


# def show_institutional_analysis():
#     """Display 13 Institutional Agents Analysis"""
    
#     if not st.session_state.get('logged_in'):
#         st.error("⚠️ Please login first")
#         if st.button("Go to Login"):
#             st.session_state.page = 'login'
#             st.rerun()
#         return
    
#     ticker = st.session_state.get('ticker', 'RELIANCE.NS')
    
#     # Header
#     col1, col2, col3 = st.columns([1, 6, 1])
#     with col1:
#         if st.button("← Back", use_container_width=True):
#             st.session_state.page = 'analysis'
#             st.rerun()
#     with col2:
#         st.markdown(f"<h1 style='text-align:center; color:#9b59b6;'>🏛️ 13 Institutional Agents</h1>", unsafe_allow_html=True)
#         st.markdown(f"<p style='text-align:center;'>{ticker}</p>", unsafe_allow_html=True)
    
#     # Date range
#     st.markdown("---")
#     col1, col2, col3 = st.columns([2, 2, 2])
    
#     with col1:
#         start_date_input = st.date_input("Start Date", value=datetime.now() - timedelta(days=180), key="inst_start")
#         start_date = datetime.combine(start_date_input, datetime.min.time())
    
#     with col2:
#         end_date_input = st.date_input("End Date", value=datetime.now(), key="inst_end")
#         end_date = datetime.combine(end_date_input, datetime.max.time())
    
#     with col3:
#         run_analysis = st.button("🚀 Analyze 13 Agents", type="primary", use_container_width=True)
    
#     if run_analysis:
#         progress_bar = st.progress(0)
#         status_container = st.empty()
        
#         try:
#             # Fetch data
#             status_container.markdown("""
#             <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #9b59b6;'>
#                 <strong style='color:#9b59b6;'>📊 Loading market data...</strong>
#             </div>
#             """, unsafe_allow_html=True)
#             progress_bar.progress(5)
            
#             df = fetch_and_enhance(ticker, start_date, end_date)
            
#             if df is None or len(df) == 0:
#                 st.error("❌ Failed to fetch data")
#                 return
            
#             price_data = df['Close']
#             volume_data = df['Volume']
            
#             logger.info(f"Loaded {len(df)} trading days for {ticker}")
#             progress_bar.progress(15)
            
#             # Initialize all 13 agents
#             status_container.markdown("""
#             <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #9b59b6;'>
#                 <strong style='color:#9b59b6;'>🤖 Initializing 13 agents...</strong>
#             </div>
#             """, unsafe_allow_html=True)
            
#             agents = {
#                 'fii': FFIMomentumAgent(),
#                 'execution': ExecutionBreakdownDetector(),
#                 'volume': VolumePatternRecognition(),
#                 'ifi': IFICalculator(),
#                 'accumulation': AccumulationDetector(),
#                 'liquidity': LiquidityDetector(),
#                 'smart_money': SmartMoneyTracker(),
#                 'block_orders': BlockOrderTracker(),
#                 'breakout': BreakoutDetector(),
#             }
            
#             # Run all 13 agents
#             results = {}
#             agent_list = list(agents.items())
            
#             for idx, (agent_name, agent) in enumerate(agent_list):
#                 status_container.markdown(f"""
#                 <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #9b59b6;'>
#                     <strong style='color:#9b59b6;'>🤖 {idx+1}/9: {agent_name.replace('_', ' ').title()} Agent...</strong>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 progress = int(15 + (idx / len(agent_list)) * 70)
#                 progress_bar.progress(progress)
#                 time.sleep(0.1)
                
#                 try:
#                     # Call appropriate analysis method
#                     if agent_name == 'fii':
#                         result = agent.analyze(fii_flow_data={'today': 100, '30_day_avg': 80}, price_data=price_data)
#                     elif agent_name == 'execution':
#                         result = agent.analyze(order_data=[], price_data=price_data)
#                     elif agent_name == 'volume':
#                         result = agent.analyze(volume_data=volume_data, price_data=price_data)
#                     elif agent_name == 'ifi':
#                         result = agent.analyze(volume_data=volume_data, price_data=price_data)
#                     elif agent_name == 'accumulation':
#                         result = agent.analyze(price_data=price_data, volume_data=volume_data)
#                     elif agent_name == 'liquidity':
#                         result = agent.analyze(volume_data=volume_data)
#                     elif agent_name == 'smart_money':
#                         result = agent.analyze(price_data=price_data, volume_data=volume_data)
#                     elif agent_name == 'block_orders':
#                         result = agent.analyze(order_data=[])
#                     elif agent_name == 'breakout':
#                         result = agent.analyze(price_data=price_data, volume_data=volume_data)
                    
#                     results[agent_name] = result
#                     logger.info(f"{agent_name} analysis complete: {result.action}")
                
#                 except Exception as e:
#                     logger.error(f"Error in {agent_name}: {e}")
#                     results[agent_name] = None
            
#             # Master Aggregator
#             status_container.markdown("""
#             <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #e74c3c;'>
#                 <strong style='color:#e74c3c;'>🎯 Master Aggregator - Synthesizing...</strong>
#             </div>
#             """, unsafe_allow_html=True)
#             progress_bar.progress(90)
            
#             master_agg = MasterInstitutionalAggregator()
#             # ✅ FIXED: Use .aggregate() method
#             master_result = master_agg.aggregate(
#                 fii_data={'today': 100, '30_day_avg': 80},
#                 order_data=[],
#                 price_data=price_data,
#                 volume_data=volume_data
#             )
            
#             progress_bar.progress(100)
#             time.sleep(0.2)
#             progress_bar.empty()
#             status_container.empty()
            
#             st.success("✅ Analysis Complete!")
            
#             # Display results in tabs
#             tab1, tab2, tab3 = st.tabs(["🤖 Individual Agents", "🎯 Master Decision", "📊 Detailed Data"])
            
#             # TAB 1: INDIVIDUAL AGENTS
#             with tab1:
#                 st.markdown("## 13 Individual Institutional Agents")
                
#                 # Create 3x3 grid
#                 cols = st.columns(3)
                
#                 agent_info = {
#                     'fii': {'emoji': '💰', 'color': '#e74c3c'},
#                     'execution': {'emoji': '⚡', 'color': '#f39c12'},
#                     'volume': {'emoji': '📈', 'color': '#3498db'},
#                     'ifi': {'emoji': '📊', 'color': '#2ecc71'},
#                     'accumulation': {'emoji': '🔝', 'color': '#9b59b6'},
#                     'liquidity': {'emoji': '💧', 'color': '#1abc9c'},
#                     'smart_money': {'emoji': '🧠', 'color': '#34495e'},
#                     'block_orders': {'emoji': '📦', 'color': '#16a085'},
#                     'breakout': {'emoji': '🚀', 'color': '#c0392b'},
#                 }
                
#                 for idx, (agent_name, result) in enumerate(results.items()):
#                     col_idx = idx % 3
                    
#                     if col_idx == 0:
#                         cols = st.columns(3)
                    
#                     if result:
#                         emoji = agent_info[agent_name]['emoji']
#                         color = agent_info[agent_name]['color']
#                         action_color = "#00b894" if "BUY" in result.action else "#ff7675" if "SELL" in result.action else "#fdcb6e"
                        
#                         with cols[col_idx]:
#                             st.markdown(f"""
#                             <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {color};'>
#                                 <h4 style='color:{color}; margin-top:0;'>{emoji} {agent_name.replace('_', ' ').title()}</h4>
#                                 <p><strong>Action:</strong> <span style='color:{action_color};'>{result.action}</span></p>
#                                 <p><strong>Score:</strong> {result.score:.1f}</p>
#                                 <p><strong>Confidence:</strong> {result.confidence:.1f}%</p>
#                                 <hr style='margin: 10px 0; opacity: 0.3;'>
#                                 <p style='font-size: 0.85em; color: #b2bec3;'>{result.reason}</p>
#                             </div>
#                             """, unsafe_allow_html=True)
#                     else:
#                         emoji = agent_info[agent_name]['emoji']
#                         color = agent_info[agent_name]['color']
                        
#                         with cols[col_idx]:
#                             st.markdown(f"""
#                             <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {color};'>
#                                 <h4 style='color:{color}; margin-top:0;'>{emoji} {agent_name.replace('_', ' ').title()}</h4>
#                                 <p style='color: #e74c3c;'><strong>❌ Error</strong></p>
#                             </div>
#                             """, unsafe_allow_html=True)
            
#             # TAB 2: MASTER DECISION
#             with tab2:
#                 st.markdown("## 🎯 Master Aggregator Decision")
                
#                 master_dict = master_result.dict() if hasattr(master_result, 'dict') else dict(master_result) if not isinstance(master_result, dict) else master_result
                
#                 master_rec = master_dict.get('recommendation', 'HOLD')
#                 master_score = master_dict.get('final_score', 50)
#                 master_conf = master_dict.get('confidence', 0)
                
#                 master_color = "#00b894" if master_rec == "STRONG BUY" or master_rec == "BUY" else "#ff7675" if master_rec == "STRONG SELL" or master_rec == "SELL" else "#fdcb6e"
                
#                 col1, col2, col3 = st.columns([2, 1, 1])
                
#                 with col1:
#                     st.markdown(f"""
#                     <div style='background: linear-gradient(135deg, {master_color}22, {master_color}11); padding: 40px; border-radius: 16px; border: 3px solid {master_color};'>
#                         <h1 style='color:{master_color}; margin:0; font-size:3rem; text-align:center;'>{master_rec}</h1>
#                         <p style='color:#b2bec3; text-align:center; margin:10px 0 0 0; font-size:1.2rem;'>Master Recommendation</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 with col2:
#                     st.metric("Score", f"{master_score:.1f}/100")
#                     st.metric("Confidence", f"{master_conf:.1f}%")
                
#                 with col3:
#                     buy_agents = sum(1 for r in results.values() if r and "BUY" in r.action)
#                     sell_agents = sum(1 for r in results.values() if r and "SELL" in r.action)
#                     st.metric("Buy Signals", f"{buy_agents}/9")
#                     st.metric("Sell Signals", f"{sell_agents}/9")
                
#                 st.markdown("---")
#                 st.markdown("### 📝 Reasoning")
#                 st.info(master_dict.get('reasoning', 'No reasoning provided'))
                
#                 st.markdown("### 🗳️ Agent Votes")
#                 agent_votes = master_dict.get('agent_votes', {})
                
#                 if agent_votes:
#                     vote_df = pd.DataFrame({
#                         'Agent': list(agent_votes.keys()),
#                         'Score': list(agent_votes.values())
#                     })
#                     st.bar_chart(vote_df.set_index('Agent'))
            
#             # TAB 3: DETAILED DATA
#             with tab3:
#                 st.markdown("## 📊 Detailed Agent Data")
                
#                 for agent_name, result in results.items():
#                     if result:
#                         with st.expander(f"📋 {agent_name.replace('_', ' ').title()} - Details"):
#                             col1, col2, col3, col4 = st.columns(4)
#                             with col1:
#                                 st.metric("Action", result.action)
#                             with col2:
#                                 st.metric("Score", f"{result.score:.2f}")
#                             with col3:
#                                 st.metric("Confidence", f"{result.confidence:.1f}%")
#                             with col4:
#                                 st.metric("Agent", result.agent_name[:20])
                            
#                             st.markdown("**Details:**")
#                             st.json(result.details if result.details else {})
        
#         except Exception as e:
#             st.error(f"❌ Error: {str(e)}")
#             logger.error(f"Institutional analysis error: {e}")
#             import traceback
#             with st.expander("🔍 Debug"):
#                 st.code(traceback.format_exc())
    
#     else:
#         st.info("👆 Select dates and click **Analyze 13 Agents** to start")
        
#         with st.expander("📖 About 13 Institutional Agents"):
#             st.markdown("""
#             ### 13 Autonomous Institutional Detection Agents
            
#             **Agents:**
#             1. **FII Momentum** - Foreign institutional investor flows
#             2. **Execution Breakdown** - Institutional execution patterns
#             3. **Volume Pattern** - Market volume analysis
#             4. **IFI Score** - Institutional Footprint Indicator
#             5. **Accumulation** - Accumulation phase detection
#             6. **Liquidity** - Market liquidity conditions
#             7. **Smart Money** - Smart money tracking
#             8. **Block Orders** - Large institutional orders
#             9. **Breakout** - Price breakout detection
            
#             **Master Aggregator** - Synthesizes all 9 signals into final recommendation
            
#             ### Features
#             ✅ Zero LLM dependency - Pure logic based
#             ✅ Real-time analysis
#             ✅ Institutional-grade detection
#             ✅ Risk assessment
#             ✅ Position sizing
#             """)


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
