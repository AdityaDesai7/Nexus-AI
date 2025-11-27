# # agents/master_ai_agent.py
# import logging
# import json
# import os
# from typing import Dict, Any
# from datetime import datetime

# logger = logging.getLogger(__name__)

# # ===== API KEY =====
# GROQ_API_KEY = "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"

# try:
#     from groq import Groq
#     GROQ_AVAILABLE = True
# except ImportError:
#     logger.warning("Groq package not available. Install with: pip install groq")
#     GROQ_AVAILABLE = False


# class MasterAgent:
#     """
#     MASTER AGENT - Uses Groq API for intelligent reasoning
#     """

#     def __init__(self, min_confidence: float = 60.0, groq_api_key: str = None):
#         self.min_confidence = min_confidence
#         self.client = None
#         self.model = "llama-3.1-8b-instant"

#         logger.info("MasterAgent initializing...")

#         if GROQ_AVAILABLE:
#             try:
#                 api_key = groq_api_key or os.getenv("GROQ_API_KEY") or GROQ_API_KEY
#                 self.client = Groq(api_key=api_key)
#                 logger.info(f"Groq client initialized with model: {self.model}")
#             except Exception as e:
#                 logger.error(f"Failed to initialize Groq client: {e}")
#                 self.client = None
#         else:
#             logger.warning("Groq not available")

#         logger.info(f"MasterAgent LLM: {'ENABLED' if self.client else 'DISABLED'}")

#     # ============================================================
#     # MAIN SYNTHESIS
#     # ============================================================
#     def synthesize(
#         self,
#         ticker: str,
#         technical_result: Dict,
#         sentiment_result: Dict = None,
#         risk_metrics: Dict = None,
#         portfolio_metrics: Dict = None,
#         current_price: float = None,
#     ) -> Dict[str, Any]:

#         logger.info(f"MasterAgent analyzing {ticker}")

#         if self.client:
#             try:
#                 return self._synthesize_with_groq(
#                     ticker, technical_result, sentiment_result,
#                     risk_metrics, portfolio_metrics, current_price
#                 )
#             except Exception as e:
#                 logger.error(f"Groq failed: {e}. Falling back to logic.")

#         return self._synthesize_with_logic(
#             ticker, technical_result, sentiment_result,
#             risk_metrics, portfolio_metrics, current_price
#         )

#     # ============================================================
#     # GROQ SYNTHESIS
#     # ============================================================
#     def _synthesize_with_groq(
#         self, ticker, technical, sentiment, risk, portfolio, current_price
#     ):

#         groq_response = self._get_groq_analysis(
#             ticker, technical, sentiment, risk, portfolio, current_price
#         )

#         return self._create_exact_output(
#             groq_response, ticker, current_price, technical, risk, portfolio
#         )

#     def _get_groq_analysis(
#         self, ticker, technical, sentiment, risk, portfolio, current_price
#     ):

#         prompt = self._build_exact_prompt(
#             ticker, technical, sentiment, risk, portfolio, current_price
#         )

#         logger.info(f"Calling Groq for {ticker}...")

#         response = self.client.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model=self.model,
#             temperature=0.1,
#             max_tokens=1024,
#             response_format={"type": "json_object"}
#         )

#         return json.loads(response.choices[0].message.content)

#     # ============================================================
#     # FIXED — SAFE PROMPT (NO INVALID F-STRINGS)
#     # ============================================================
#     def _build_exact_prompt(self, ticker, technical, sentiment, risk, portfolio, current_price):

#         tech = technical.get("technical", technical)
#         action = tech.get("action", "HOLD")
#         tech_conf = tech.get("confidence", 50)
#         rsi = tech.get("rsi", 50)
#         macd_hist = tech.get("macd_hist", 0)
#         support = tech.get("support")
#         resistance = tech.get("resistance")
#         last_close = tech.get("latest_close", current_price)
#         current = current_price or last_close

#         # SENTIMENT
#         sent_text = "NEUTRAL"
#         sent_conf = 50
#         if sentiment:
#             d = sentiment.get("sentiment", sentiment)
#             sent_text = d.get("overall_sentiment", "NEUTRAL").upper()
#             sent_conf = d.get("overall_confidence", 50)

#         # RISK
#         risk_level = "MEDIUM"
#         stop_loss = None
#         take_profit = None
#         if risk:
#             r = risk.get("risk", risk)
#             risk_level = r.get("risk_level", "MEDIUM")
#             stop_loss = r.get("stop_loss_price")
#             take_profit = r.get("take_profit_price")

#         # SAFE FORMATTING
#         stop_loss_str = f"{stop_loss:.2f}" if stop_loss else "Not available"
#         take_profit_str = f"{take_profit:.2f}" if take_profit else "Not available"

#         prompt = f"""
# You are a professional trading analyst. Analyze {ticker} and provide a trading decision.

# Return ONLY a JSON object with EXACT fields.

# DATA:
# - Ticker: {ticker}
# - Current Price: {current:.2f}
# - Technical Action: {action} ({tech_conf}%)
# - RSI: {rsi:.2f}
# - MACD: {macd_hist:.4f}
# - Support: {support:.2f}
# - Resistance: {resistance:.2f}
# - Sentiment: {sent_text} ({sent_conf}%)
# - Risk Level: {risk_level}
# - Stop Loss: {stop_loss_str}
# - Take Profit: {take_profit_str}

# RETURN JSON ONLY.
# """
#         return prompt

#     # ============================================================
#     # FINAL OUTPUT BUILDER
#     # ============================================================
#     def _create_exact_output(self, groq, ticker, current_price, tech, risk, portfolio):

#         output = {
#             "ticker": groq.get("ticker", ticker),
#             "current_price": groq.get("current_price", current_price),
#             "action": groq.get("action", "HBUY"),
#             "confidence": groq.get("confidence", 50),
#             "reasoning": groq.get("reasoning", ""),
#             "entry_price": current_price,
#             "stop_loss": groq.get("stop_loss"),
#             "take_profit": groq.get("take_profit"),
#             "risk_reward_ratio": groq.get("risk_reward_ratio"),
#             "quantity": groq.get("quantity", 0),
#             "position_size": groq.get("position_size", 0.0),
#             "risk_level": groq.get("risk_level", "MEDIUM"),
#             "timestamp": datetime.now().isoformat(),
#             "status": "SUCCESS",
#             "ai_enhanced": True,
#         }

#         return self._validate_and_fix_output(output, current_price, risk, portfolio)

#     # ============================================================
#     # LOGIC FALLBACK
#     # ============================================================
#     def _synthesize_with_logic(
#         self, ticker, technical, sentiment, risk, portfolio, current_price
#     ):

#         tech = technical.get("technical", technical)
#         action = tech.get("action", "HOLD")
#         tech_conf = tech.get("confidence", 50)

#         sent_conf = 50
#         if sentiment:
#             d = sentiment.get("sentiment", sentiment)
#             sent_conf = d.get("overall_confidence", 50)

#         risk_level = "MEDIUM"
#         stop_loss = None
#         take_profit = None
#         if risk:
#             r = risk.get("risk", risk)
#             risk_level = r.get("risk_level", "MEDIUM")
#             stop_loss = r.get("stop_loss_price")
#             take_profit = r.get("take_profit_price")

#         avg_conf = (tech_conf + sent_conf) / 2

#         if action == "BUY" and tech_conf > 65:
#             final_action = "BUY"
#             final_conf = min(90, avg_conf + 10)
#         elif action == "SELL" and tech_conf > 65:
#             final_action = "SELL"
#             final_conf = min(85, avg_conf + 5)
#         else:
#             final_action = "HOLD"
#             final_conf = 50

#         # position
#         position_size = 0.05 if final_action != "HOLD" else 0
#         qty = int((position_size * 10000) / current_price) if final_action != "HOLD" else 0

#         return {
#             "ticker": ticker,
#             "current_price": current_price,
#             "action": final_action,
#             "confidence": final_conf,
#             "reasoning": "Logic fallback decision.",
#             "entry_price": current_price,
#             "stop_loss": stop_loss,
#             "take_profit": take_profit,
#             "risk_reward_ratio": None,
#             "quantity": qty,
#             "position_size": position_size,
#             "risk_level": risk_level,
#             "timestamp": datetime.now().isoformat(),
#             "status": "SUCCESS",
#             "ai_enhanced": False
#         }

#     # ============================================================
#     # OUTPUT CLEANER
#     # ============================================================
#     def _validate_and_fix_output(self, out, current_price, risk, portfolio):

#         out["entry_price"] = current_price

#         if out["confidence"] < 50:
#             out["confidence"] = 50
#         if out["confidence"] > 95:
#             out["confidence"] = 95

#         if out["action"] != "HOLD" and out["quantity"] <= 0:
#             out["quantity"] = max(10, int((0.05 * 10000) / current_price))

#         return out

# import logging
# import json
# import os
# from typing import Dict, Any
# from datetime import datetime

# logger = logging.getLogger(__name__)

# GROQ_API_KEY = "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"

# try:
#     from groq import Groq
#     GROQ_AVAILABLE = True
# except ImportError:
#     logger.warning("Groq package not available. Install with: pip install groq")
#     GROQ_AVAILABLE = False


# class MasterAgent:
#     """
#     MINIMAL MASTER AGENT - Uses only Technical, Risk, Sentiment inputs
#     Provides clean output with reasoning
#     """

#     def __init__(self, min_confidence: float = 60.0, groq_api_key: str = None):
#         self.min_confidence = min_confidence
#         self.client = None
#         self.model = "llama-3.1-8b-instant"

#         logger.info("Minimal MasterAgent initializing...")

#         # === PASTE YOUR API KEY HERE ===
#         self.api_key = groq_api_key or os.getenv("GROQ_API_KEY") or "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"
        
#         if GROQ_AVAILABLE and self.api_key and self.api_key != "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":
#             try:
#                 self.client = Groq(api_key=self.api_key)
#                 logger.info(f"✅ Groq client initialized with model: {self.model}")
#             except Exception as e:
#                 logger.error(f"❌ Failed to initialize Groq client: {e}")
#                 self.client = None
#         else:
#             logger.warning("⚠️ Groq not available - using logic fallback")
#             self.client = None

#     def synthesize(
#         self,
#         ticker: str,
#         technical_result: Dict,
#         risk_metrics: Dict,
#         sentiment_result: Dict = None,
#         current_price: float = None,
#     ) -> Dict[str, Any]:

#         logger.info(f"MasterAgent analyzing {ticker}")

#         # Extract only the essential data from the three agents
#         analysis_data = self._extract_core_data(technical_result, risk_metrics, sentiment_result, current_price)

#         if self.client:
#             try:
#                 return self._synthesize_with_groq(ticker, analysis_data)
#             except Exception as e:
#                 logger.error(f"Groq failed: {e}. Falling back to logic.")
#                 return self._synthesize_with_logic(ticker, analysis_data)
#         else:
#             return self._synthesize_with_logic(ticker, analysis_data)

#     def _extract_core_data(self, technical, risk, sentiment, current_price):
#         """Extract only essential data from the three core agents"""
        
#         # Technical Data
#         tech_data = technical.get("technical", technical) if technical else {}
#         tech_info = {
#             "action": tech_data.get("action", "HOLD"),
#             "confidence": tech_data.get("confidence", 50),
#             "rsi": tech_data.get("rsi", 50),
#             "macd": tech_data.get("macd_hist", 0),
#             "support": tech_data.get("support"),
#             "resistance": tech_data.get("resistance"),
#         }

#         # Risk Data  
#         risk_data = risk.get("risk", risk) if risk else {}
#         risk_info = {
#             "risk_level": risk_data.get("risk_level", "MEDIUM"),
#             "stop_loss": risk_data.get("stop_loss_price"),
#             "take_profit": risk_data.get("take_profit_price"),
#         }

#         # Sentiment Data
#         sentiment_data = sentiment.get("sentiment", sentiment) if sentiment else {}
#         sentiment_info = {
#             "sentiment": sentiment_data.get("overall_sentiment", "NEUTRAL"),
#             "confidence": sentiment_data.get("confidence", 50),
#         }

#         return {
#             "technical": tech_info,
#             "risk": risk_info, 
#             "sentiment": sentiment_info,
#             "current_price": current_price
#         }

#     def _synthesize_with_groq(self, ticker: str, data: Dict):
#         """Use Groq to synthesize the three agent inputs"""
        
#         prompt = self._build_minimal_prompt(ticker, data)
        
#         logger.info("🤖 Calling Groq for reasoning...")

#         try:
#             response = self.client.chat.completions.create(
#                 messages=[{"role": "user", "content": prompt}],
#                 model=self.model,
#                 temperature=0.1,
#                 max_tokens=800,
#                 response_format={"type": "json_object"}
#             )

#             groq_response = json.loads(response.choices[0].message.content)
#             return self._create_clean_output(ticker, data, groq_response)
            
#         except Exception as e:
#             logger.error(f"Groq API call failed: {e}")
#             raise

#     def _build_minimal_prompt(self, ticker: str, data: Dict):
#         """Build clean prompt with only the three essential inputs"""
        
#         tech = data["technical"]
#         risk = data["risk"]
#         sentiment = data["sentiment"]
#         current_price = data["current_price"]

#         prompt = f"""
# You are a trading analyst. Analyze {ticker} using these three inputs and provide a clear trading decision.

# INPUTS:
# 1. TECHNICAL: {tech.get('action', 'HOLD')} signal ({tech.get('confidence', 50)}% confidence)
#    - RSI: {tech.get('rsi', 50)}
#    - Support: {tech.get('support', 'N/A')}
#    - Resistance: {tech.get('resistance', 'N/A')}

# 2. RISK: {risk.get('risk_level', 'MEDIUM')} risk level
#    - Stop Loss: {risk.get('stop_loss', 'N/A')}  
#    - Take Profit: {risk.get('take_profit', 'N/A')}

# 3. SENTIMENT: {sentiment.get('sentiment', 'NEUTRAL')} ({sentiment.get('confidence', 50)}% confidence)

# CURRENT PRICE: {current_price or 'N/A'}

# Provide a clear JSON response with this structure:
# {{
#   "action": "BUY|SELL|HOLD",
#   "confidence": 0-100,
#   "reasoning": "Brief explanation synthesizing all three inputs",
#   "risk_level": "LOW|MEDIUM|HIGH",
#   "stop_loss": number,
#   "take_profit": number,
#   "risk_reward_ratio": number
# }}

# Keep the reasoning concise and focused on the three inputs provided.
# """

#         return prompt

#     def _create_clean_output(self, ticker: str, data: Dict, groq_response: Dict):
#         """Create clean minimal output"""
        
#         current_price = data["current_price"] or 0
        
#         return {
#             "ticker": ticker,
#             "action": groq_response.get("action", "HOLD"),
#             "confidence": groq_response.get("confidence", 50),
#             "reasoning": groq_response.get("reasoning", "Analysis based on technical, risk, and sentiment inputs"),
#             "risk_level": groq_response.get("risk_level", "MEDIUM"),
#             "current_price": current_price,
#             "entry_price": current_price,
#             "stop_loss": groq_response.get("stop_loss"),
#             "take_profit": groq_response.get("take_profit"),
#             "risk_reward_ratio": groq_response.get("risk_reward_ratio"),
#             "quantity": self._calculate_quantity(groq_response.get("action"), current_price),
#             "timestamp": datetime.now().isoformat(),
#             "ai_enhanced": True,
#             "status": "SUCCESS"
#         }

#     def _synthesize_with_logic(self, ticker: str, data: Dict):
#         """Simple logic-based fallback"""
        
#         tech = data["technical"]
#         risk = data["risk"]
#         sentiment = data["sentiment"]
#         current_price = data["current_price"] or 0

#         # Simple decision logic
#         tech_action = tech.get("action", "HOLD")
#         tech_conf = tech.get("confidence", 50)
#         sent_conf = sentiment.get("confidence", 50)
        
#         avg_conf = (tech_conf + sent_conf) / 2
        
#         if tech_action == "BUY" and avg_conf > 60:
#             action = "BUY"
#             confidence = min(85, avg_conf + 10)
#         elif tech_action == "SELL" and avg_conf > 60:
#             action = "SELL"
#             confidence = min(80, avg_conf + 5)
#         else:
#             action = "HOLD"
#             confidence = 50

#         return {
#             "ticker": ticker,
#             "action": action,
#             "confidence": confidence,
#             "reasoning": f"Technical: {tech_action} ({tech_conf}%), Sentiment: {sentiment.get('sentiment', 'NEUTRAL')} ({sent_conf}%), Risk: {risk.get('risk_level', 'MEDIUM')}",
#             "risk_level": risk.get("risk_level", "MEDIUM"),
#             "current_price": current_price,
#             "entry_price": current_price,
#             "stop_loss": risk.get("stop_loss"),
#             "take_profit": risk.get("take_profit"),
#             "risk_reward_ratio": None,
#             "quantity": self._calculate_quantity(action, current_price),
#             "timestamp": datetime.now().isoformat(),
#             "ai_enhanced": False,
#             "status": "SUCCESS"
#         }

#     def _calculate_quantity(self, action: str, current_price: float) -> int:
#         """Calculate simple position quantity"""
#         if action == "HOLD" or current_price <= 0:
#             return 0
#         return max(1, int(1000 / current_price))  # Simple fixed position size


# import logging
# import json
# import os
# from typing import Dict, Any
# from datetime import datetime

# logger = logging.getLogger(__name__)

# try:
#     from groq import Groq
#     GROQ_AVAILABLE = True
# except ImportError:
#     GROQ_AVAILABLE = False


# class MasterAgent:
#     """
#     TRUE DECISION MASTER AGENT
#     - Uses rule-based quant scoring
#     - Uses LLM (Groq) reasoning when available
#     - Produces consistent BUY/SELL/HOLD + realistic confidence
#     """

#     def __init__(self, groq_api_key=None):
#         self.model = "llama-3.1-8b-instant"
#         self.api_key = (
#             groq_api_key or
#             os.getenv("GROQ_API_KEY") or
#             ""
#         )

#         self.client = None
#         if GROQ_AVAILABLE and self.api_key.strip():
#             try:
#                 self.client = Groq(api_key=self.api_key)
#                 logger.info("Groq client initialized.")
#             except Exception as e:
#                 logger.error(f"Groq init failed: {e}")
#                 self.client = None

#     # =======================================================================
#     # MAIN ENTRY
#     # =======================================================================
#     def synthesize(self, ticker: str, technical_result, risk_metrics, sentiment_result, current_price):
#         data = self._extract(technical_result, risk_metrics, sentiment_result, current_price)

#         # 1) Always calculate quant score
#         logic_output = self._quant_decision(ticker, data)

#         # 2) If LLM available → enhance logic output
#         if self.client:
#             try:
#                 llm_output = self._llm_decision(ticker, data)
#                 return self._merge_outputs(logic_output, llm_output)
#             except Exception:
#                 return logic_output

#         return logic_output

#     # =======================================================================
#     # DATA EXTRACTION
#     # =======================================================================
#     def _extract(self, technical, risk, sentiment, current_price):
#         tech = technical or {}
#         risk = risk or {}
#         sent = sentiment or {}

#         return {
#             "technical": {
#                 "signal": tech.get("action", "HOLD"),
#                 "confidence": float(tech.get("confidence", 50)),
#                 "rsi": tech.get("rsi", None),
#                 "macd": tech.get("macd", None)
#             },
#             "sentiment": {
#                 "signal": sent.get("sentiment", sent.get("overall_sentiment", "NEUTRAL")),
#                 "confidence": float(sent.get("confidence", 50))
#             },
#             "risk": {
#                 "risk_level": risk.get("risk_level", "MEDIUM"),
#                 "stop_loss": risk.get("stop_loss_price"),
#                 "take_profit": risk.get("take_profit_price")
#             },
#             "price": current_price or 0
#         }

#     # =======================================================================
#     # QUANT DECISION ENGINE
#     # =======================================================================
#     def _quant_decision(self, ticker, data):
#         tech = data["technical"]
#         sent = data["sentiment"]
#         risk = data["risk"]
#         price = data["price"]

#         # ---- Technical score ----
#         def tech_score(sig, conf):
#             sig = sig.upper()
#             if sig in ["BUY", "BULLISH"]:  return 65 + (conf - 50) * 0.7
#             if sig in ["SELL", "BEARISH"]: return 35 - (conf - 50) * 0.7
#             return 45 + (conf - 50) * 0.2

#         # ---- Sentiment score ----
#         def sent_score(sig, conf):
#             sig = sig.upper()
#             if sig in ["POSITIVE", "BULLISH"]: return 60 + (conf - 50) * 0.5
#             if sig in ["NEGATIVE", "BEARISH"]: return 40 - (conf - 50) * 0.5
#             return 50

#         # ---- Risk penalty ----
#         def risk_penalty(level):
#             m = {
#                 "VERY_LOW": +5,
#                 "LOW": +0,
#                 "MEDIUM": -5,
#                 "HIGH": -15,
#                 "VERY_HIGH": -25
#             }
#             return m.get(level.upper(), -5)

#         ts = tech_score(tech["signal"], tech["confidence"])
#         ss = sent_score(sent["signal"], sent["confidence"])
#         rp = risk_penalty(risk["risk_level"])

#         final_conf = (
#             0.45 * ts +
#             0.30 * ss +
#             0.15 * rp
#         )
#         final_conf = max(5, min(95, round(final_conf)))

#         # ---- Decision Rule ----
#         if final_conf >= 65:
#             action = "BUY"
#         elif final_conf <= 40:
#             action = "SELL"
#         else:
#             action = "HOLD"

#         return {
#             "ticker": ticker,
#             "action": action,
#             "confidence": final_conf,
#             "risk_level": risk["risk_level"],
#             "current_price": price,
#             "entry_price": price,
#             "stop_loss": risk["stop_loss"],
#             "take_profit": risk["take_profit"],
#             "reasoning": f"Quant synthesis: tech={ts:.1f}, sentiment={ss:.1f}, risk_penalty={rp}",
#             "risk_reward_ratio": None,
#             "quantity": self._qty(action, price),
#             "timestamp": datetime.now().isoformat(),
#             "ai_enhanced": False,
#             "status": "SUCCESS"
#         }

#     # =======================================================================
#     # LLM DECISION
#     # =======================================================================
#     def _llm_decision(self, ticker, data):
#         prompt = f"""
# You are a financial decision engine.

# Technical: {data['technical']}
# Sentiment: {data['sentiment']}
# Risk: {data['risk']}
# Price: {data['price']}

# Return JSON:
# {{
#   "action": "BUY|SELL|HOLD",
#   "confidence": 0-100,
#   "reasoning": "text"
# }}
# """

#         response = self.client.chat.completions.create(
#             model=self.model,
#             temperature=0.1,
#             response_format={"type": "json_object"},
#             messages=[{"role": "user", "content": prompt}]
#         )

#         return json.loads(response.choices[0].message.content)

#     # =======================================================================
#     # MERGE LOGIC + LLM
#     # =======================================================================
#     def _merge_outputs(self, logic, llm):
#         # LLM can adjust confidence ±10
#         final_conf = round(min(95, max(5, logic["confidence"] + (llm.get("confidence", 50) - 50) * 0.2)))

#         # LLM only overrides action if conviction high
#         final_action = logic["action"]
#         if llm.get("confidence", 50) >= 70:
#             final_action = llm.get("action", final_action)

#         logic.update({
#             "action": final_action,
#             "confidence": final_conf,
#             "llm_reasoning": llm.get("reasoning", "")
#         })

#         logic["ai_enhanced"] = True
#         return logic

#     # =======================================================================
#     # POSITION SIZE
#     # =======================================================================
#     def _qty(self, action, price):
#         if action == "HOLD" or price <= 0:
#             return 0
#         return max(1, int(1000 / price))




import logging
import json
import os
from typing import Dict, Any
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class MasterAgent:
    """
    ENHANCED MASTER AGENT - REALISTIC QUANT TRADING DECISIONS
    - Dynamic confidence scoring with market context
    - Realistic position sizing based on volatility
    - Improved price target calculation
    - Better LLM integration for nuanced reasoning
    """

    def __init__(self, groq_api_key=None):
        self.model = "llama-3.1-8b-instant"
        self.api_key = (
            groq_api_key or
            os.getenv("GROQ_API_KEY") or
            ""
        )

        self.client = None
        if GROQ_AVAILABLE and self.api_key.strip():
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq client initialized.")
            except Exception as e:
                logger.error(f"Groq init failed: {e}")
                self.client = None

    # =======================================================================
    # MAIN ENTRY - ENHANCED
    # =======================================================================
    def synthesize(self, ticker: str, technical_result, risk_metrics, sentiment_result, current_price):
        data = self._extract(technical_result, risk_metrics, sentiment_result, current_price)

        # Enhanced quant decision with market context
        logic_output = self._enhanced_quant_decision(ticker, data)

        # Only use LLM if we have meaningful signals
        if self.client and self._should_use_llm(data):
            try:
                llm_output = self._enhanced_llm_decision(ticker, data, logic_output)
                final_output = self._smart_merge_outputs(logic_output, llm_output, data)
                return final_output
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")
                return logic_output

        return logic_output

    def _should_use_llm(self, data):
        """Only use LLM when we have meaningful signals to enhance"""
        tech_conf = data["technical"]["confidence"]
        sent_conf = data["sentiment"]["confidence"]
        return tech_conf > 40 or sent_conf > 40

    # =======================================================================
    # ENHANCED QUANT DECISION ENGINE
    # =======================================================================
    def _enhanced_quant_decision(self, ticker, data):
        tech = data["technical"]
        sent = data["sentiment"]
        risk = data["risk"]
        price = data["price"]

        # ---- Enhanced Technical scoring with RSI/MACD context ----
        def enhanced_tech_score(sig, conf, rsi, macd):
            base_score = 50  # Start from neutral
            
            # Signal strength
            sig = sig.upper()
            if sig in ["BUY", "BULLISH", "STRONG_BUY"]:  
                base_score += 25 + (conf - 50) * 0.5
            elif sig in ["SELL", "BEARISH", "STRONG_SELL"]: 
                base_score -= 25 - (conf - 50) * 0.5
            else:  # HOLD, NEUTRAL
                base_score += (conf - 50) * 0.3

            # RSI adjustment
            if rsi is not None:
                if rsi < 30:  # Oversold - bullish
                    base_score += 10
                elif rsi > 70:  # Overbought - bearish
                    base_score -= 10
                elif 45 <= rsi <= 55:  # Neutral RSI
                    base_score -= 5  # Slight penalty for lack of momentum

            # MACD adjustment (simplified)
            if macd is not None:
                if macd > 0:  # Bullish MACD
                    base_score += 8
                else:  # Bearish MACD
                    base_score -= 8

            return max(10, min(90, base_score))

        # ---- Enhanced Sentiment scoring ----
        def enhanced_sent_score(sig, conf):
            sig = sig.upper()
            if sig in ["POSITIVE", "BULLISH", "VERY_POSITIVE"]: 
                return 60 + (conf - 50) * 0.6
            elif sig in ["NEGATIVE", "BEARISH", "VERY_NEGATIVE"]: 
                return 40 - (conf - 50) * 0.6
            elif sig in ["MIXED", "CONFLICTING"]:
                return 45 + (conf - 50) * 0.2
            return 50  # NEUTRAL

        # ---- Dynamic Risk adjustment ----
        def dynamic_risk_adjustment(level, volatility=None):
            adjustments = {
                "VERY_LOW": +12,
                "LOW": +5,
                "MEDIUM": -3,
                "HIGH": -15,
                "VERY_HIGH": -25,
                "EXTREME": -35
            }
            return adjustments.get(level.upper(), -5)

        # Calculate scores
        ts = enhanced_tech_score(tech["signal"], tech["confidence"], tech.get("rsi"), tech.get("macd"))
        ss = enhanced_sent_score(sent["signal"], sent["confidence"])
        ra = dynamic_risk_adjustment(risk["risk_level"])

        # Dynamic weights based on confidence levels
        tech_weight = 0.50 if tech["confidence"] > 40 else 0.30
        sent_weight = 0.35 if sent["confidence"] > 40 else 0.25
        risk_weight = 0.15

        final_conf = (
            tech_weight * ts +
            sent_weight * ss +
            risk_weight * ra
        )
        
        # Apply confidence curve (more extreme moves for high/low confidence)
        if final_conf > 70:
            final_conf = 70 + (final_conf - 70) * 1.2  # Amplify high confidence
        elif final_conf < 30:
            final_conf = 30 - (30 - final_conf) * 1.2  # Amplify low confidence
            
        final_conf = max(5, min(95, round(final_conf)))

        # ---- Enhanced Decision Rule with conviction levels ----
        if final_conf >= 68:
            action = "STRONG_BUY"
        elif final_conf >= 58:
            action = "BUY"
        elif final_conf <= 32:
            action = "STRONG_SELL"
        elif final_conf <= 42:
            action = "SELL"
        else:
            action = "HOLD"

        # Calculate realistic price targets
        stop_loss, take_profit = self._calculate_price_targets(action, price, tech.get("rsi"), risk.get("volatility"))

        return {
            "ticker": ticker,
            "action": action,
            "confidence": final_conf,
            "risk_level": risk["risk_level"],
            "current_price": price,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": self._calculate_rr_ratio(price, stop_loss, take_profit, action),
            "quantity": self._dynamic_position_size(action, price, final_conf, risk["risk_level"]),
            "reasoning": f"Enhanced quant: tech={ts:.1f} (w{tech_weight}), sentiment={ss:.1f} (w{sent_weight}), risk_adj={ra}",
            "timestamp": datetime.now().isoformat(),
            "ai_enhanced": False,
            "status": "SUCCESS",
            "volatility": risk.get("volatility", "MEDIUM"),
            "momentum": "BULLISH" if ts > 60 else "BEARISH" if ts < 40 else "NEUTRAL"
        }

    def _calculate_price_targets(self, action, current_price, rsi, volatility):
        """Calculate realistic stop loss and take profit levels"""
        if current_price <= 0:
            return None, None

        # Base volatility adjustments
        vol_multiplier = {
            "LOW": 0.02,      # 2%
            "MEDIUM": 0.035,  # 3.5%
            "HIGH": 0.05,     # 5%
            "VERY_HIGH": 0.07 # 7%
        }.get(volatility, 0.035)

        # RSI-based adjustments
        rsi_multiplier = 1.0
        if rsi is not None:
            if rsi < 25 or rsi > 75:  # Extreme RSI - wider stops
                rsi_multiplier = 1.3
            elif 30 <= rsi <= 70:     # Normal RSI - standard stops
                rsi_multiplier = 1.0
            else:                     # Moderate extremes
                rsi_multiplier = 1.15

        final_multiplier = vol_multiplier * rsi_multiplier

        if action in ["STRONG_BUY", "BUY"]:
            stop_loss = current_price * (1 - final_multiplier)
            take_profit = current_price * (1 + final_multiplier * 1.8)  # 1.8:1 R/R
        elif action in ["STRONG_SELL", "SELL"]:
            stop_loss = current_price * (1 + final_multiplier)
            take_profit = current_price * (1 - final_multiplier * 1.8)
        else:  # HOLD
            return None, None

        return round(stop_loss, 2), round(take_profit, 2)

    def _calculate_rr_ratio(self, price, stop_loss, take_profit, action):
        """Calculate realistic risk/reward ratio"""
        if not all([price, stop_loss, take_profit]):
            return None
            
        try:
            if action in ["STRONG_BUY", "BUY"]:
                risk = price - stop_loss
                reward = take_profit - price
            elif action in ["STRONG_SELL", "SELL"]:
                risk = stop_loss - price
                reward = price - take_profit
            else:
                return None
                
            if risk > 0:
                return round(reward / risk, 2)
        except:
            pass
        return None

    def _dynamic_position_size(self, action, price, confidence, risk_level):
        """Calculate position size based on confidence, risk, and price"""
        if action == "HOLD" or price <= 0:
            return 0

        # Base investment amount
        base_investment = 5000  # ₹5000 base
        
        # Confidence multiplier (0.3 to 1.5)
        conf_multiplier = 0.3 + (confidence / 100) * 1.2
        
        # Risk adjustment
        risk_multiplier = {
            "VERY_LOW": 1.4,
            "LOW": 1.2,
            "MEDIUM": 1.0,
            "HIGH": 0.6,
            "VERY_HIGH": 0.3
        }.get(risk_level, 0.5)
        
        # Action strength multiplier
        action_multiplier = {
            "STRONG_BUY": 1.3,
            "BUY": 1.0,
            "STRONG_SELL": 1.3,
            "SELL": 1.0,
            "HOLD": 0
        }.get(action, 0)
        
        if action_multiplier == 0:
            return 0

        total_investment = base_investment * conf_multiplier * risk_multiplier * action_multiplier
        quantity = max(1, int(total_investment / price))
        
        # Round to nearest 10 for lots
        quantity = (quantity // 10) * 10
        return max(10, quantity)  # Minimum 10 shares

    # =======================================================================
    # ENHANCED LLM DECISION
    # =======================================================================
    def _enhanced_llm_decision(self, ticker, data, quant_output):
        """Enhanced LLM reasoning with quant context"""
        prompt = f"""
As a quantitative trading analyst, analyze {ticker} and provide a nuanced trading decision.

CONTEXT:
- Current Price: ${data['price']:.2f}
- Technical: {data['technical']['signal']} ({data['technical']['confidence']}% confidence)
- Sentiment: {data['sentiment']['signal']} ({data['sentiment']['confidence']}% confidence) 
- Risk Level: {data['risk']['risk_level']}
- Quant Recommendation: {quant_output['action']} ({quant_output['confidence']}% confidence)

Provide a nuanced analysis considering:
1. Market context and sector trends
2. Strength of technical vs sentiment signals
3. Risk-adjusted return potential
4. Short-term vs medium-term outlook

Return JSON with realistic, nuanced values:
{{
  "action": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
  "confidence": 1-100,
  "reasoning": "detailed market analysis",
  "time_horizon": "VERY_SHORT|SHORT|MEDIUM|LONG",
  "conviction": "LOW|MEDIUM|HIGH|VERY_HIGH"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,  # Slightly higher for nuanced reasoning
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return None

    def _smart_merge_outputs(self, quant, llm, data):
        """Smart merging of quant and LLM outputs"""
        if not llm:
            return quant

        # Use LLM to adjust confidence within bounds
        llm_conf = llm.get("confidence", 50)
        quant_conf = quant["confidence"]
        
        # Only adjust if LLM has strong conviction
        llm_conviction = llm.get("conviction", "LOW")
        conviction_multiplier = {
            "VERY_HIGH": 0.3,
            "HIGH": 0.2,
            "MEDIUM": 0.1,
            "LOW": 0.05
        }.get(llm_conviction, 0.05)
        
        confidence_adjustment = (llm_conf - 50) * conviction_multiplier
        final_conf = max(5, min(95, round(quant_conf + confidence_adjustment)))

        # Only override action if LLM has high conviction and significant difference
        final_action = quant["action"]
        if (llm_conviction in ["HIGH", "VERY_HIGH"] and 
            abs(llm_conf - quant_conf) > 15):
            llm_action = llm.get("action")
            if llm_action and llm_action != quant["action"]:
                final_action = llm_action

        quant.update({
            "action": final_action,
            "confidence": final_conf,
            "llm_reasoning": llm.get("reasoning", ""),
            "time_horizon": llm.get("time_horizon", "SHORT"),
            "conviction_level": llm_conviction,
            "ai_enhanced": True
        })

        return quant

    # =======================================================================
    # DATA EXTRACTION (Keep original)
    # =======================================================================
    def _extract(self, technical, risk, sentiment, current_price):
        tech = technical or {}
        risk = risk or {}
        sent = sentiment or {}

        return {
            "technical": {
                "signal": tech.get("action", "HOLD"),
                "confidence": float(tech.get("confidence", 50)),
                "rsi": tech.get("rsi", None),
                "macd": tech.get("macd", None)
            },
            "sentiment": {
                "signal": sent.get("sentiment", sent.get("overall_sentiment", "NEUTRAL")),
                "confidence": float(sent.get("confidence", 50))
            },
            "risk": {
                "risk_level": risk.get("risk_level", "MEDIUM"),
                "stop_loss": risk.get("stop_loss_price"),
                "take_profit": risk.get("take_profit_price"),
                "volatility": risk.get("volatility", "MEDIUM")
            },
            "price": current_price or 0
        }
