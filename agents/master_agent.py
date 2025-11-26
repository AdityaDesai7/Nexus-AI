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

import logging
import json
import os
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

GROQ_API_KEY = "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"

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    logger.warning("Groq package not available. Install with: pip install groq")
    GROQ_AVAILABLE = False


class MasterAgent:
    """
    MINIMAL MASTER AGENT - Uses only Technical, Risk, Sentiment inputs
    Provides clean output with reasoning
    """

    def __init__(self, min_confidence: float = 60.0, groq_api_key: str = None):
        self.min_confidence = min_confidence
        self.client = None
        self.model = "llama-3.1-8b-instant"

        logger.info("Minimal MasterAgent initializing...")

        # === PASTE YOUR API KEY HERE ===
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY") or "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"
        
        if GROQ_AVAILABLE and self.api_key and self.api_key != "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":
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"✅ Groq client initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq client: {e}")
                self.client = None
        else:
            logger.warning("⚠️ Groq not available - using logic fallback")
            self.client = None

    def synthesize(
        self,
        ticker: str,
        technical_result: Dict,
        risk_metrics: Dict,
        sentiment_result: Dict = None,
        current_price: float = None,
    ) -> Dict[str, Any]:

        logger.info(f"MasterAgent analyzing {ticker}")

        # Extract only the essential data from the three agents
        analysis_data = self._extract_core_data(technical_result, risk_metrics, sentiment_result, current_price)

        if self.client:
            try:
                return self._synthesize_with_groq(ticker, analysis_data)
            except Exception as e:
                logger.error(f"Groq failed: {e}. Falling back to logic.")
                return self._synthesize_with_logic(ticker, analysis_data)
        else:
            return self._synthesize_with_logic(ticker, analysis_data)

    def _extract_core_data(self, technical, risk, sentiment, current_price):
        """Extract only essential data from the three core agents"""
        
        # Technical Data
        tech_data = technical.get("technical", technical) if technical else {}
        tech_info = {
            "action": tech_data.get("action", "HOLD"),
            "confidence": tech_data.get("confidence", 50),
            "rsi": tech_data.get("rsi", 50),
            "macd": tech_data.get("macd_hist", 0),
            "support": tech_data.get("support"),
            "resistance": tech_data.get("resistance"),
        }

        # Risk Data  
        risk_data = risk.get("risk", risk) if risk else {}
        risk_info = {
            "risk_level": risk_data.get("risk_level", "MEDIUM"),
            "stop_loss": risk_data.get("stop_loss_price"),
            "take_profit": risk_data.get("take_profit_price"),
        }

        # Sentiment Data
        sentiment_data = sentiment.get("sentiment", sentiment) if sentiment else {}
        sentiment_info = {
            "sentiment": sentiment_data.get("overall_sentiment", "NEUTRAL"),
            "confidence": sentiment_data.get("confidence", 50),
        }

        return {
            "technical": tech_info,
            "risk": risk_info, 
            "sentiment": sentiment_info,
            "current_price": current_price
        }

    def _synthesize_with_groq(self, ticker: str, data: Dict):
        """Use Groq to synthesize the three agent inputs"""
        
        prompt = self._build_minimal_prompt(ticker, data)
        
        logger.info("🤖 Calling Groq for reasoning...")

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            groq_response = json.loads(response.choices[0].message.content)
            return self._create_clean_output(ticker, data, groq_response)
            
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise

    def _build_minimal_prompt(self, ticker: str, data: Dict):
        """Build clean prompt with only the three essential inputs"""
        
        tech = data["technical"]
        risk = data["risk"]
        sentiment = data["sentiment"]
        current_price = data["current_price"]

        prompt = f"""
You are a trading analyst. Analyze {ticker} using these three inputs and provide a clear trading decision.

INPUTS:
1. TECHNICAL: {tech.get('action', 'HOLD')} signal ({tech.get('confidence', 50)}% confidence)
   - RSI: {tech.get('rsi', 50)}
   - Support: {tech.get('support', 'N/A')}
   - Resistance: {tech.get('resistance', 'N/A')}

2. RISK: {risk.get('risk_level', 'MEDIUM')} risk level
   - Stop Loss: {risk.get('stop_loss', 'N/A')}  
   - Take Profit: {risk.get('take_profit', 'N/A')}

3. SENTIMENT: {sentiment.get('sentiment', 'NEUTRAL')} ({sentiment.get('confidence', 50)}% confidence)

CURRENT PRICE: {current_price or 'N/A'}

Provide a clear JSON response with this structure:
{{
  "action": "BUY|SELL|HOLD",
  "confidence": 0-100,
  "reasoning": "Brief explanation synthesizing all three inputs",
  "risk_level": "LOW|MEDIUM|HIGH",
  "stop_loss": number,
  "take_profit": number,
  "risk_reward_ratio": number
}}

Keep the reasoning concise and focused on the three inputs provided.
"""

        return prompt

    def _create_clean_output(self, ticker: str, data: Dict, groq_response: Dict):
        """Create clean minimal output"""
        
        current_price = data["current_price"] or 0
        
        return {
            "ticker": ticker,
            "action": groq_response.get("action", "HOLD"),
            "confidence": groq_response.get("confidence", 50),
            "reasoning": groq_response.get("reasoning", "Analysis based on technical, risk, and sentiment inputs"),
            "risk_level": groq_response.get("risk_level", "MEDIUM"),
            "current_price": current_price,
            "entry_price": current_price,
            "stop_loss": groq_response.get("stop_loss"),
            "take_profit": groq_response.get("take_profit"),
            "risk_reward_ratio": groq_response.get("risk_reward_ratio"),
            "quantity": self._calculate_quantity(groq_response.get("action"), current_price),
            "timestamp": datetime.now().isoformat(),
            "ai_enhanced": True,
            "status": "SUCCESS"
        }

    def _synthesize_with_logic(self, ticker: str, data: Dict):
        """Simple logic-based fallback"""
        
        tech = data["technical"]
        risk = data["risk"]
        sentiment = data["sentiment"]
        current_price = data["current_price"] or 0

        # Simple decision logic
        tech_action = tech.get("action", "HOLD")
        tech_conf = tech.get("confidence", 50)
        sent_conf = sentiment.get("confidence", 50)
        
        avg_conf = (tech_conf + sent_conf) / 2
        
        if tech_action == "BUY" and avg_conf > 60:
            action = "BUY"
            confidence = min(85, avg_conf + 10)
        elif tech_action == "SELL" and avg_conf > 60:
            action = "SELL"
            confidence = min(80, avg_conf + 5)
        else:
            action = "HOLD"
            confidence = 50

        return {
            "ticker": ticker,
            "action": action,
            "confidence": confidence,
            "reasoning": f"Technical: {tech_action} ({tech_conf}%), Sentiment: {sentiment.get('sentiment', 'NEUTRAL')} ({sent_conf}%), Risk: {risk.get('risk_level', 'MEDIUM')}",
            "risk_level": risk.get("risk_level", "MEDIUM"),
            "current_price": current_price,
            "entry_price": current_price,
            "stop_loss": risk.get("stop_loss"),
            "take_profit": risk.get("take_profit"),
            "risk_reward_ratio": None,
            "quantity": self._calculate_quantity(action, current_price),
            "timestamp": datetime.now().isoformat(),
            "ai_enhanced": False,
            "status": "SUCCESS"
        }

    def _calculate_quantity(self, action: str, current_price: float) -> int:
        """Calculate simple position quantity"""
        if action == "HOLD" or current_price <= 0:
            return 0
        return max(1, int(1000 / current_price))  # Simple fixed position size