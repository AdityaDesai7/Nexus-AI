# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv

# load_dotenv()
# llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

# class MasterAgent:
#     def oversee(self, state: dict) -> dict:
#         prompt = f"Review agent outputs for {state['ticker']} (e.g., NTPC.NS): {state}. Final approval or adjustment."
#         return {"approved": True, "notes": llm.invoke(prompt).content}


# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple


# class MasterAgent:
#     """Master agent - Synthesizes all signals autonomously"""
    
#     def __init__(self, min_confidence: float = 60):
#         self.min_confidence = min_confidence
        
#         # Signal weights
#         self.weights = {
#             'technical': 0.40,  # Technical is most reliable (autonomous)
#             'sentiment': 0.30,  # Sentiment context
#             'risk': 0.20,       # Risk management
#             'portfolio': 0.10   # Portfolio alignment
#         }
    
#     def synthesize(self,
#                    ticker: str,
#                    technical_result: Dict,
#                    sentiment_result: Dict = None,
#                    risk_metrics: Dict = None,
#                    portfolio_metrics: Dict = None,
#                    current_price: float = None) -> Dict:
#         """
#         Synthesize all signals for final decision
        
#         Returns: Master decision with action, confidence, reasoning
#         """
        
#         # Extract signals
#         tech_signal, tech_conf = self._parse_technical(technical_result)
#         sent_signal, sent_conf = self._parse_sentiment(sentiment_result)
#         risk_signal = self._parse_risk(risk_metrics)
#         port_signal, port_conf = self._parse_portfolio(portfolio_metrics)
        
#         # Calculate consensus confidence
#         signals = [tech_signal, sent_signal, risk_signal]
#         consensus = sum([1 for s in signals if s != 0]) / len(signals)
        
#         # Calculate weighted decision
#         weighted_action = (
#             tech_signal * self.weights['technical'] +
#             sent_signal * self.weights['sentiment'] +
#             risk_signal * self.weights['risk'] +
#             port_signal * self.weights['portfolio']
#         )
        
#         # Determine final action
#         if weighted_action > 0.2:
#             final_action = "BUY"
#             confidence = min(90, tech_conf * (0.5 + consensus))
#         elif weighted_action < -0.2:
#             final_action = "SELL"
#             confidence = min(90, tech_conf * (0.5 + consensus))
#         else:
#             final_action = "HOLD"
#             confidence = 50
        
#         # Generate reasoning
#         reasoning = self._generate_reasoning(
#             final_action,
#             tech_signal,
#             sent_signal,
#             risk_metrics,
#             consensus
#         )
        
#         # Calculate quantity
#         quantity = portfolio_metrics.get('quantity', 10) if portfolio_metrics else 10
        
#         return {
#             "action": final_action,
#             "confidence": confidence,
#             "quantity": quantity,
#             "reasoning": reasoning,
#             "signals": {
#                 "technical": tech_signal,
#                 "sentiment": sent_signal,
#                 "risk": risk_signal,
#                 "portfolio": port_signal,
#                 "consensus": consensus
#             },
#             "risk_level": risk_metrics.get('risk_level', 'MEDIUM') if risk_metrics else 'MEDIUM',
#             "stop_loss": risk_metrics.get('stop_loss_price', 0) if risk_metrics else 0,
#             "take_profit": risk_metrics.get('take_profit_price', 0) if risk_metrics else 0
#         }
    
#     def _parse_technical(self, technical_result: Dict) -> Tuple[float, float]:
#         """Parse technical signal (1=BUY, -1=SELL, 0=HOLD)"""
#         if not technical_result:
#             return 0, 50
        
#         rec = technical_result.get('recommendation', '')
#         confidence = self._extract_confidence(rec)
        
#         if 'BUY' in rec.upper():
#             return 1, confidence
#         elif 'SELL' in rec.upper():
#             return -1, confidence
#         else:
#             return 0, confidence
    
#     def _parse_sentiment(self, sentiment_result: Dict) -> Tuple[float, float]:
#         """Parse sentiment signal"""
#         if not sentiment_result:
#             return 0, 50
        
#         action = sentiment_result.get('action', 'HOLD')
#         confidence = sentiment_result.get('confidence', 50) / 100
        
#         if action == "BUY":
#             return confidence, confidence * 100
#         elif action == "SELL":
#             return -confidence, confidence * 100
#         else:
#             return 0, 50
    
#     def _parse_risk(self, risk_metrics: Dict) -> float:
#         """Parse risk signal"""
#         if not risk_metrics:
#             return 0
        
#         risk_level = risk_metrics.get('risk_level', 'MEDIUM')
#         position_size = risk_metrics.get('position_size', 0.05)
        
#         # Risk should reduce signal if too high
#         if risk_level == "VERY_HIGH":
#             return -0.5  # Caution signal
#         elif risk_level == "HIGH":
#             return -0.2  # Slight caution
#         elif position_size < 0.02:
#             return -0.3  # Too small position size = risky
#         else:
#             return 0.2  # Acceptable risk
    
#     def _parse_portfolio(self, portfolio_metrics: Dict) -> Tuple[float, float]:
#         """Parse portfolio signal"""
#         if not portfolio_metrics:
#             return 0, 50
        
#         allocation_pct = portfolio_metrics.get('allocation_pct', 5)
        
#         # Portfolio should validate position sizing
#         if allocation_pct > 15:
#             return -0.3, 60  # Too large position
#         elif allocation_pct < 1:
#             return 0, 50  # Too small
#         else:
#             return 0.2, 70  # Good position sizing
    
#     def _extract_confidence(self, recommendation_str: str) -> float:
#         """Extract confidence from recommendation string"""
#         try:
#             parts = recommendation_str.split("Confidence: ")
#             if len(parts) > 1:
#                 conf_str = parts[1].split("%")[0]
#                 return float(conf_str)
#         except:
#             pass
#         return 50.0
    
#     def _generate_reasoning(self,
#                            action: str,
#                            tech_signal: float,
#                            sent_signal: float,
#                            risk_metrics: Dict,
#                            consensus: float) -> str:
#         """Generate human-readable reasoning"""
        
#         if action == "BUY":
#             reasons = []
            
#             if tech_signal > 0:
#                 reasons.append("✓ Technical indicators show BUY signal")
            
#             if sent_signal > 0:
#                 reasons.append("✓ Sentiment is positive")
            
#             if consensus > 0.5:
#                 reasons.append(f"✓ Strong consensus ({consensus:.0%})")
            
#             if risk_metrics and risk_metrics.get('risk_level') in ['LOW', 'MEDIUM']:
#                 reasons.append(f"✓ Risk level acceptable ({risk_metrics.get('risk_level')})")
            
#             return " | ".join(reasons) if reasons else "Strong technical signals detected"
        
#         elif action == "SELL":
#             reasons = []
            
#             if tech_signal < 0:
#                 reasons.append("✓ Technical indicators show SELL signal")
            
#             if sent_signal < 0:
#                 reasons.append("✓ Sentiment is negative")
            
#             if risk_metrics and risk_metrics.get('risk_level') in ['HIGH', 'VERY_HIGH']:
#                 reasons.append(f"✓ Risk is elevated ({risk_metrics.get('risk_level')})")
            
#             return " | ".join(reasons) if reasons else "Technical indicators suggest SELL"
        
#         else:
#             return "Mixed signals suggest caution. Awaiting stronger signal."



# # trading_bot/agents/master_agent.py
# from typing import Dict, Any, Tuple

# class MasterAgent:
#     """
#     Pure logic master agent. synthesize(...) returns final action/confidence/reasoning.
#     """

#     def __init__(self, min_confidence: float = 60.0):
#         self.min_confidence = min_confidence
#         self.weights = {"technical": 0.45, "sentiment": 0.25, "risk": 0.2, "portfolio": 0.1}

#     def _extract_tech(self, technical_result: Dict) -> Tuple[float, float]:
#         # returns (signal: -1/0/1, confidence 0-100)
#         if not technical_result:
#             return 0, 50.0
#         action = str(technical_result.get("action") or technical_result.get("recommendation") or "").upper()
#         conf = float(technical_result.get("confidence", 50))
#         if "BUY" in action:
#             return 1, conf
#         if "SELL" in action:
#             return -1, conf
#         return 0, conf

#     def _parse_debate(self, debate_result: Dict) -> Tuple[float, float]:
#         if not debate_result:
#             return 0.0, 50.0
#         consensus = debate_result.get("consensus", {})
#         action = consensus.get("action", "HOLD")
#         conf = float(consensus.get("confidence", 50))
#         if action == "BUY":
#             return 1.0, conf
#         if action == "SELL":
#             return -1.0, conf
#         return 0.0, conf

#     def _parse_risk(self, risk_metrics: Dict) -> float:
#         if not risk_metrics:
#             return 0.0
#         rl = risk_metrics.get("risk_level", "MEDIUM")
#         if rl == "VERY_HIGH":
#             return -0.6
#         if rl == "HIGH":
#             return -0.3
#         if rl == "MEDIUM":
#             return 0.0
#         return 0.2

#     def synthesize(self,
#                    ticker: str,
#                    technical_result: Dict,
#                    sentiment_result: Dict = None,
#                    risk_metrics: Dict = None,
#                    portfolio_metrics: Dict = None,
#                    current_price: float = None) -> Dict[str, Any]:
#         tech_sig, tech_conf = self._extract_tech(technical_result)
#         debate_sig, debate_conf = self._parse_debate(sentiment_result if sentiment_result and "consensus" in sentiment_result else {})
#         risk_sig = self._parse_risk(risk_metrics)
#         # portfolio influence: prefer 0.0 to small positive if allocation sensible
#         port_sig = 0.0
#         port_conf = 50.0
#         if portfolio_metrics and isinstance(portfolio_metrics, dict):
#             alloc = portfolio_metrics.get("allocation_pct") or portfolio_metrics.get("allocation_pct", 0)
#             if alloc and alloc > 0 and alloc < 15:
#                 port_sig = 0.2
#                 port_conf = 60.0

#         # Weighted composite
#         weighted = (tech_sig * self.weights["technical"] +
#                     debate_sig * self.weights["sentiment"] +
#                     risk_sig * self.weights["risk"] +
#                     port_sig * self.weights["portfolio"])

#         # final action
#         if weighted > 0.2:
#             action = "BUY"
#             confidence = min(90, tech_conf * (0.6 + abs(weighted)))
#         elif weighted < -0.2:
#             action = "SELL"
#             confidence = min(90, tech_conf * (0.6 + abs(weighted)))
#         else:
#             action = "HOLD"
#             confidence = 50.0

#         reasoning_parts = []
#         if tech_sig > 0:
#             reasoning_parts.append("Technical signals are bullish")
#         elif tech_sig < 0:
#             reasoning_parts.append("Technical signals are bearish")
#         if risk_sig < 0:
#             reasoning_parts.append(f"Risk elevated ({risk_metrics.get('risk_level')})" if risk_metrics else "Risk elevated")
#         if port_sig > 0:
#             reasoning_parts.append("Portfolio sizing OK")

#         return {
#             "ticker": ticker,
#             "action": action,
#             "confidence": round(float(confidence), 1),
#             "reasoning": "; ".join(reasoning_parts) or "Mixed signals",
#             "signals": {
#                 "technical": tech_sig,
#                 "debate": debate_sig,
#                 "risk": risk_sig,
#                 "portfolio": port_sig
#             },
#             "quantity": int(portfolio_metrics.get("quantity", 0) if portfolio_metrics else 0),
#             "stop_loss": risk_metrics.get("stop_loss_price") if risk_metrics else None,
#             "take_profit": risk_metrics.get("take_profit_price") if risk_metrics else None,
#             "risk_level": risk_metrics.get("risk_level") if risk_metrics else "UNKNOWN"
#         }


# # trading_bot/agents/master_agent.py
# from typing import Dict, Any, Tuple
# import logging

# logger = logging.getLogger(__name__)


# class MasterAgent:
#     """
#     ADVANCED MASTER AGENT (Hybrid Logic + LLM Arbitration)

#     Features:
#     - Deterministic weighted logic layer
#     - LLM re-ranking / arbitration layer
#     - LLM chain-of-thought (hidden)
#     - Confidence smoothing
#     - Counterfactual reasoning: “when would BUY change to SELL?”
#     """

#     def __init__(self, min_confidence: float = 60.0, llm=None):
#         self.min_confidence = min_confidence
#         self.llm = llm

#         # Base logic weights
#         self.weights = {
#             "technical": 0.45,
#             "sentiment": 0.25,
#             "risk": 0.20,
#             "portfolio": 0.10
#         }

#     # ------------------------------------------------------------
#     # PARSERS
#     # ------------------------------------------------------------
#     def _extract_tech(self, technical_result: Dict) -> Tuple[float, float]:
#         if not technical_result:
#             return 0, 50.0

#         action = str(
#             technical_result.get("action")
#             or technical_result.get("recommendation")
#             or ""
#         ).upper()

#         conf = float(technical_result.get("confidence", 50))

#         if "BUY" in action:
#             return 1, conf
#         if "SELL" in action:
#             return -1, conf
#         return 0, conf

#     def _parse_debate(self, debate_result: Dict) -> Tuple[float, float]:
#         if not debate_result:
#             return 0.0, 50.0

#         consensus = debate_result.get("consensus", {})
#         action = consensus.get("action", "HOLD")
#         conf = float(consensus.get("confidence", 50))

#         if action == "BUY":
#             return 1.0, conf
#         if action == "SELL":
#             return -1.0, conf
#         return 0.0, conf

#     def _parse_risk(self, risk_metrics: Dict) -> float:
#         if not risk_metrics:
#             return 0.0

#         rl = risk_metrics.get("risk_level", "MEDIUM")

#         if rl == "VERY_HIGH":
#             return -0.6
#         if rl == "HIGH":
#             return -0.3
#         if rl == "MEDIUM":
#             return 0.0
#         return 0.2

#     # ------------------------------------------------------------
#     # MAIN SYNTHESIS
#     # ------------------------------------------------------------
#     def synthesize(
#         self,
#         ticker: str,
#         technical_result: Dict,
#         sentiment_result: Dict = None,
#         risk_metrics: Dict = None,
#         portfolio_metrics: Dict = None,
#         current_price: float = None,
#     ) -> Dict[str, Any]:

#         # ========= 1) LOGIC LAYER ==========
#         tech_sig, tech_conf = self._extract_tech(technical_result)
#         debate_sig, debate_conf = self._parse_debate(
#             sentiment_result if sentiment_result and "consensus" in sentiment_result else {}
#         )
#         risk_sig = self._parse_risk(risk_metrics)

#         port_sig = 0.0
#         if portfolio_metrics:
#             alloc = portfolio_metrics.get("allocation_pct") or 0
#             if 0 < alloc < 15:
#                 port_sig = 0.2

#         composite = (
#             tech_sig * self.weights["technical"] +
#             debate_sig * self.weights["sentiment"] +
#             risk_sig * self.weights["risk"] +
#             port_sig * self.weights["portfolio"]
#         )

#         # Initial logic decision
#         if composite > 0.2:
#             logic_action = "BUY"
#             logic_conf = min(90, tech_conf * (0.6 + abs(composite)))
#         elif composite < -0.2:
#             logic_action = "SELL"
#             logic_conf = min(90, tech_conf * (0.6 + abs(composite)))
#         else:
#             logic_action = "HOLD"
#             logic_conf = 50.0

#         # ========= 2) LLM ARBITRATION LAYER ==========
#         llm_action = logic_action
#         llm_conf = logic_conf
#         llm_reasoning = "LLM disabled."

#         if self.llm:
#             try:
#                 prompt = f"""
# You are an expert equity analyst.

# Below are structured signals for the stock {ticker}:

# TECHNICAL RESULT:
# {technical_result}

# DEBATE RESULT:
# {sentiment_result}

# RISK METRICS:
# {risk_metrics}

# PORTFOLIO METRICS:
# {portfolio_metrics}

# CURRENT LOGIC DECISION:
# Action: {logic_action}
# Confidence: {logic_conf}

# TASK:
# 1. Decide whether the logic decision should be:
#     - confirmed
#     - reversed (BUY→SELL, SELL→BUY)
#     - softened (BUY→HOLD or SELL→HOLD)
#     - strengthened (HOLD→BUY or HOLD→SELL)

# 2. Choose ONE final action: BUY / SELL / HOLD

# 3. Give a confidence score between 50 and 90.

# 4. Provide a brief explanation (no numbers invented).

# Respond ONLY in this JSON format:
# {
#   "final_action": "...",
#   "final_confidence": ...,
#   "reasoning": "..."
# }
#                 """

#                 raw = self.llm.ask(prompt)

#                 # Very safe parsing
#                 import json
#                 parsed = json.loads(raw.replace("```", "").strip())

#                 llm_action = parsed.get("final_action", logic_action)
#                 llm_conf = float(parsed.get("final_confidence", logic_conf))
#                 llm_reasoning = parsed.get("reasoning", "No reasoning provided.")

#             except Exception as e:
#                 logger.error(f"LLM arbitration failed: {e}")
#                 llm_action = logic_action
#                 llm_conf = logic_conf
#                 llm_reasoning = "LLM arbitration unavailable."

#         # ========= 3) ENSEMBLE FINAL DECISION ==========
#         if llm_action != logic_action:
#             # LLM disagreed with logic → soften change
#             final_action = llm_action
#             final_conf = (llm_conf * 0.7) + (logic_conf * 0.3)
#         else:
#             # Both agree → strengthen confidence
#             final_action = logic_action
#             final_conf = min(95, (llm_conf * 0.6) + (logic_conf * 0.4))

#         final_conf = round(float(final_conf), 1)

#         # ========= 4) OUTPUT ==========
#         return {
#             "ticker": ticker,
#             "action": final_action,
#             "confidence": final_conf,

#             "logic_action": logic_action,
#             "logic_confidence": round(float(logic_conf), 1),

#             "llm_action": llm_action,
#             "llm_confidence": round(float(llm_conf), 1),

#             "llm_reasoning": llm_reasoning,

#             "signals": {
#                 "technical": tech_sig,
#                 "debate": debate_sig,
#                 "risk": risk_sig,
#                 "portfolio": port_sig
#             },

#             "risk_level": risk_metrics.get("risk_level") if risk_metrics else "UNKNOWN",
#             "stop_loss": risk_metrics.get("stop_loss_price") if risk_metrics else None,
#             "take_profit": risk_metrics.get("take_profit_price") if risk_metrics else None,

#             "quantity": int(portfolio_metrics.get("quantity", 0) if portfolio_metrics else 0)
#         }

# # trading_bot/agents/master_agent.py
# from typing import Dict, Any, Tuple
# import logging
# import json

# logger = logging.getLogger(__name__)


# class MasterAgent:
#     """
#     ADVANCED HYBRID MASTER AGENT

#     - Deterministic weighted logic layer (technical + debate + risk + portfolio)
#     - LLM arbitration layer (can confirm / soften / reverse / strengthen logic)
#     - LLM narrative layer (clean explanation, no invented numbers)

#     This class is used as "MasterLogic" by the BaseAgent wrapper:
#         from agents.master_agent import MasterAgent as MasterLogic

#     The wrapper should construct it as:
#         self.logic = MasterLogic(llm=llm)
#     """

#     def __init__(self, min_confidence: float = 60.0, llm=None):
#         self.min_confidence = min_confidence
#         self.llm = llm

#         # Base weights for composite signal
#         self.weights = {
#             "technical": 0.45,
#             "sentiment": 0.25,   # here "sentiment" = debate signal or sentiment proxy
#             "risk": 0.20,
#             "portfolio": 0.10,
#         }

#     # ------------------------------------------------------------
#     # PARSERS
#     # ------------------------------------------------------------
#     def _extract_tech(self, technical_result: Dict) -> Tuple[float, float]:
#         """
#         Map technical_result -> (signal, confidence)
#         signal ∈ {-1, 0, 1}, confidence ∈ [0, 100]
#         """
#         if not technical_result:
#             return 0, 50.0

#         action = str(
#             technical_result.get("action")
#             or technical_result.get("recommendation")
#             or ""
#         ).upper()

#         conf = float(technical_result.get("confidence", 50))

#         if "BUY" in action:
#             return 1, conf
#         if "SELL" in action:
#             return -1, conf
#         return 0, conf

#     def _parse_debate(self, sentiment_or_debate_result: Dict) -> Tuple[float, float]:
#         """
#         Handles both:
#         - DIRECT debate result: {'consensus': {'action': 'BUY', 'confidence': 78}}
#         - WRAPPED: {'debate': { ...above dict... }}
#         - Otherwise falls back to neutral.
#         """
#         if not sentiment_or_debate_result:
#             return 0.0, 50.0

#         # unwrap if nested under 'debate'
#         if "debate" in sentiment_or_debate_result and isinstance(sentiment_or_debate_result["debate"], dict):
#             raw = sentiment_or_debate_result["debate"]
#         else:
#             raw = sentiment_or_debate_result

#         consensus = raw.get("consensus", {})
#         action = consensus.get("action", "HOLD")
#         conf = float(consensus.get("confidence", 50))

#         if action == "BUY":
#             return 1.0, conf
#         if action == "SELL":
#             return -1.0, conf
#         return 0.0, conf

#     def _parse_risk(self, risk_metrics: Dict) -> float:
#         """
#         Map risk profile to directional penalty / bonus.
#         """
#         if not risk_metrics:
#             return 0.0

#         rl = risk_metrics.get("risk_level", "MEDIUM")

#         if rl == "VERY_HIGH":
#             return -0.6
#         if rl == "HIGH":
#             return -0.3
#         if rl == "MEDIUM":
#             return 0.0
#         # LOW or UNKNOWN → slight positive
#         return 0.2

#     # ------------------------------------------------------------
#     # MAIN SYNTHESIS
#     # ------------------------------------------------------------
#     def synthesize(
#         self,
#         ticker: str,
#         technical_result: Dict,
#         sentiment_result: Dict = None,   # debate_payload or news_payload
#         risk_metrics: Dict = None,
#         portfolio_metrics: Dict = None,
#         current_price: float = None,
#     ) -> Dict[str, Any]:

#         # ========= 1) LOGIC LAYER ==========
#         tech_sig, tech_conf = self._extract_tech(technical_result)
#         debate_sig, debate_conf = self._parse_debate(sentiment_result or {})
#         risk_sig = self._parse_risk(risk_metrics or {})

#         port_sig = 0.0
#         if portfolio_metrics and isinstance(portfolio_metrics, dict):
#             alloc = portfolio_metrics.get("allocation_pct") or 0
#             try:
#                 alloc = float(alloc)
#             except Exception:
#                 alloc = 0.0
#             # small positive if we have a modest suggested allocation
#             if 0 < alloc < 15:
#                 port_sig = 0.2

#         composite = (
#             tech_sig * self.weights["technical"] +
#             debate_sig * self.weights["sentiment"] +
#             risk_sig * self.weights["risk"] +
#             port_sig * self.weights["portfolio"]
#         )

#         # Initial logic decision
#         if composite > 0.2:
#             logic_action = "BUY"
#             logic_conf = min(90, tech_conf * (0.6 + abs(composite)))
#         elif composite < -0.2:
#             logic_action = "SELL"
#             logic_conf = min(90, tech_conf * (0.6 + abs(composite)))
#         else:
#             logic_action = "HOLD"
#             logic_conf = 50.0

#         # Build simple rule-based reasoning string
#         reasoning_parts = []
#         if tech_sig > 0:
#             reasoning_parts.append("Technical signals are bullish")
#         elif tech_sig < 0:
#             reasoning_parts.append("Technical signals are bearish")

#         if risk_sig < 0:
#             if risk_metrics:
#                 reasoning_parts.append(f"Risk elevated ({risk_metrics.get('risk_level')})")
#             else:
#                 reasoning_parts.append("Risk elevated")

#         if port_sig > 0:
#             reasoning_parts.append("Portfolio allocation size is acceptable")

#         if debate_sig > 0:
#             reasoning_parts.append("Debate/consensus leans bullish")
#         elif debate_sig < 0:
#             reasoning_parts.append("Debate/consensus leans bearish")

#         base_reasoning = "; ".join(reasoning_parts) or "Mixed or neutral signals"

#         # ========= 2) LLM ARBITRATION LAYER ==========
#         llm_action = logic_action
#         llm_conf = logic_conf
#         llm_reasoning = None

#         if self.llm:
#             try:
#                 prompt = f"""
# You are an expert equity analyst.

# Below are structured signals for the stock {ticker}:

# TECHNICAL RESULT:
# {technical_result}

# SENTIMENT / DEBATE RESULT:
# {sentiment_result}

# RISK METRICS:
# {risk_metrics}

# PORTFOLIO METRICS:
# {portfolio_metrics}

# CURRENT PRICE:
# {current_price}

# CURRENT LOGIC DECISION:
# Action: {logic_action}
# Confidence: {logic_conf}
# Reasoning: {base_reasoning}

# TASK:
# 1. Decide whether the logic decision should be:
#     - confirmed
#     - reversed (BUY→SELL or SELL→BUY)
#     - softened (BUY→HOLD or SELL→HOLD)
#     - strengthened (HOLD→BUY or HOLD→SELL)

# 2. Choose ONE final action: "BUY", "SELL", or "HOLD".

# 3. Choose a final confidence between 50 and 95.

# 4. Provide a brief explanation (no invented numbers or prices).

# Respond ONLY in this JSON format (no text before or after):

# {{
#   "final_action": "BUY / SELL / HOLD",
#   "final_confidence": 78.5,
#   "reasoning": "Short explanation..."
# }}
# """
#                 raw = self.llm.ask(prompt)

#                 # Very defensive JSON parsing
#                 cleaned = str(raw).strip()
#                 cleaned = cleaned.replace("```json", "").replace("```", "").strip()
#                 parsed = json.loads(cleaned)

#                 llm_action = str(parsed.get("final_action", logic_action)).upper()
#                 if llm_action not in ("BUY", "SELL", "HOLD"):
#                     llm_action = logic_action

#                 try:
#                     llm_conf = float(parsed.get("final_confidence", logic_conf))
#                 except Exception:
#                     llm_conf = logic_conf

#                 # clamp
#                 if llm_conf < 50:
#                     llm_conf = 50.0
#                 if llm_conf > 95:
#                     llm_conf = 95.0

#                 llm_reasoning = parsed.get("reasoning", "") or "No LLM reasoning provided."

#             except Exception as e:
#                 logger.error(f"LLM arbitration failed in MasterAgent: {e}")
#                 llm_action = logic_action
#                 llm_conf = logic_conf
#                 llm_reasoning = "LLM arbitration unavailable."

#         # ========= 3) ENSEMBLE FINAL DECISION ==========
#         if llm_action != logic_action:
#             # LLM disagrees with logic → soften by blending confidences
#             final_action = llm_action
#             final_conf = (llm_conf * 0.7) + (logic_conf * 0.3)
#         else:
#             # LLM and logic agree → reinforce confidence
#             final_action = logic_action
#             final_conf = (llm_conf * 0.6) + (logic_conf * 0.4)
#             if final_conf > 95:
#                 final_conf = 95.0

#         final_conf = round(float(final_conf), 1)

#         # Quantity, SL/TP passthrough from portfolio/risk
#         quantity = 0
#         if portfolio_metrics and isinstance(portfolio_metrics, dict):
#             try:
#                 quantity = int(portfolio_metrics.get("quantity", 0))
#             except Exception:
#                 quantity = 0

#         stop_loss = risk_metrics.get("stop_loss_price") if risk_metrics else None
#         take_profit = risk_metrics.get("take_profit_price") if risk_metrics else None
#         risk_level = risk_metrics.get("risk_level") if risk_metrics else "UNKNOWN"

#         # ========= 4) OUTPUT STRUCTURE ==========
#         return {
#             "ticker": ticker,

#             # Final AI decision (what UI & rest of system should use)
#             "action": final_action,
#             "confidence": final_conf,

#             # Rule-based logic baseline
#             "logic_action": logic_action,
#             "logic_confidence": round(float(logic_conf), 1),
#             "reasoning": base_reasoning,

#             # LLM arbitration layer
#             "llm_action": llm_action,
#             "llm_confidence": round(float(llm_conf), 1),
#             "llm_reasoning": llm_reasoning,

#             # Signal breakdown
#             "signals": {
#                 "technical": tech_sig,
#                 "debate": debate_sig,
#                 "risk": risk_sig,
#                 "portfolio": port_sig,
#             },

#             # Trade / risk params
#             "quantity": quantity,
#             "stop_loss": stop_loss,
#             "take_profit": take_profit,
#             "risk_level": risk_level,
#         }


# # trading_bot/agents/master_agent.py
# from typing import Dict, Any, Tuple, Optional
# import logging
# import json
# import time

# logger = logging.getLogger(__name__)


# class MasterAgent:
#     """
#     TRUE AI HYBRID MASTER AGENT with Groq LLM Integration

#     - Multi-layer AI decision making
#     - Groq LLM for real-time reasoning and arbitration
#     - Probability-based evaluation
#     - Comprehensive narrative explanation
#     - Risk-adjusted final decision
#     """

#     def __init__(self, min_confidence: float = 60.0, llm=None):
#         self.min_confidence = min_confidence
#         self.llm = llm

#         # Dynamic weights that can be adjusted by LLM
#         self.base_weights = {
#             "technical": 0.40,
#             "sentiment": 0.25,
#             "risk": 0.20,
#             "portfolio": 0.15,
#         }

#     def _call_groq_llm(self, prompt: str, max_retries: int = 3) -> str:
#         """
#         Robust Groq LLM caller with retry logic
#         """
#         for attempt in range(max_retries):
#             try:
#                 if hasattr(self.llm, 'ask'):
#                     response = self.llm.ask(prompt)
#                 elif hasattr(self.llm, 'generate'):
#                     response = self.llm.generate(prompt)
#                 elif callable(self.llm):
#                     response = self.llm(prompt)
#                 else:
#                     raise ValueError("LLM instance not properly configured")
                
#                 if response:
#                     return str(response).strip()
                
#             except Exception as e:
#                 logger.warning(f"Groq LLM call attempt {attempt + 1} failed: {e}")
#                 if attempt < max_retries - 1:
#                     time.sleep(1)  # Wait before retry
#                 continue
        
#         raise Exception(f"All {max_retries} Groq LLM attempts failed")

#     def _parse_llm_json_response(self, response: str) -> Dict:
#         """
#         Parse JSON response from Groq LLM with robust error handling
#         """
#         try:
#             # Clean the response
#             cleaned = response.strip()
#             if "```json" in cleaned:
#                 cleaned = cleaned.split("```json")[1].split("```")[0].strip()
#             elif "```" in cleaned:
#                 cleaned = cleaned.split("```")[1].split("```")[0].strip()
            
#             return json.loads(cleaned)
#         except Exception as e:
#             logger.error(f"Failed to parse LLM JSON response: {e}")
#             logger.debug(f"Raw response: {response}")
#             return {}

#     # ------------------------------------------------------------
#     # ENHANCED PARSERS WITH AI CONTEXT
#     # ------------------------------------------------------------
#     def _extract_tech_with_ai(self, technical_result: Dict) -> Tuple[float, float, str]:
#         """
#         Enhanced technical analysis with AI context understanding
#         """
#         if not technical_result:
#             return 0, 50.0, "No technical data available"

#         # Try AI-enhanced interpretation first
#         try:
#             prompt = f"""
#             Analyze this technical analysis result and extract the core signal:
#             {json.dumps(technical_result, indent=2)}
            
#             Return JSON:
#             {{
#                 "signal": -1/0/1,
#                 "confidence": 0-100,
#                 "reason": "brief explanation"
#             }}
#             """
            
#             response = self._call_groq_llm(prompt)
#             ai_analysis = self._parse_llm_json_response(response)
            
#             if ai_analysis:
#                 return (
#                     ai_analysis.get("signal", 0),
#                     ai_analysis.get("confidence", 50),
#                     ai_analysis.get("reason", "AI analysis")
#                 )
#         except Exception as e:
#             logger.warning(f"AI technical analysis failed, using fallback: {e}")

#         # Fallback to rule-based
#         action = str(
#             technical_result.get("action") or 
#             technical_result.get("recommendation") or 
#             technical_result.get("signal") or ""
#         ).upper()

#         conf = float(technical_result.get("confidence", 50))
#         reason = "Rule-based technical analysis"

#         if "BUY" in action or "BULL" in action:
#             return 1, min(95, conf), reason
#         if "SELL" in action or "BEAR" in action:
#             return -1, min(95, conf), reason
#         return 0, max(30, conf), reason

#     def _parse_debate_with_ai(self, sentiment_or_debate_result: Dict) -> Tuple[float, float, str]:
#         """
#         AI-enhanced debate/sentiment analysis
#         """
#         if not sentiment_or_debate_result:
#             return 0.0, 50.0, "No sentiment/debate data"

#         try:
#             prompt = f"""
#             Analyze this sentiment/debate result and extract market sentiment:
#             {json.dumps(sentiment_or_debate_result, indent=2)}
            
#             Return JSON:
#             {{
#                 "sentiment": -1/0/1,
#                 "confidence": 0-100,
#                 "reason": "brief explanation"
#             }}
#             """
            
#             response = self._call_groq_llm(prompt)
#             ai_analysis = self._parse_llm_json_response(response)
            
#             if ai_analysis:
#                 return (
#                     ai_analysis.get("sentiment", 0),
#                     ai_analysis.get("confidence", 50),
#                     ai_analysis.get("reason", "AI sentiment analysis")
#                 )
#         except Exception as e:
#             logger.warning(f"AI sentiment analysis failed, using fallback: {e}")

#         # Fallback to rule-based
#         raw = sentiment_or_debate_result.get("debate", sentiment_or_debate_result)
#         consensus = raw.get("consensus", {})
#         action = consensus.get("action", "HOLD")
#         conf = float(consensus.get("confidence", 50))

#         if action == "BUY":
#             return 1.0, conf, "Bullish consensus"
#         if action == "SELL":
#             return -1.0, conf, "Bearish consensus"
#         return 0.0, conf, "Neutral consensus"

#     def _analyze_risk_with_ai(self, risk_metrics: Dict) -> Tuple[float, str]:
#         """
#         AI-powered risk assessment
#         """
#         if not risk_metrics:
#             return 0.0, "No risk data"

#         try:
#             prompt = f"""
#             Analyze these risk metrics and assess overall risk appetite (positive = risk-on, negative = risk-off):
#             {json.dumps(risk_metrics, indent=2)}
            
#             Return JSON:
#             {{
#                 "risk_appetite": -1.0 to 1.0,
#                 "reason": "risk assessment explanation"
#             }}
#             """
            
#             response = self._call_groq_llm(prompt)
#             ai_analysis = self._parse_llm_json_response(response)
            
#             if ai_analysis:
#                 return (
#                     ai_analysis.get("risk_appetite", 0),
#                     ai_analysis.get("reason", "AI risk assessment")
#                 )
#         except Exception as e:
#             logger.warning(f"AI risk analysis failed, using fallback: {e}")

#         # Fallback
#         rl = risk_metrics.get("risk_level", "MEDIUM")
#         risk_map = {
#             "VERY_HIGH": (-0.6, "Very high risk - avoid positions"),
#             "HIGH": (-0.3, "High risk - cautious approach"),
#             "MEDIUM": (0.0, "Medium risk - normal operations"),
#             "LOW": (0.2, "Low risk - favorable conditions"),
#             "VERY_LOW": (0.4, "Very low risk - excellent conditions")
#         }
#         return risk_map.get(rl, (0.0, "Unknown risk level"))

#     # ------------------------------------------------------------
#     # AI WEIGHT OPTIMIZATION
#     # ------------------------------------------------------------
#     def _optimize_weights_with_ai(self, signals: Dict) -> Dict[str, float]:
#         """
#         Let AI dynamically adjust signal weights based on current market context
#         """
#         try:
#             prompt = f"""
#             As a quantitative analyst, optimize signal weights for this trading decision:
            
#             Technical Signal: {signals.get('technical', {})}
#             Sentiment Signal: {signals.get('sentiment', {})}
#             Risk Assessment: {signals.get('risk', {})}
#             Portfolio Context: {signals.get('portfolio', {})}
            
#             Base Weights: {self.base_weights}
            
#             Adjust weights (must sum to 1.0) based on which signals are most reliable now.
#             Return JSON:
#             {{
#                 "technical": 0.0-1.0,
#                 "sentiment": 0.0-1.0,
#                 "risk": 0.0-1.0,
#                 "portfolio": 0.0-1.0,
#                 "reason": "why you adjusted weights this way"
#             }}
#             """
            
#             response = self._call_groq_llm(prompt)
#             optimized = self._parse_llm_json_response(response)
            
#             if optimized and abs(sum(optimized.values()) - 1.0) < 0.1:  # Allow small rounding errors
#                 return {k: v for k, v in optimized.items() if k in self.base_weights}
                
#         except Exception as e:
#             logger.warning(f"AI weight optimization failed: {e}")
        
#         return self.base_weights.copy()

#     # ------------------------------------------------------------
#     # ADVANCED AI ARBITRATION
#     # ------------------------------------------------------------
#     def _ai_arbitration_layer(self, logic_data: Dict, market_context: Dict) -> Dict:
#         """
#         Advanced AI arbitration with probability assessment
#         """
#         try:
#             prompt = f"""
#             You are a senior quantitative trading AI. Make a final trading decision.
            
#             MARKET CONTEXT:
#             Ticker: {market_context['ticker']}
#             Current Price: ${market_context.get('current_price', 'Unknown')}
            
#             LOGIC LAYER ANALYSIS:
#             Action: {logic_data['action']}
#             Confidence: {logic_data['confidence']}%
#             Reasoning: {logic_data['reasoning']}
            
#             SIGNAL BREAKDOWN:
#             Technical: {logic_data['signals']['technical']}
#             Sentiment: {logic_data['signals']['sentiment']} 
#             Risk: {logic_data['signals']['risk']}
#             Portfolio: {logic_data['signals']['portfolio']}
            
#             WEIGHTS USED: {logic_data['weights']}
            
#             TASK:
#             1. Provide probability assessment (0-100%) for trade success
#             2. Make final arbitration decision (CONFIRM/REVERSE/SOFTEN/STRENGTHEN)
#             3. Set final action (BUY/SELL/HOLD) with confidence (50-95)
#             4. Calculate position size (0-1000 shares) based on confidence and risk
#             5. Provide comprehensive reasoning
            
#             Return JSON:
#             {{
#                 "probability_assessment": {{
#                     "success_probability": 0-100,
#                     "risk_adjusted_return": 0-100,
#                     "confidence_level": "low/medium/high"
#                 }},
#                 "arbitration_decision": "CONFIRM/REVERSE/SOFTEN/STRENGTHEN",
#                 "final_action": "BUY/SELL/HOLD",
#                 "final_confidence": 50-95,
#                 "position_size": 0-1000,
#                 "reasoning": "comprehensive explanation",
#                 "risk_management_notes": "key risk considerations"
#             }}
#             """
            
#             response = self._call_groq_llm(prompt)
#             return self._parse_llm_json_response(response)
            
#         except Exception as e:
#             logger.error(f"AI arbitration layer failed: {e}")
#             return {}

#     # ------------------------------------------------------------
#     # COMPREHENSIVE SYNTHESIS
#     # ------------------------------------------------------------
#     def synthesize(
#         self,
#         ticker: str,
#         technical_result: Dict,
#         sentiment_result: Dict = None,
#         risk_metrics: Dict = None,
#         portfolio_metrics: Dict = None,
#         current_price: float = None,
#     ) -> Dict[str, Any]:

#         logger.info(f"🤖 MASTER AGENT: Starting AI synthesis for {ticker}")

#         # ========= 1) AI-ENHANCED SIGNAL EXTRACTION ==========
#         tech_signal, tech_confidence, tech_reason = self._extract_tech_with_ai(technical_result)
#         sentiment_signal, sentiment_confidence, sentiment_reason = self._parse_debate_with_ai(sentiment_result or {})
#         risk_appetite, risk_reason = self._analyze_risk_with_ai(risk_metrics or {})
        
#         # Portfolio signal
#         port_signal = 0.0
#         port_reason = "No portfolio data"
#         if portfolio_metrics:
#             try:
#                 alloc = float(portfolio_metrics.get("allocation_pct", 0))
#                 if 0 < alloc < 20:
#                     port_signal = 0.3
#                     port_reason = f"Moderate allocation suggested ({alloc}%)"
#                 elif alloc >= 20:
#                     port_signal = 0.1
#                     port_reason = f"High allocation ({alloc}%) - cautious"
#             except:
#                 pass

#         # ========= 2) DYNAMIC WEIGHT OPTIMIZATION ==========
#         signal_context = {
#             "technical": {"signal": tech_signal, "confidence": tech_confidence, "reason": tech_reason},
#             "sentiment": {"signal": sentiment_signal, "confidence": sentiment_confidence, "reason": sentiment_reason},
#             "risk": {"appetite": risk_appetite, "reason": risk_reason},
#             "portfolio": {"signal": port_signal, "reason": port_reason}
#         }
        
#         optimized_weights = self._optimize_weights_with_ai(signal_context)

#         # ========= 3) COMPOSITE SCORE CALCULATION ==========
#         composite = (
#             tech_signal * optimized_weights["technical"] +
#             sentiment_signal * optimized_weights["sentiment"] +
#             risk_appetite * optimized_weights["risk"] +
#             port_signal * optimized_weights["portfolio"]
#         )

#         # Initial logic decision
#         if composite > 0.15:
#             logic_action = "BUY"
#             logic_confidence = min(90, (tech_confidence * 0.4 + sentiment_confidence * 0.3 + 70 * 0.3))
#         elif composite < -0.15:
#             logic_action = "SELL" 
#             logic_confidence = min(90, (tech_confidence * 0.4 + sentiment_confidence * 0.3 + 70 * 0.3))
#         else:
#             logic_action = "HOLD"
#             logic_confidence = 50.0

#         # Build reasoning
#         logic_reasoning = f"Composite score: {composite:.3f}. "
#         logic_reasoning += f"Technical: {tech_reason}. "
#         logic_reasoning += f"Sentiment: {sentiment_reason}. "
#         logic_reasoning += f"Risk: {risk_reason}. "
#         logic_reasoning += f"Portfolio: {port_reason}"

#         logic_data = {
#             "action": logic_action,
#             "confidence": logic_confidence,
#             "reasoning": logic_reasoning,
#             "signals": {
#                 "technical": tech_signal,
#                 "sentiment": sentiment_signal, 
#                 "risk": risk_appetite,
#                 "portfolio": port_signal
#             },
#             "weights": optimized_weights
#         }

#         # ========= 4) ADVANCED AI ARBITRATION ==========
#         market_context = {
#             "ticker": ticker,
#             "current_price": current_price,
#             "technical_data": technical_result,
#             "sentiment_data": sentiment_result,
#             "risk_data": risk_metrics,
#             "portfolio_data": portfolio_metrics
#         }

#         ai_arbitration = self._ai_arbitration_layer(logic_data, market_context)

#         # ========= 5) FINAL DECISION ENSEMBLE ==========
#         if ai_arbitration and "final_action" in ai_arbitration:
#             final_action = ai_arbitration["final_action"]
#             final_confidence = ai_arbitration.get("final_confidence", logic_confidence)
#             position_size = ai_arbitration.get("position_size", 0)
#             ai_reasoning = ai_arbitration.get("reasoning", "AI arbitration completed")
#             probability_assessment = ai_arbitration.get("probability_assessment", {})
#         else:
#             # Fallback to logic layer
#             final_action = logic_action
#             final_confidence = logic_confidence
#             position_size = 100 if final_confidence > 70 else 50
#             ai_reasoning = "AI arbitration unavailable - using logic layer"
#             probability_assessment = {"success_probability": final_confidence, "confidence_level": "medium"}

#         # ========= 6) RISK MANAGEMENT ==========
#         stop_loss, take_profit = self._calculate_risk_levels(
#             final_action, current_price, final_confidence, risk_metrics
#         )

#         # ========= 7) COMPREHENSIVE OUTPUT ==========
#         return {
#             "ticker": ticker,
#             "current_price": current_price,
            
#             # Final AI Decision
#             "action": final_action,
#             "confidence": round(final_confidence, 1),
#             "quantity": position_size,
            
#             # AI Arbitration Results
#             "ai_arbitration": {
#                 "decision": ai_arbitration.get("arbitration_decision", "CONFIRM"),
#                 "probability_assessment": probability_assessment,
#                 "reasoning": ai_reasoning,
#                 "risk_notes": ai_arbitration.get("risk_management_notes", "")
#             },
            
#             # Logic Layer (for comparison)
#             "logic_layer": {
#                 "action": logic_action,
#                 "confidence": round(logic_confidence, 1),
#                 "reasoning": logic_reasoning,
#                 "composite_score": round(composite, 3)
#             },
            
#             # Signal Analysis
#             "signal_analysis": {
#                 "technical": {"signal": tech_signal, "confidence": tech_confidence, "reason": tech_reason},
#                 "sentiment": {"signal": sentiment_signal, "confidence": sentiment_confidence, "reason": sentiment_reason},
#                 "risk": {"appetite": risk_appetite, "reason": risk_reason},
#                 "portfolio": {"signal": port_signal, "reason": port_reason}
#             },
            
#             # Weight Analysis
#             "weight_analysis": {
#                 "optimized_weights": optimized_weights,
#                 "base_weights": self.base_weights
#             },
            
#             # Risk Management
#             "risk_management": {
#                 "stop_loss": stop_loss,
#                 "take_profit": take_profit,
#                 "position_size": position_size,
#                 "risk_level": risk_metrics.get("risk_level", "MEDIUM") if risk_metrics else "UNKNOWN"
#             },
            
#             # Metadata
#             "timestamp": time.time(),
#             "version": "AI_HYBRID_2.0"
#         }

#     def _calculate_risk_levels(self, action: str, current_price: float, confidence: float, risk_metrics: Dict) -> Tuple[Optional[float], Optional[float]]:
#         """Calculate stop loss and take profit levels"""
#         if not current_price or action == "HOLD":
#             return None, None
            
#         try:
#             # Base risk parameters
#             base_sl_pct = 0.08  # 8% stop loss
#             base_tp_pct = 0.15  # 15% take profit
            
#             # Adjust based on confidence
#             conf_factor = confidence / 100.0
#             sl_pct = base_sl_pct * (1.5 - conf_factor)  # Higher confidence = tighter stop
#             tp_pct = base_tp_pct * conf_factor          # Higher confidence = higher target
            
#             if action == "BUY":
#                 stop_loss = current_price * (1 - sl_pct)
#                 take_profit = current_price * (1 + tp_pct)
#             else:  # SELL
#                 stop_loss = current_price * (1 + sl_pct)
#                 take_profit = current_price * (1 - tp_pct)
                
#             return round(stop_loss, 2), round(take_profit, 2)
            
#         except Exception as e:
#             logger.error(f"Risk level calculation failed: {e}")
#             return None, None



# # trading_bot/agents/master_agent.py
# from typing import Dict, Any, Tuple
# import logging
# import json

# logger = logging.getLogger(__name__)


# class MasterAgent:
#     """
#     ADVANCED HYBRID MASTER AGENT

#     - Deterministic weighted logic layer (technical + debate + risk + portfolio)
#     - LLM arbitration layer (can confirm / soften / reverse / strengthen logic)
#     - LLM narrative layer (clean explanation, no invented numbers)

#     This class is used as "MasterLogic" by the BaseAgent wrapper:
#         from agents.master_agent import MasterAgent as MasterLogic

#     The wrapper should construct it as:
#         self.logic = MasterLogic(llm=llm)
#     """

#     def __init__(self, min_confidence: float = 60.0, llm=None):
#         self.min_confidence = min_confidence
#         self.llm = llm

#         # Base weights for composite signal
#         self.weights = {
#             "technical": 0.45,
#             "sentiment": 0.25,   # here "sentiment" = debate signal or sentiment proxy
#             "risk": 0.20,
#             "portfolio": 0.10,
#         }

#     # ------------------------------------------------------------
#     # PARSERS
#     # ------------------------------------------------------------
#     def _extract_tech(self, technical_result: Dict) -> Tuple[float, float]:
#         """
#         Map technical_result -> (signal, confidence)
#         signal ∈ {-1, 0, 1}, confidence ∈ [0, 100]
#         """
#         if not technical_result:
#             return 0, 50.0

#         action = str(
#             technical_result.get("action")
#             or technical_result.get("recommendation")
#             or ""
#         ).upper()

#         conf = float(technical_result.get("confidence", 50))

#         if "BUY" in action:
#             return 1, conf
#         if "SELL" in action:
#             return -1, conf
#         return 0, conf

#     def _parse_debate(self, sentiment_or_debate_result: Dict) -> Tuple[float, float]:
#         """
#         Handles both:
#         - DIRECT debate result: {'consensus': {'action': 'BUY', 'confidence': 78}}
#         - WRAPPED: {'debate': { ...above dict... }}
#         - Otherwise falls back to neutral.
#         """
#         if not sentiment_or_debate_result:
#             return 0.0, 50.0

#         # unwrap if nested under 'debate'
#         if "debate" in sentiment_or_debate_result and isinstance(sentiment_or_debate_result["debate"], dict):
#             raw = sentiment_or_debate_result["debate"]
#         else:
#             raw = sentiment_or_debate_result

#         consensus = raw.get("consensus", {})
#         action = consensus.get("action", "HOLD")
#         conf = float(consensus.get("confidence", 50))

#         if action == "BUY":
#             return 1.0, conf
#         if action == "SELL":
#             return -1.0, conf
#         return 0.0, conf

#     def _parse_risk(self, risk_metrics: Dict) -> float:
#         """
#         Map risk profile to directional penalty / bonus.
#         """
#         if not risk_metrics:
#             return 0.0

#         rl = risk_metrics.get("risk_level", "MEDIUM")

#         if rl == "VERY_HIGH":
#             return -0.6
#         if rl == "HIGH":
#             return -0.3
#         if rl == "MEDIUM":
#             return 0.0
#         # LOW or UNKNOWN → slight positive
#         return 0.2

#     # ------------------------------------------------------------
#     # MAIN SYNTHESIS
#     # ------------------------------------------------------------
#     def synthesize(
#         self,
#         ticker: str,
#         technical_result: Dict,
#         sentiment_result: Dict = None,   # debate_payload or news_payload
#         risk_metrics: Dict = None,
#         portfolio_metrics: Dict = None,
#         current_price: float = None,
#     ) -> Dict[str, Any]:

#         # ========= 1) LOGIC LAYER ==========
#         tech_sig, tech_conf = self._extract_tech(technical_result)
#         debate_sig, debate_conf = self._parse_debate(sentiment_result or {})
#         risk_sig = self._parse_risk(risk_metrics or {})

#         port_sig = 0.0
#         if portfolio_metrics and isinstance(portfolio_metrics, dict):
#             alloc = portfolio_metrics.get("allocation_pct") or 0
#             try:
#                 alloc = float(alloc)
#             except Exception:
#                 alloc = 0.0
#             # small positive if we have a modest suggested allocation
#             if 0 < alloc < 15:
#                 port_sig = 0.2

#         composite = (
#             tech_sig * self.weights["technical"] +
#             debate_sig * self.weights["sentiment"] +
#             risk_sig * self.weights["risk"] 
            
#         )

#         # Initial logic decision
#         if composite > 0.2:
#             logic_action = "BUY"
#             logic_conf = min(90, tech_conf * (0.6 + abs(composite)))
#         elif composite < -0.2:
#             logic_action = "SELL"
#             logic_conf = min(90, tech_conf * (0.6 + abs(composite)))
#         else:
#             logic_action = "HOLD"
#             logic_conf = 50.0

#         # Build simple rule-based reasoning string
#         reasoning_parts = []
#         if tech_sig > 0:
#             reasoning_parts.append("Technical signals are bullish")
#         elif tech_sig < 0:
#             reasoning_parts.append("Technical signals are bearish")

#         if risk_sig < 0:
#             if risk_metrics:
#                 reasoning_parts.append(f"Risk elevated ({risk_metrics.get('risk_level')})")
#             else:
#                 reasoning_parts.append("Risk elevated")

#         if port_sig > 0:
#             reasoning_parts.append("Portfolio allocation size is acceptable")

#         if debate_sig > 0:
#             reasoning_parts.append("Debate/consensus leans bullish")
#         elif debate_sig < 0:
#             reasoning_parts.append("Debate/consensus leans bearish")

#         base_reasoning = "; ".join(reasoning_parts) or "Mixed or neutral signals"

#         # ========= 2) LLM ARBITRATION LAYER ==========
#         llm_action = logic_action
#         llm_conf = logic_conf
#         llm_reasoning = None

#         if self.llm:
#             try:
#                 prompt = f"""
# You are an expert equity analyst.

# Below are structured signals for the stock {ticker}:

# TECHNICAL RESULT:
# {technical_result}

# SENTIMENT / DEBATE RESULT:
# {sentiment_result}

# RISK METRICS:
# {risk_metrics}

# PORTFOLIO METRICS:
# {portfolio_metrics}

# CURRENT PRICE:
# {current_price}

# CURRENT LOGIC DECISION:
# Action: {logic_action}
# Confidence: {logic_conf}
# Reasoning: {base_reasoning}

# TASK:
# 1. Decide whether the logic decision should be:
#     - confirmed
#     - reversed (BUY→SELL or SELL→BUY)
#     - softened (BUY→HOLD or SELL→HOLD)
#     - strengthened (HOLD→BUY or HOLD→SELL)

# 2. Choose ONE final action: "BUY", "SELL", or "HOLD".

# 3. Choose a final confidence between 50 and 95.

# 4. Provide a brief explanation (no invented numbers or prices).

# Respond ONLY in this JSON format (no text before or after):

# {{
#   "final_action": "BUY / SELL / HOLD",
#   "final_confidence": 78.5,
#   "reasoning": "Short explanation..."
# }}
# """
#                 raw = self.llm.ask(prompt)

#                 # Very defensive JSON parsing
#                 cleaned = str(raw).strip()
#                 cleaned = cleaned.replace("```json", "").replace("```", "").strip()
#                 parsed = json.loads(cleaned)

#                 llm_action = str(parsed.get("final_action", logic_action)).upper()
#                 if llm_action not in ("BUY", "SELL", "HOLD"):
#                     llm_action = logic_action

#                 try:
#                     llm_conf = float(parsed.get("final_confidence", logic_conf))
#                 except Exception:
#                     llm_conf = logic_conf

#                 # clamp
#                 if llm_conf < 50:
#                     llm_conf = 50.0
#                 if llm_conf > 95:
#                     llm_conf = 95.0

#                 llm_reasoning = parsed.get("reasoning", "") or "No LLM reasoning provided."

#             except Exception as e:
#                 logger.error(f"LLM arbitration failed in MasterAgent: {e}")
#                 llm_action = logic_action
#                 llm_conf = logic_conf
#                 llm_reasoning = "LLM arbitration unavailable."

#         # ========= 3) ENSEMBLE FINAL DECISION ==========
#         if llm_action != logic_action:
#             # LLM disagrees with logic → soften by blending confidences
#             final_action = llm_action
#             final_conf = (llm_conf * 0.7) + (logic_conf * 0.3)
#         else:
#             # LLM and logic agree → reinforce confidence
#             final_action = logic_action
#             final_conf = (llm_conf * 0.6) + (logic_conf * 0.4)
#             if final_conf > 95:
#                 final_conf = 95.0

#         final_conf = round(float(final_conf), 1)

#         # Quantity, SL/TP passthrough from portfolio/risk
#         quantity = 0
#         if portfolio_metrics and isinstance(portfolio_metrics, dict):
#             try:
#                 quantity = int(portfolio_metrics.get("quantity", 0))
#             except Exception:
#                 quantity = 0

#         stop_loss = risk_metrics.get("stop_loss_price") if risk_metrics else None
#         take_profit = risk_metrics.get("take_profit_price") if risk_metrics else None
#         risk_level = risk_metrics.get("risk_level") if risk_metrics else "UNKNOWN"

#         # ========= 4) OUTPUT STRUCTURE ==========
#         return {
#             "ticker": ticker,

#             # Final AI decision (what UI & rest of system should use)
#             "action": final_action,
#             "confidence": final_conf,

#             # Rule-based logic baseline
#             "logic_action": logic_action,
#             "logic_confidence": round(float(logic_conf), 1),
#             "reasoning": base_reasoning,

#             # LLM arbitration layer
#             "llm_action": llm_action,
#             "llm_confidence": round(float(llm_conf), 1),
#             "llm_reasoning": llm_reasoning,

#             # Signal breakdown
#             "signals": {
#                 "technical": tech_sig,
#                 "debate": debate_sig,
#                 "risk": risk_sig,
#                 # "portfolio": port_sig,
#             },

#             # Trade / risk params
#             "quantity": quantity,
#             "stop_loss": stop_loss,
#             "take_profit": take_profit,
#             "risk_level": risk_level,
#         }


# trading_bot/agents/master_agent.py
from typing import Dict, Any, Tuple, List
import logging
import json
import re

logger = logging.getLogger(__name__)


class MasterAgent:
    """
    ENHANCED HYBRID MASTER AGENT
    
    Features:
    - Multi-dimensional analysis (Technical, Sentiment, Risk, Portfolio)
    - Structured reasoning with clear breakdown
    - Dynamic confidence calculation
    - Price target and risk parameter generation
    - LLM arbitration with enhanced reasoning
    """

    def __init__(self, min_confidence: float = 60.0, llm=None):
        self.min_confidence = min_confidence
        self.llm = llm

        # Enhanced weights for composite signal
        self.weights = {
            "technical": 0.40,
            "sentiment": 0.25,
            "risk": 0.20,
            "portfolio": 0.15,
        }

    # ------------------------------------------------------------
    # ENHANCED PARSERS
    # ------------------------------------------------------------
    def _extract_tech(self, technical_result: Dict) -> Tuple[float, float, Dict]:
        """
        Enhanced technical parser with detailed signal extraction
        Returns: (signal, confidence, metadata)
        """
        if not technical_result:
            return 0, 50.0, {}

        action = str(
            technical_result.get("action") or 
            technical_result.get("recommendation") or 
            technical_result.get("signal") or ""
        ).upper()

        conf = float(technical_result.get("confidence", 50))
        
        # Extract price targets if available
        metadata = {
            "price_targets": {
                "entry": technical_result.get("entry_price"),
                "stop_loss": technical_result.get("stop_loss"),
                "take_profit": technical_result.get("take_profit")
            },
            "indicators": technical_result.get("indicators", {})
        }

        if "BUY" in action or "LONG" in action:
            return 1, conf, metadata
        if "SELL" in action or "SHORT" in action:
            return -1, conf, metadata
        return 0, conf, metadata

    def _parse_debate(self, sentiment_or_debate_result: Dict) -> Tuple[float, float, str]:
        """
        Enhanced debate parser with sentiment extraction
        """
        if not sentiment_or_debate_result:
            return 0.0, 50.0, "No sentiment data"

        # Unwrap nested debate structure
        if "debate" in sentiment_or_debate_result and isinstance(sentiment_or_debate_result["debate"], dict):
            raw = sentiment_or_debate_result["debate"]
        else:
            raw = sentiment_or_debate_result

        consensus = raw.get("consensus", {})
        action = consensus.get("action", "HOLD")
        conf = float(consensus.get("confidence", 50))
        reasoning = consensus.get("reasoning", "No consensus reasoning")

        if action == "BUY":
            return 1.0, conf, reasoning
        if action == "SELL":
            return -1.0, conf, reasoning
        return 0.0, conf, reasoning

    def _parse_risk(self, risk_metrics: Dict, current_price: float = None) -> Tuple[float, Dict]:
        """
        Enhanced risk analysis with dynamic penalty/bonus
        Returns: (risk_signal, risk_params)
        """
        if not risk_metrics:
            return 0.0, {}

        risk_level = risk_metrics.get("risk_level", "MEDIUM")
        volatility = risk_metrics.get("volatility", "MEDIUM")
        
        # Calculate risk signal with more granularity
        risk_signals = {
            "VERY_HIGH": -0.8,
            "HIGH": -0.4,
            "MEDIUM": 0.0,
            "LOW": 0.3,
            "VERY_LOW": 0.5
        }
        
        risk_signal = risk_signals.get(risk_level, 0.0)
        
        # Adjust for volatility
        if volatility == "HIGH":
            risk_signal -= 0.2
        elif volatility == "LOW":
            risk_signal += 0.1

        # Extract risk parameters
        risk_params = {
            "stop_loss": risk_metrics.get("stop_loss_price"),
            "take_profit": risk_metrics.get("take_profit_price"),
            "position_size": risk_metrics.get("position_size"),
            "risk_reward_ratio": risk_metrics.get("risk_reward_ratio")
        }

        return risk_signal, risk_params

    def _calculate_dynamic_confidence(self, signals: Dict, composite: float) -> float:
        """
        Calculate dynamic confidence based on signal strength and agreement
        """
        base_conf = 50.0
        
        # Signal strength contribution
        signal_strength = abs(composite) * 40  # Up to 40 points
        
        # Agreement bonus (when signals point in same direction)
        positive_signals = sum(1 for sig in signals.values() if sig > 0)
        negative_signals = sum(1 for sig in signals.values() if sig < 0)
        total_signals = len(signals)
        
        if total_signals > 0:
            agreement_ratio = max(positive_signals, negative_signals) / total_signals
            agreement_bonus = agreement_ratio * 20  # Up to 20 points
        else:
            agreement_bonus = 0
        
        confidence = base_conf + signal_strength + agreement_bonus
        
        return min(95.0, max(50.0, confidence))

    # ------------------------------------------------------------
    # ENHANCED SYNTHESIS
    # ------------------------------------------------------------
    def synthesize(
        self,
        ticker: str,
        technical_result: Dict,
        sentiment_result: Dict = None,
        risk_metrics: Dict = None,
        portfolio_metrics: Dict = None,
        current_price: float = None,
    ) -> Dict[str, Any]:

        # ========= 1) ENHANCED LOGIC LAYER ==========
        tech_sig, tech_conf, tech_meta = self._extract_tech(technical_result)
        debate_sig, debate_conf, debate_reasoning = self._parse_debate(sentiment_result or {})
        risk_sig, risk_params = self._parse_risk(risk_metrics or {}, current_price)

        # Portfolio signal calculation
        port_sig = 0.0
        portfolio_reasoning = ""
        if portfolio_metrics and isinstance(portfolio_metrics, dict):
            alloc = portfolio_metrics.get("allocation_pct") or 0
            try:
                alloc = float(alloc)
                if 0 < alloc < 10:  # Optimal allocation range
                    port_sig = 0.3
                    portfolio_reasoning = "Portfolio allocation within optimal range"
                elif alloc >= 10:
                    port_sig = -0.2
                    portfolio_reasoning = "Portfolio allocation nearing limit"
                else:
                    portfolio_reasoning = "No current portfolio allocation"
            except Exception:
                portfolio_reasoning = "Unable to parse portfolio allocation"

        # Calculate composite signal
        signals = {
            "technical": tech_sig,
            "sentiment": debate_sig,
            "risk": risk_sig,
            "portfolio": port_sig
        }
        
        composite = sum(signal * self.weights[category] 
                       for category, signal in signals.items())

        # Enhanced logic decision with dynamic confidence
        logic_conf = self._calculate_dynamic_confidence(signals, composite)
        
        if composite > 0.15:
            logic_action = "BUY"
        elif composite < -0.15:
            logic_action = "SELL"
        else:
            logic_action = "HOLD"

        # Build comprehensive reasoning
        reasoning_parts = []
        
        # Technical reasoning
        if tech_sig != 0:
            direction = "bullish" if tech_sig > 0 else "bearish"
            reasoning_parts.append(f"Technical analysis shows {direction} signals (conf: {tech_conf}%)")
        
        # Sentiment reasoning
        if debate_sig != 0:
            direction = "bullish" if debate_sig > 0 else "bearish"
            reasoning_parts.append(f"Market sentiment leans {direction}")
            if debate_reasoning and debate_reasoning != "No consensus reasoning":
                reasoning_parts.append(f"Sentiment insight: {debate_reasoning[:100]}...")
        
        # Risk reasoning
        risk_level = (risk_metrics or {}).get("risk_level", "UNKNOWN")
        if risk_sig < -0.3:
            reasoning_parts.append(f"High risk environment ({risk_level}) suggests caution")
        elif risk_sig > 0.2:
            reasoning_parts.append(f"Favorable risk conditions ({risk_level})")
        
        # Portfolio reasoning
        if portfolio_reasoning:
            reasoning_parts.append(portfolio_reasoning)

        base_reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Mixed or neutral signals across all dimensions"

        # ========= 2) ENHANCED LLM ARBITRATION LAYER ==========
        llm_action = logic_action
        llm_conf = logic_conf
        llm_reasoning = None
        llm_price_targets = {}

        if self.llm:
            try:
                prompt = self._build_llm_prompt(
                    ticker, technical_result, sentiment_result, 
                    risk_metrics, portfolio_metrics, current_price,
                    logic_action, logic_conf, base_reasoning
                )
                
                raw = self.llm.ask(prompt)
                parsed = self._parse_llm_response(raw)
                
                llm_action = parsed.get("final_action", logic_action).upper()
                if llm_action not in ("BUY", "SELL", "HOLD"):
                    llm_action = logic_action

                llm_conf = max(50.0, min(95.0, float(parsed.get("final_confidence", logic_conf))))
                llm_reasoning = parsed.get("reasoning", "No LLM reasoning provided.")
                
                # Extract price targets from LLM if provided
                llm_price_targets = {
                    "entry_price": parsed.get("entry_price"),
                    "stop_loss": parsed.get("stop_loss"),
                    "take_profit": parsed.get("take_profit")
                }

            except Exception as e:
                logger.error(f"LLM arbitration failed in MasterAgent: {e}")
                llm_action = logic_action
                llm_conf = logic_conf
                llm_reasoning = "LLM arbitration unavailable."

        # ========= 3) ENSEMBLE FINAL DECISION ==========
        if llm_action != logic_action:
            # LLM disagrees - weighted average with penalty
            final_action = llm_action
            final_conf = (llm_conf * 0.6) + (logic_conf * 0.4)
        else:
            # Agreement - reinforce confidence
            final_action = logic_action
            final_conf = (llm_conf * 0.7) + (logic_conf * 0.3)

        final_conf = round(max(self.min_confidence, min(95.0, final_conf)), 1)

        # ========= 4) PRICE TARGETS & RISK PARAMETERS ==========
        price_targets = self._calculate_price_targets(
            tech_meta, risk_params, llm_price_targets, current_price, final_action
        )

        # Quantity calculation
        quantity = self._calculate_position_size(
            portfolio_metrics, risk_metrics, final_conf, current_price
        )

        # ========= 5) ENHANCED OUTPUT STRUCTURE ==========
        return {
            "ticker": ticker,
            "current_price": current_price,

            # Final decision
            "action": final_action,
            "confidence": final_conf,
            "reasoning": llm_reasoning or base_reasoning,

            # Price targets
            "entry_price": price_targets["entry_price"],
            "stop_loss": price_targets["stop_loss"],
            "take_profit": price_targets["take_profit"],
            "risk_reward_ratio": price_targets["risk_reward_ratio"],

            # Position sizing
            "quantity": quantity,
            "position_size": portfolio_metrics.get("position_size") if portfolio_metrics else None,

            # Signal breakdown
            "signals": {
                "technical": round(tech_sig, 3),
                "sentiment": round(debate_sig, 3),
                "risk": round(risk_sig, 3),
                "portfolio": round(port_sig, 3),
                "composite": round(composite, 3)
            },

            # Confidence breakdown
            "confidence_breakdown": {
                "technical": tech_conf,
                "sentiment": debate_conf,
                "composite": logic_conf,
                "final": final_conf
            },

            # Metadata
            "risk_level": (risk_metrics or {}).get("risk_level", "UNKNOWN"),
            "timestamp": self._get_current_timestamp()
        }

    def _build_llm_prompt(self, ticker: str, technical_result: Dict, sentiment_result: Dict,
                         risk_metrics: Dict, portfolio_metrics: Dict, current_price: float,
                         logic_action: str, logic_conf: float, base_reasoning: str) -> str:
        """Build enhanced LLM prompt for arbitration"""
        return f"""
You are an expert financial analyst providing final trading decisions.

CONTEXT FOR {ticker}:
- Current Price: {current_price}
- Technical Signals: {json.dumps(technical_result, indent=2)}
- Market Sentiment: {json.dumps(sentiment_result, indent=2)}
- Risk Assessment: {json.dumps(risk_metrics, indent=2)}
- Portfolio Context: {json.dumps(portfolio_metrics, indent=2)}

INITIAL ANALYSIS:
Action: {logic_action}
Confidence: {logic_conf}%
Reasoning: {base_reasoning}

YOUR TASK:
1. Analyze all factors (technical, sentiment, risk, portfolio)
2. Provide FINAL trading decision with confidence (50-95%)
3. Suggest realistic price targets based on current price {current_price}
4. Explain your reasoning clearly

RESPONSE FORMAT (JSON only):
{{
  "final_action": "BUY/SELL/HOLD",
  "final_confidence": 75.0,
  "entry_price": 150.25,
  "stop_loss": 148.50,
  "take_profit": 155.75,
  "reasoning": "Clear explanation covering technical setup, market sentiment, risk assessment, and portfolio fit..."
}}
"""

    def _parse_llm_response(self, raw_response: str) -> Dict:
        """Parse LLM response with enhanced error handling"""
        try:
            cleaned = str(raw_response).strip()
            cleaned = re.sub(r'```json\n?|\n?```', '', cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Fallback: extract key information using regex
            return self._extract_fallback_response(cleaned)

    def _extract_fallback_response(self, text: str) -> Dict:
        """Fallback response parser using regex"""
        fallback = {
            "final_action": "HOLD",
            "final_confidence": 50.0,
            "reasoning": "Fallback: Could not parse LLM response"
        }
        
        # Simple regex extraction for critical fields
        action_match = re.search(r'(BUY|SELL|HOLD)', text.upper())
        if action_match:
            fallback["final_action"] = action_match.group(1)
            
        conf_match = re.search(r'(\d+\.?\d*)%?', text)
        if conf_match:
            try:
                conf = float(conf_match.group(1))
                fallback["final_confidence"] = max(50.0, min(95.0, conf))
            except ValueError:
                pass
                
        return fallback

    def _calculate_price_targets(self, tech_meta: Dict, risk_params: Dict, 
                               llm_targets: Dict, current_price: float, action: str) -> Dict:
        """Calculate final price targets with fallback logic"""
        if action == "HOLD" or not current_price:
            return {
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "risk_reward_ratio": None
            }

        # Priority: LLM -> Technical -> Risk -> Calculated
        entry_price = (
            llm_targets.get("entry_price") or
            tech_meta["price_targets"].get("entry") or
            risk_params.get("entry_price") or
            current_price
        )

        stop_loss = (
            llm_targets.get("stop_loss") or
            tech_meta["price_targets"].get("stop_loss") or
            risk_params.get("stop_loss")
        )

        take_profit = (
            llm_targets.get("take_profit") or
            tech_meta["price_targets"].get("take_profit") or
            risk_params.get("take_profit")
        )

        # Calculate risk-reward ratio if both SL and TP are available
        risk_reward_ratio = None
        if stop_loss and take_profit and entry_price:
            try:
                if action == "BUY":
                    risk = entry_price - stop_loss
                    reward = take_profit - entry_price
                else:  # SELL
                    risk = stop_loss - entry_price
                    reward = entry_price - take_profit
                
                if risk > 0:
                    risk_reward_ratio = round(reward / risk, 2)
            except (TypeError, ZeroDivisionError):
                pass

        return {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": risk_reward_ratio
        }

    def _calculate_position_size(self, portfolio_metrics: Dict, risk_metrics: Dict,
                               confidence: float, current_price: float) -> int:
        """Calculate position size based on confidence and risk parameters"""
        if not current_price or current_price <= 0:
            return 0

        # Base position size from portfolio metrics
        base_quantity = 0
        if portfolio_metrics:
            try:
                base_quantity = int(portfolio_metrics.get("quantity", 0))
            except (TypeError, ValueError):
                pass

        # Adjust based on confidence
        confidence_multiplier = confidence / 100.0
        
        # Adjust based on risk
        risk_multiplier = 1.0
        risk_level = (risk_metrics or {}).get("risk_level", "MEDIUM")
        if risk_level in ["HIGH", "VERY_HIGH"]:
            risk_multiplier = 0.5
        elif risk_level == "LOW":
            risk_multiplier = 1.2

        final_quantity = int(base_quantity * confidence_multiplier * risk_multiplier)
        return max(0, final_quantity)

    def _get_current_timestamp(self) -> str:
        """Get current timestamp for logging"""
        from datetime import datetime
        return datetime.now().isoformat()