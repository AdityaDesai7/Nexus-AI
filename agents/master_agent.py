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



# trading_bot/agents/master_agent.py
from typing import Dict, Any, Tuple

class MasterAgent:
    """
    Pure logic master agent. synthesize(...) returns final action/confidence/reasoning.
    """

    def __init__(self, min_confidence: float = 60.0):
        self.min_confidence = min_confidence
        self.weights = {"technical": 0.45, "sentiment": 0.25, "risk": 0.2, "portfolio": 0.1}

    def _extract_tech(self, technical_result: Dict) -> Tuple[float, float]:
        # returns (signal: -1/0/1, confidence 0-100)
        if not technical_result:
            return 0, 50.0
        action = str(technical_result.get("action") or technical_result.get("recommendation") or "").upper()
        conf = float(technical_result.get("confidence", 50))
        if "BUY" in action:
            return 1, conf
        if "SELL" in action:
            return -1, conf
        return 0, conf

    def _parse_debate(self, debate_result: Dict) -> Tuple[float, float]:
        if not debate_result:
            return 0.0, 50.0
        consensus = debate_result.get("consensus", {})
        action = consensus.get("action", "HOLD")
        conf = float(consensus.get("confidence", 50))
        if action == "BUY":
            return 1.0, conf
        if action == "SELL":
            return -1.0, conf
        return 0.0, conf

    def _parse_risk(self, risk_metrics: Dict) -> float:
        if not risk_metrics:
            return 0.0
        rl = risk_metrics.get("risk_level", "MEDIUM")
        if rl == "VERY_HIGH":
            return -0.6
        if rl == "HIGH":
            return -0.3
        if rl == "MEDIUM":
            return 0.0
        return 0.2

    def synthesize(self,
                   ticker: str,
                   technical_result: Dict,
                   sentiment_result: Dict = None,
                   risk_metrics: Dict = None,
                   portfolio_metrics: Dict = None,
                   current_price: float = None) -> Dict[str, Any]:
        tech_sig, tech_conf = self._extract_tech(technical_result)
        debate_sig, debate_conf = self._parse_debate(sentiment_result if sentiment_result and "consensus" in sentiment_result else {})
        risk_sig = self._parse_risk(risk_metrics)
        # portfolio influence: prefer 0.0 to small positive if allocation sensible
        port_sig = 0.0
        port_conf = 50.0
        if portfolio_metrics and isinstance(portfolio_metrics, dict):
            alloc = portfolio_metrics.get("allocation_pct") or portfolio_metrics.get("allocation_pct", 0)
            if alloc and alloc > 0 and alloc < 15:
                port_sig = 0.2
                port_conf = 60.0

        # Weighted composite
        weighted = (tech_sig * self.weights["technical"] +
                    debate_sig * self.weights["sentiment"] +
                    risk_sig * self.weights["risk"] +
                    port_sig * self.weights["portfolio"])

        # final action
        if weighted > 0.2:
            action = "BUY"
            confidence = min(90, tech_conf * (0.6 + abs(weighted)))
        elif weighted < -0.2:
            action = "SELL"
            confidence = min(90, tech_conf * (0.6 + abs(weighted)))
        else:
            action = "HOLD"
            confidence = 50.0

        reasoning_parts = []
        if tech_sig > 0:
            reasoning_parts.append("Technical signals are bullish")
        elif tech_sig < 0:
            reasoning_parts.append("Technical signals are bearish")
        if risk_sig < 0:
            reasoning_parts.append(f"Risk elevated ({risk_metrics.get('risk_level')})" if risk_metrics else "Risk elevated")
        if port_sig > 0:
            reasoning_parts.append("Portfolio sizing OK")

        return {
            "ticker": ticker,
            "action": action,
            "confidence": round(float(confidence), 1),
            "reasoning": "; ".join(reasoning_parts) or "Mixed signals",
            "signals": {
                "technical": tech_sig,
                "debate": debate_sig,
                "risk": risk_sig,
                "portfolio": port_sig
            },
            "quantity": int(portfolio_metrics.get("quantity", 0) if portfolio_metrics else 0),
            "stop_loss": risk_metrics.get("stop_loss_price") if risk_metrics else None,
            "take_profit": risk_metrics.get("take_profit_price") if risk_metrics else None,
            "risk_level": risk_metrics.get("risk_level") if risk_metrics else "UNKNOWN"
        }
