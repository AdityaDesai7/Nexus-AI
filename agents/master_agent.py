# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv

# load_dotenv()
# llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

# class MasterAgent:
#     def oversee(self, state: dict) -> dict:
#         prompt = f"Review agent outputs for {state['ticker']} (e.g., NTPC.NS): {state}. Final approval or adjustment."
#         return {"approved": True, "notes": llm.invoke(prompt).content}


import pandas as pd
import numpy as np
from typing import Dict, Tuple


class MasterAgent:
    """Master agent - Synthesizes all signals autonomously"""
    
    def __init__(self, min_confidence: float = 60):
        self.min_confidence = min_confidence
        
        # Signal weights
        self.weights = {
            'technical': 0.40,  # Technical is most reliable (autonomous)
            'sentiment': 0.30,  # Sentiment context
            'risk': 0.20,       # Risk management
            'portfolio': 0.10   # Portfolio alignment
        }
    
    def synthesize(self,
                   ticker: str,
                   technical_result: Dict,
                   sentiment_result: Dict = None,
                   risk_metrics: Dict = None,
                   portfolio_metrics: Dict = None,
                   current_price: float = None) -> Dict:
        """
        Synthesize all signals for final decision
        
        Returns: Master decision with action, confidence, reasoning
        """
        
        # Extract signals
        tech_signal, tech_conf = self._parse_technical(technical_result)
        sent_signal, sent_conf = self._parse_sentiment(sentiment_result)
        risk_signal = self._parse_risk(risk_metrics)
        port_signal, port_conf = self._parse_portfolio(portfolio_metrics)
        
        # Calculate consensus confidence
        signals = [tech_signal, sent_signal, risk_signal]
        consensus = sum([1 for s in signals if s != 0]) / len(signals)
        
        # Calculate weighted decision
        weighted_action = (
            tech_signal * self.weights['technical'] +
            sent_signal * self.weights['sentiment'] +
            risk_signal * self.weights['risk'] +
            port_signal * self.weights['portfolio']
        )
        
        # Determine final action
        if weighted_action > 0.2:
            final_action = "BUY"
            confidence = min(90, tech_conf * (0.5 + consensus))
        elif weighted_action < -0.2:
            final_action = "SELL"
            confidence = min(90, tech_conf * (0.5 + consensus))
        else:
            final_action = "HOLD"
            confidence = 50
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            final_action,
            tech_signal,
            sent_signal,
            risk_metrics,
            consensus
        )
        
        # Calculate quantity
        quantity = portfolio_metrics.get('quantity', 10) if portfolio_metrics else 10
        
        return {
            "action": final_action,
            "confidence": confidence,
            "quantity": quantity,
            "reasoning": reasoning,
            "signals": {
                "technical": tech_signal,
                "sentiment": sent_signal,
                "risk": risk_signal,
                "portfolio": port_signal,
                "consensus": consensus
            },
            "risk_level": risk_metrics.get('risk_level', 'MEDIUM') if risk_metrics else 'MEDIUM',
            "stop_loss": risk_metrics.get('stop_loss_price', 0) if risk_metrics else 0,
            "take_profit": risk_metrics.get('take_profit_price', 0) if risk_metrics else 0
        }
    
    def _parse_technical(self, technical_result: Dict) -> Tuple[float, float]:
        """Parse technical signal (1=BUY, -1=SELL, 0=HOLD)"""
        if not technical_result:
            return 0, 50
        
        rec = technical_result.get('recommendation', '')
        confidence = self._extract_confidence(rec)
        
        if 'BUY' in rec.upper():
            return 1, confidence
        elif 'SELL' in rec.upper():
            return -1, confidence
        else:
            return 0, confidence
    
    def _parse_sentiment(self, sentiment_result: Dict) -> Tuple[float, float]:
        """Parse sentiment signal"""
        if not sentiment_result:
            return 0, 50
        
        action = sentiment_result.get('action', 'HOLD')
        confidence = sentiment_result.get('confidence', 50) / 100
        
        if action == "BUY":
            return confidence, confidence * 100
        elif action == "SELL":
            return -confidence, confidence * 100
        else:
            return 0, 50
    
    def _parse_risk(self, risk_metrics: Dict) -> float:
        """Parse risk signal"""
        if not risk_metrics:
            return 0
        
        risk_level = risk_metrics.get('risk_level', 'MEDIUM')
        position_size = risk_metrics.get('position_size', 0.05)
        
        # Risk should reduce signal if too high
        if risk_level == "VERY_HIGH":
            return -0.5  # Caution signal
        elif risk_level == "HIGH":
            return -0.2  # Slight caution
        elif position_size < 0.02:
            return -0.3  # Too small position size = risky
        else:
            return 0.2  # Acceptable risk
    
    def _parse_portfolio(self, portfolio_metrics: Dict) -> Tuple[float, float]:
        """Parse portfolio signal"""
        if not portfolio_metrics:
            return 0, 50
        
        allocation_pct = portfolio_metrics.get('allocation_pct', 5)
        
        # Portfolio should validate position sizing
        if allocation_pct > 15:
            return -0.3, 60  # Too large position
        elif allocation_pct < 1:
            return 0, 50  # Too small
        else:
            return 0.2, 70  # Good position sizing
    
    def _extract_confidence(self, recommendation_str: str) -> float:
        """Extract confidence from recommendation string"""
        try:
            parts = recommendation_str.split("Confidence: ")
            if len(parts) > 1:
                conf_str = parts[1].split("%")[0]
                return float(conf_str)
        except:
            pass
        return 50.0
    
    def _generate_reasoning(self,
                           action: str,
                           tech_signal: float,
                           sent_signal: float,
                           risk_metrics: Dict,
                           consensus: float) -> str:
        """Generate human-readable reasoning"""
        
        if action == "BUY":
            reasons = []
            
            if tech_signal > 0:
                reasons.append("✓ Technical indicators show BUY signal")
            
            if sent_signal > 0:
                reasons.append("✓ Sentiment is positive")
            
            if consensus > 0.5:
                reasons.append(f"✓ Strong consensus ({consensus:.0%})")
            
            if risk_metrics and risk_metrics.get('risk_level') in ['LOW', 'MEDIUM']:
                reasons.append(f"✓ Risk level acceptable ({risk_metrics.get('risk_level')})")
            
            return " | ".join(reasons) if reasons else "Strong technical signals detected"
        
        elif action == "SELL":
            reasons = []
            
            if tech_signal < 0:
                reasons.append("✓ Technical indicators show SELL signal")
            
            if sent_signal < 0:
                reasons.append("✓ Sentiment is negative")
            
            if risk_metrics and risk_metrics.get('risk_level') in ['HIGH', 'VERY_HIGH']:
                reasons.append(f"✓ Risk is elevated ({risk_metrics.get('risk_level')})")
            
            return " | ".join(reasons) if reasons else "Technical indicators suggest SELL"
        
        else:
            return "Mixed signals suggest caution. Awaiting stronger signal."