# from langchain_groq import ChatGroq
# from models.simulation import Trade
# import os
# from dotenv import load_dotenv

# load_dotenv()
# llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

# class PortfolioManagerAgent:
#     def decide(self, state: dict, risk: dict) -> Trade:
#         prompt = f"Decide trade for {state['ticker']} (e.g., NTPC.NS). Technical: {state['technical']}. Sentiment: {state['sentiment']}. Risk: {risk}. Action: Buy/Sell/Hold, quantity based on balance."
#         response = llm.invoke(prompt).content
#         # Parse response for simplicity (adjust based on LLM output)
#         action = "Buy" if "buy" in response.lower() else "Sell" if "sell" in response.lower() else "Hold"
#         if action == "Hold":
#             return None
#         return Trade(ticker=state["ticker"], action=action, quantity=10, price=300.0)  # Approx NTPC.NS price




import pandas as pd
import numpy as np
from typing import Dict, Tuple
from models.simulation import Trade


class PortfolioManagerAgent:
    """Autonomous portfolio management - NO LLM DEPENDENCY"""
    
    def __init__(self, portfolio_value: float = 1000000):
        self.portfolio_value = portfolio_value
        self.max_position_pct = 0.10  # Max 10% per position
        self.min_position_pct = 0.01  # Min 1% per position
    
    def decide(self, 
               ticker: str, 
               current_price: float,
               technical_signal: Dict,
               sentiment_signal: Dict = None,
               risk_metrics: Dict = None,
               portfolio_state: Dict = None) -> Tuple[str, int, Dict]:
        """
        Autonomous decision on position sizing based on signals
        
        Args:
            ticker: Stock ticker
            current_price: Current stock price
            technical_signal: Signal from technical agent
            sentiment_signal: Signal from sentiment agent (optional)
            risk_metrics: Risk metrics from risk agent
            portfolio_state: Current portfolio state
        
        Returns:
            (action, quantity, metadata)
        """
        
        # Extract technical confidence
        tech_action = technical_signal.get('action', 'HOLD')
        tech_confidence = self._extract_confidence(technical_signal.get('recommendation', ''))
        
        # Extract sentiment signal if available
        sentiment_confidence = 0
        if sentiment_signal:
            sentiment_action = sentiment_signal.get('action', 'HOLD')
            sentiment_confidence = sentiment_signal.get('confidence', 0)
            
            # Adjust confidence based on sentiment agreement
            if tech_action == sentiment_action:
                tech_confidence = min(90, tech_confidence + 5)  # Boost if aligned
        
        # Get risk metrics
        position_size_pct = self._calculate_position_size(
            tech_confidence, 
            risk_metrics or {}
        )
        
        # Determine action
        if tech_confidence < 30:
            return "HOLD", 0, {"reason": "Low confidence"}
        
        if tech_action == "BUY":
            quantity = self._calculate_quantity_buy(current_price, position_size_pct)
            return "BUY", quantity, {
                "confidence": tech_confidence,
                "position_pct": position_size_pct,
                "allocation": f"₹{quantity * current_price:,.0f}"
            }
        
        elif tech_action == "SELL":
            quantity = self._calculate_quantity_sell(current_price, position_size_pct)
            return "SELL", quantity, {
                "confidence": tech_confidence,
                "position_pct": position_size_pct,
                "allocation": f"₹{quantity * current_price:,.0f}"
            }
        
        else:
            return "HOLD", 0, {"reason": "No clear signal"}
    
    def _extract_confidence(self, recommendation_str: str) -> float:
        """Extract confidence percentage from recommendation string"""
        try:
            parts = recommendation_str.split("Confidence: ")
            if len(parts) > 1:
                conf_str = parts[1].split("%")[0]
                return float(conf_str)
        except:
            pass
        return 50.0
    
    def _calculate_position_size(self, confidence: float, risk_metrics: Dict) -> float:
        """
        Calculate position size based on confidence and risk
        
        Logic:
        - Low confidence (0-30%): 0% position
        - Medium confidence (30-60%): 1-5% position
        - High confidence (60-90%): 5-10% position
        - Very high confidence (>90%): 8-10% position
        """
        if confidence < 30:
            return 0.0
        elif confidence < 60:
            # Linear interpolation: 30% -> 1%, 60% -> 5%
            return 0.01 + (confidence - 30) / 30 * 0.04
        elif confidence < 90:
            # Linear interpolation: 60% -> 5%, 90% -> 10%
            return 0.05 + (confidence - 60) / 30 * 0.05
        else:
            # Very high confidence: 8-10%
            return min(0.10, 0.08 + (confidence - 90) / 10 * 0.02)
    
    def _calculate_quantity_buy(self, current_price: float, position_pct: float) -> int:
        """Calculate quantity to buy"""
        allocation = self.portfolio_value * position_pct
        quantity = int(allocation / current_price)
        return max(1, quantity)  # Minimum 1 share
    
    def _calculate_quantity_sell(self, current_price: float, position_pct: float) -> int:
        """Calculate quantity to sell (assume we have positions)"""
        allocation = self.portfolio_value * position_pct
        quantity = int(allocation / current_price)
        return max(1, quantity)
    
    def get_allocation_metrics(self, 
                               action: str, 
                               quantity: int, 
                               current_price: float) -> Dict:
        """Get detailed allocation metrics"""
        allocation = quantity * current_price
        pct_of_portfolio = (allocation / self.portfolio_value) * 100
        
        return {
            "action": action,
            "quantity": quantity,
            "price": current_price,
            "allocation": allocation,
            "allocation_pct": pct_of_portfolio,
            "remaining_capital": self.portfolio_value - allocation
        }
