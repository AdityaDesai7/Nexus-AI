# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv

# load_dotenv()
# bull_llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")
# bear_llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

# class DebateAgent:
#     def debate(self, state: dict) -> str:
#         bull_prompt = f"Argue BULL case for {state['ticker']} (e.g., NTPC.NS): {state['technical']}, Sentiment: {state['sentiment']}."
#         bear_prompt = f"Argue BEAR case for {state['ticker']} (e.g., NTPC.NS): {state['technical']}, Sentiment: {state['sentiment']}."
#         bull_case = bull_llm.invoke(bull_prompt).content
#         bear_case = bear_llm.invoke(bear_prompt).content
#         mod_prompt = f"Bull: {bull_case}\nBear: {bear_case}\nConsensus recommendation:"
#         consensus = bull_llm.invoke(mod_prompt).content
#         return consensus


# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple


# class DebateAgent:
#     """Autonomous debate agent - NO LLM DEPENDENCY"""
    
#     def __init__(self):
#         self.debate_history = []
    
#     def debate(self,
#                ticker: str,
#                technical_result: Dict,
#                risk_metrics: Dict,
#                price_data: pd.DataFrame = None,
#                sentiment_score: float = 50) -> Dict:
#         """
#         Autonomous debate between BULL and BEAR perspectives
        
#         Args:
#             ticker: Stock ticker
#             technical_result: Result from technical agent
#             risk_metrics: Risk metrics from risk agent
#             price_data: Historical price data (optional)
#             sentiment_score: Sentiment score (0-100, 50 = neutral)
        
#         Returns:
#             Dict with bull_case, bear_case, and consensus
#         """
        
#         # Generate BULL case
#         bull_case = self._generate_bull_case(
#             ticker,
#             technical_result,
#             risk_metrics,
#             price_data,
#             sentiment_score
#         )
        
#         # Generate BEAR case
#         bear_case = self._generate_bear_case(
#             ticker,
#             technical_result,
#             risk_metrics,
#             price_data,
#             sentiment_score
#         )
        
#         # Synthesize consensus
#         consensus = self._synthesize_consensus(
#             bull_case,
#             bear_case,
#             technical_result,
#             risk_metrics
#         )
        
#         return {
#             "ticker": ticker,
#             "bull_case": bull_case,
#             "bear_case": bear_case,
#             "consensus": consensus,
#             "bull_strength": bull_case["strength"],
#             "bear_strength": bear_case["strength"]
#         }
    
#     def _generate_bull_case(self,
#                            ticker: str,
#                            technical_result: Dict,
#                            risk_metrics: Dict,
#                            price_data: pd.DataFrame = None,
#                            sentiment_score: float = 50) -> Dict:
#         """Generate BULL (BUY) case with logical arguments"""
        
#         arguments = []
#         strength = 0  # 0-100 scale
        
#         # Extract technical signals
#         tech_rec = technical_result.get('recommendation', '')
        
#         # Argument 1: Technical Analysis
#         if "BUY" in tech_rec.upper():
#             confidence = self._extract_confidence(tech_rec)
#             arguments.append(f"✓ Technical Signal: Autonomous agent detected BUY signal ({confidence:.1f}% confidence)")
#             strength += min(30, confidence * 0.3)
        
#         # Argument 2: MACD Momentum
#         if technical_result.get('macd', 0) > technical_result.get('macd_signal', 0):
#             arguments.append("✓ MACD Momentum: Fast EMA above Slow EMA indicates upward momentum")
#             strength += 15
        
#         # Argument 3: Price at Lower Bollinger Band
#         bb_position = self._calculate_bb_position(technical_result)
#         if bb_position < 0.2:
#             arguments.append(f"✓ Bollinger Bands: Price near lower band ({bb_position:.1%}) suggests bounce potential")
#             strength += 15
        
#         # Argument 4: RSI Oversold
#         rsi = technical_result.get('rsi', 50)
#         if rsi < 30:
#             arguments.append(f"✓ RSI Oversold: RSI at {rsi:.1f} indicates oversold condition, potential reversal")
#             strength += 15
        
#         # Argument 5: Sentiment
#         if sentiment_score > 60:
#             arguments.append(f"✓ Market Sentiment: Positive sentiment ({sentiment_score:.0f}/100) supports upside")
#             strength += 10
        
#         # Argument 6: Risk Level
#         risk_level = risk_metrics.get('risk_level', 'MEDIUM')
#         if risk_level in ['LOW', 'MEDIUM']:
#             arguments.append(f"✓ Risk Management: Risk level is {risk_level}, acceptable for entry")
#             strength += 10
        
#         # Argument 7: Support Level
#         support = risk_metrics.get('support', 0)
#         if price_data is not None and len(price_data) > 0:
#             current_price = price_data['Close'].iloc[-1]
#             distance_to_support = (current_price - support) / current_price
#             if distance_to_support < 0.15:  # Within 15% of support
#                 arguments.append(f"✓ Support Level: Price near support (15% away), strong downside protection")
#                 strength += 10
        
#         # Argument 8: Volume confirmation (if available)
#         if price_data is not None and len(price_data) > 1:
#             recent_vol = price_data['Volume'].iloc[-1]
#             avg_vol = price_data['Volume'].rolling(20).mean().iloc[-1]
#             if recent_vol > avg_vol * 1.2:
#                 arguments.append(f"✓ Volume Confirmation: Above average volume confirms BUY signal")
#                 strength += 5
        
#         if not arguments:
#             arguments.append("• Market conditions neutral, no strong bull signals")
#             strength = 25
        
#         strength = min(100, strength)
        
#         return {
#             "arguments": arguments,
#             "strength": strength,
#             "recommendation": "BUY" if strength > 50 else "HOLD"
#         }
    
#     def _generate_bear_case(self,
#                            ticker: str,
#                            technical_result: Dict,
#                            risk_metrics: Dict,
#                            price_data: pd.DataFrame = None,
#                            sentiment_score: float = 50) -> Dict:
#         """Generate BEAR (SELL) case with logical arguments"""
        
#         arguments = []
#         strength = 0
        
#         # Extract technical signals
#         tech_rec = technical_result.get('recommendation', '')
        
#         # Argument 1: Technical Analysis
#         if "SELL" in tech_rec.upper():
#             confidence = self._extract_confidence(tech_rec)
#             arguments.append(f"✓ Technical Signal: Autonomous agent detected SELL signal ({confidence:.1f}% confidence)")
#             strength += min(30, confidence * 0.3)
        
#         # Argument 2: MACD Divergence
#         if technical_result.get('macd', 0) < technical_result.get('macd_signal', 0):
#             arguments.append("✓ MACD Divergence: Fast EMA below Slow EMA indicates downward momentum")
#             strength += 15
        
#         # Argument 3: Price at Upper Bollinger Band
#         bb_position = self._calculate_bb_position(technical_result)
#         if bb_position > 0.8:
#             arguments.append(f"✓ Bollinger Bands: Price near upper band ({bb_position:.1%}) suggests reversal risk")
#             strength += 15
        
#         # Argument 4: RSI Overbought
#         rsi = technical_result.get('rsi', 50)
#         if rsi > 70:
#             arguments.append(f"✓ RSI Overbought: RSI at {rsi:.1f} indicates overbought, pullback likely")
#             strength += 15
        
#         # Argument 5: Negative Sentiment
#         if sentiment_score < 40:
#             arguments.append(f"✓ Market Sentiment: Negative sentiment ({sentiment_score:.0f}/100) warns of downside")
#             strength += 10
        
#         # Argument 6: High Risk Level
#         risk_level = risk_metrics.get('risk_level', 'MEDIUM')
#         if risk_level in ['HIGH', 'VERY_HIGH']:
#             arguments.append(f"✓ Risk Alert: Risk level is {risk_level}, elevated volatility suggests caution")
#             strength += 15
        
#         # Argument 7: Resistance Level
#         resistance = risk_metrics.get('resistance', float('inf'))
#         if price_data is not None and len(price_data) > 0:
#             current_price = price_data['Close'].iloc[-1]
#             distance_to_resistance = (resistance - current_price) / current_price
#             if distance_to_resistance < 0.10:  # Within 10% of resistance
#                 arguments.append(f"✓ Resistance Level: Price near resistance (10% away), upside limited")
#                 strength += 10
        
#         # Argument 8: Declining Volume
#         if price_data is not None and len(price_data) > 1:
#             recent_vol = price_data['Volume'].iloc[-1]
#             avg_vol = price_data['Volume'].rolling(20).mean().iloc[-1]
#             if recent_vol < avg_vol * 0.8:
#                 arguments.append(f"✓ Weak Volume: Below average volume, momentum lacks confirmation")
#                 strength += 5
        
#         if not arguments:
#             arguments.append("• No strong bear signals present")
#             strength = 25
        
#         strength = min(100, strength)
        
#         return {
#             "arguments": arguments,
#             "strength": strength,
#             "recommendation": "SELL" if strength > 50 else "HOLD"
#         }
    
#     def _synthesize_consensus(self,
#                              bull_case: Dict,
#                              bear_case: Dict,
#                              technical_result: Dict,
#                              risk_metrics: Dict) -> Dict:
#         """Synthesize BULL vs BEAR debate into consensus recommendation"""
        
#         bull_strength = bull_case["strength"]
#         bear_strength = bear_case["strength"]
        
#         total = bull_strength + bear_strength
        
#         if total == 0:
#             bull_pct = 50
#             bear_pct = 50
#         else:
#             bull_pct = (bull_strength / total) * 100
#             bear_pct = (bear_strength / total) * 100
        
#         # Determine consensus action
#         if abs(bull_strength - bear_strength) < 15:
#             # Close debate - HOLD
#             consensus_action = "HOLD"
#             confidence = 40
#             reasoning = "Debate is close: both bull and bear cases have merit"
#         elif bull_strength > bear_strength:
#             # Bull wins
#             difference = bull_strength - bear_strength
#             consensus_action = "BUY"
#             confidence = min(85, 50 + (difference / 2))
#             reasoning = f"Bull case is stronger (+{difference:.0f} points). {len(bull_case['arguments'])} strong arguments for upside."
#         else:
#             # Bear wins
#             difference = bear_strength - bull_strength
#             consensus_action = "SELL"
#             confidence = min(85, 50 + (difference / 2))
#             reasoning = f"Bear case is stronger (+{difference:.0f} points). {len(bear_case['arguments'])} strong arguments for downside."
        
#         # Get technical agent's view for reference
#         tech_rec = technical_result.get('recommendation', '')
#         tech_action = "BUY" if "BUY" in tech_rec.upper() else "SELL" if "SELL" in tech_rec.upper() else "HOLD"
        
#         return {
#             "consensus_action": consensus_action,
#             "consensus_confidence": confidence,
#             "bull_percentage": bull_pct,
#             "bear_percentage": bear_pct,
#             "reasoning": reasoning,
#             "aligns_with_technical": consensus_action == tech_action,
#             "debate_quality": self._assess_debate_quality(bull_strength, bear_strength)
#         }
    
#     def _calculate_bb_position(self, technical_result: Dict) -> float:
#         """Calculate price position within Bollinger Bands (0-1)"""
#         upper = technical_result.get('bollinger_upper', 1)
#         lower = technical_result.get('bollinger_lower', 0)
        
#         if upper == lower:
#             return 0.5
        
#         # This would need current price, approximate as middle
#         # In real use, pass current price
#         return 0.5
    
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
    
#     def _assess_debate_quality(self, bull_strength: float, bear_strength: float) -> str:
#         """Assess the quality/clarity of the debate"""
#         difference = abs(bull_strength - bear_strength)
        
#         if difference > 40:
#             return "CLEAR_WINNER"
#         elif difference > 20:
#             return "MODERATE_CONSENSUS"
#         else:
#             return "CLOSE_DEBATE"
    
#     def get_debate_summary(self, debate_result: Dict) -> str:
#         """Generate human-readable debate summary"""
        
#         bull_pct = debate_result["bull_percentage"]
#         bear_pct = debate_result["bear_percentage"]
#         consensus = debate_result["consensus_action"]
#         confidence = debate_result["consensus_confidence"]
        
#         summary = f"""
# ╔════════════════════════════════════════╗
# ║         BULL vs BEAR DEBATE            ║
# ╚════════════════════════════════════════╝

# BULL CASE ({bull_pct:.0f}%):
# {chr(10).join('  ' + arg for arg in debate_result.get('bull_arguments', []))}

# BEAR CASE ({bear_pct:.0f}%):
# {chr(10).join('  ' + arg for arg in debate_result.get('bear_arguments', []))}

# CONSENSUS: {consensus} (Confidence: {confidence:.0f}%)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# {debate_result["reasoning"]}
# """
#         return summary

# trading_bot/agents/debate_agent.py
# trading_bot/agents/debate_agent.py



from typing import Dict, Any

class DebateAgent:
    """
    Hybrid Debate Agent:
    - Logic scoring (safe, numeric)
    - LLM summary using Groq (safe wrapper)
    """

    def __init__(self, llm=None):
        self.llm = llm

    @staticmethod
    def _extract_float(d, key, default=0.0):
        try:
            return float(d.get(key, default))
        except:
            return default

    # -------------------------- LOGIC SCORING --------------------------
    def _score_logic(self, ticker, technical_result, risk_metrics, price_df, sentiment_score):
        bull = 0
        bear = 0
        bull_args = []
        bear_args = []

        # Technical analysis
        if technical_result:
            action = str(technical_result.get("action", "")).upper()
            conf = float(technical_result.get("confidence", 50))
            rsi = self._extract_float(technical_result, "rsi", 50)
            macd = self._extract_float(technical_result, "macd", 0)
            macd_signal = self._extract_float(technical_result, "macd_signal", 0)

            if action == "BUY":
                bull += conf * 0.6
                bull_args.append(f"Technical BUY signal ({conf}%)")
            elif action == "SELL":
                bear += conf * 0.6
                bear_args.append(f"Technical SELL signal ({conf}%)")

            if rsi < 30:
                bull += 10
                bull_args.append(f"RSI oversold ({rsi})")
            if rsi > 70:
                bear += 10
                bear_args.append(f"RSI overbought ({rsi})")

            if macd > macd_signal:
                bull += 8
                bull_args.append("MACD bullish")
            if macd < macd_signal:
                bear += 8
                bear_args.append("MACD bearish")

        # Risk scoring
        if risk_metrics:
            rl = risk_metrics.get("risk_level", "MEDIUM")
            if rl in ("HIGH", "VERY_HIGH"):
                bear += 20
                bear_args.append(f"High risk level: {rl}")
            else:
                bull += 5
                bull_args.append(f"Risk acceptable: {rl}")

        # Sentiment
        if sentiment_score >= 60:
            bull += 8
            bull_args.append(f"Positive sentiment ({sentiment_score})")
        elif sentiment_score <= 40:
            bear += 8
            bear_args.append(f"Negative sentiment ({sentiment_score})")

        # Support/resistance
        try:
            if price_df is not None and len(price_df) > 0:
                current = float(price_df["Close"].iloc[-1])
                sup = risk_metrics.get("support")
                res = risk_metrics.get("resistance")

                if sup and current - float(sup) < current * 0.15:
                    bull += 5
                    bull_args.append("Price near support")

                if res and float(res) - current < current * 0.10:
                    bear += 5
                    bear_args.append("Price near resistance")
        except:
            pass

        total = bull + bear
        bull_pct = (bull / total) * 100 if total else 50
        bear_pct = (bear / total) * 100 if total else 50

        return bull, bear, bull_args, bear_args, bull_pct, bear_pct

    # -------------------------- LLM SUMMARY --------------------------
    def _llm_summary(self, ticker, bull_args, bear_args, bull_strength, bear_strength, consensus_action, consensus_conf):
        if not self.llm:
            return "LLM unavailable — logic summary only."

        prompt = f"""
You are a professional financial analyst.

Ticker: {ticker}

Bull Case:
{bull_args}

Bear Case:
{bear_args}

Bull Strength: {bull_strength}
Bear Strength: {bear_strength}

Consensus Action: {consensus_action}
Consensus Confidence: {consensus_conf}%

Write a balanced 150–200 word debate summary.
DO NOT invent numbers. Use ONLY the arguments provided.
"""

        try:
            result = self.llm.ask(prompt)
            return str(result)
        except:
            return "LLM failed — fallback to logic-only summary."

    # -------------------------- PUBLIC METHOD --------------------------
    def debate(self, ticker: str, technical_result: Dict, risk_metrics: Dict, price_data=None, sentiment_score: float = 50.0):

        price_df = price_data.get("df") if isinstance(price_data, dict) else price_data

        bull, bear, bull_args, bear_args, bull_pct, bear_pct = \
            self._score_logic(ticker, technical_result, risk_metrics, price_df, sentiment_score)

        # Consensus logic
        if bull - bear > 15:
            action = "BUY"
            conf = min(90, 50 + (bull - bear) / 2)
        elif bear - bull > 15:
            action = "SELL"
            conf = min(90, 50 + (bear - bull) / 2)
        else:
            action = "HOLD"
            conf = 50

        narrative = self._llm_summary(
            ticker, bull_args, bear_args, round(bull, 1), round(bear, 1), action, round(conf, 1)
        )

        return {
            "ticker": ticker,
            "bull_case": {"arguments": bull_args, "strength": round(bull, 1)},
            "bear_case": {"arguments": bear_args, "strength": round(bear, 1)},
            "consensus": {
                "action": action,
                "confidence": round(conf, 1),
                "bull_pct": round(bull_pct, 1),
                "bear_pct": round(bear_pct, 1)
            },
            "narrative": narrative
        }
