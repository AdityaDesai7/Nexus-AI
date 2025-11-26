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

# trading_bot/agents/debate_agent.py
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DebateAgent:
    """
    Enhanced Debate Agent with comprehensive scoring using Technical and Risk analysis
    """

    def __init__(self, llm=None):
        self.llm = llm

    @staticmethod
    def _extract_float(d, key, default=0.0):
        """Safely extract float values from nested dictionaries"""
        try:
            if isinstance(d, dict):
                return float(d.get(key, default))
            return default
        except (TypeError, ValueError):
            return default

    def _extract_technical_signals(self, technical_result: Dict) -> Tuple[float, List[str]]:
        """Extract and score technical analysis signals"""
        bull_score = 0.0
        bear_score = 0.0
        bull_args = []
        bear_args = []

        if not technical_result:
            return bull_score, bear_score, bull_args, bear_args

        try:
            # RSI Analysis
            rsi = self._extract_float(technical_result, "rsi", 50)
            if rsi < 30:
                bull_score += 20
                bull_args.append(f"RSI oversold ({rsi:.1f}) - strong buy signal")
            elif rsi < 40:
                bull_score += 10
                bull_args.append(f"RSI near oversold ({rsi:.1f}) - potential bounce")
            elif rsi > 70:
                bear_score += 20
                bear_args.append(f"RSI overbought ({rsi:.1f}) - strong sell signal")
            elif rsi > 60:
                bear_score += 10
                bear_args.append(f"RSI near overbought ({rsi:.1f}) - potential pullback")

            # MACD Analysis
            macd = self._extract_float(technical_result, "macd", 0)
            macd_signal = self._extract_float(technical_result, "macd_signal", 0)
            macd_hist = self._extract_float(technical_result, "macd_hist", 0)
            
            if macd > macd_signal and macd > 0:
                bull_score += 15
                bull_args.append("MACD bullish and above zero - strong uptrend")
            elif macd > macd_signal:
                bull_score += 8
                bull_args.append("MACD turning bullish - potential uptrend")
            elif macd < macd_signal and macd < 0:
                bear_score += 15
                bear_args.append("MACD bearish and below zero - strong downtrend")
            elif macd < macd_signal:
                bear_score += 8
                bear_args.append("MACD turning bearish - potential downtrend")

            # Bollinger Bands
            current_price = self._extract_float(technical_result, "latest_close", 0)
            bb_upper = self._extract_float(technical_result, "bollinger_upper", 0)
            bb_lower = self._extract_float(technical_result, "bollinger_lower", 0)
            
            if current_price and bb_lower and bb_upper:
                if current_price < bb_lower:
                    bull_score += 12
                    bull_args.append(f"Price below lower Bollinger Band ({bb_lower:.2f}) - oversold")
                elif current_price > bb_upper:
                    bear_score += 12
                    bear_args.append(f"Price above upper Bollinger Band ({bb_upper:.2f}) - overbought")
                elif current_price > (bb_upper + bb_lower) / 2:
                    bull_score += 5
                    bull_args.append("Price in upper Bollinger Band half - bullish bias")
                else:
                    bear_score += 5
                    bear_args.append("Price in lower Bollinger Band half - bearish bias")

            # Support/Resistance
            support = self._extract_float(technical_result, "support", 0)
            resistance = self._extract_float(technical_result, "resistance", 0)
            
            if current_price and support and resistance:
                support_distance = abs(current_price - support) / current_price
                resistance_distance = abs(resistance - current_price) / current_price
                
                if support_distance < 0.02:  # Within 2% of support
                    bull_score += 15
                    bull_args.append(f"Price near strong support (${support:.2f})")
                elif support_distance < 0.05:  # Within 5% of support
                    bull_score += 8
                    bull_args.append(f"Price approaching support (${support:.2f})")
                    
                if resistance_distance < 0.02:  # Within 2% of resistance
                    bear_score += 15
                    bear_args.append(f"Price near strong resistance (${resistance:.2f})")
                elif resistance_distance < 0.05:  # Within 5% of resistance
                    bear_score += 8
                    bear_args.append(f"Price approaching resistance (${resistance:.2f})")

            # Technical Action and Confidence
            action = technical_result.get("action", "HOLD")
            confidence = self._extract_float(technical_result, "confidence", 50)
            
            if action == "BUY":
                bull_score += confidence * 0.8  # 80% weight to technical action
                bull_args.append(f"Technical BUY signal ({confidence:.1f}% confidence)")
            elif action == "SELL":
                bear_score += confidence * 0.8
                bear_args.append(f"Technical SELL signal ({confidence:.1f}% confidence)")

            # Additional signals
            signals = technical_result.get("signals", [])
            for signal in signals:
                signal_str = str(signal).upper()
                if any(word in signal_str for word in ["BULL", "BUY", "OVERSOLD", "STRONG"]):
                    bull_score += 5
                    bull_args.append(f"Technical indicator: {signal}")
                elif any(word in signal_str for word in ["BEAR", "SELL", "OVERBOUGHT", "WEAK"]):
                    bear_score += 5
                    bear_args.append(f"Technical indicator: {signal}")

        except Exception as e:
            logger.error(f"Error processing technical signals: {e}")

        return bull_score, bear_score, bull_args, bear_args

    def _extract_risk_signals(self, risk_metrics: Dict, current_price: float) -> Tuple[float, List[str]]:
        """Extract and score risk analysis signals"""
        bull_score = 0.0
        bear_score = 0.0
        bull_args = []
        bear_args = []

        if not risk_metrics:
            return bull_score, bear_score, bull_args, bear_args

        try:
            # Risk Level Analysis
            risk_level = risk_metrics.get("risk_level", "MEDIUM")
            volatility = self._extract_float(risk_metrics, "volatility", 0)
            position_size = self._extract_float(risk_metrics, "position_size", 0)
            sharpe_ratio = self._extract_float(risk_metrics, "sharpe_ratio", 0)

            # Risk Level Scoring
            risk_level_scores = {
                "VERY_LOW": (15, 0),
                "LOW": (10, 0),
                "MEDIUM": (5, 5),
                "HIGH": (0, 10),
                "VERY_HIGH": (0, 15)
            }
            
            bull_risk, bear_risk = risk_level_scores.get(risk_level, (5, 5))
            bull_score += bull_risk
            bear_score += bear_risk
            
            if bull_risk > bear_risk:
                bull_args.append(f"Favorable risk level: {risk_level}")
            else:
                bear_args.append(f"Elevated risk level: {risk_level}")

            # Volatility Analysis
            if volatility > 0.35:  # Very high volatility
                bear_score += 12
                bear_args.append(f"Very high volatility ({volatility:.1%}) - increased risk")
            elif volatility > 0.25:  # High volatility
                bear_score += 8
                bear_args.append(f"High volatility ({volatility:.1%}) - caution advised")
            elif volatility < 0.15:  # Low volatility
                bull_score += 8
                bull_args.append(f"Low volatility ({volatility:.1%}) - stable conditions")
            elif volatility < 0.10:  # Very low volatility
                bull_score += 12
                bull_args.append(f"Very low volatility ({volatility:.1%}) - excellent conditions")

            # Position Size Analysis
            if position_size > 0.08:  # Large position size indicates confidence
                bull_score += 10
                bull_args.append(f"Large recommended position ({position_size:.1%}) - high conviction")
            elif position_size < 0.02:  # Small position size indicates caution
                bear_score += 5
                bear_args.append(f"Small recommended position ({position_size:.1%}) - low conviction")

            # Sharpe Ratio Analysis
            if sharpe_ratio > 1.0:
                bull_score += 10
                bull_args.append(f"Excellent risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
            elif sharpe_ratio > 0.5:
                bull_score += 5
                bull_args.append(f"Good risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
            elif sharpe_ratio < -0.5:
                bear_score += 10
                bear_args.append(f"Poor risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")

            # Stop Loss and Take Profit Analysis
            stop_loss = self._extract_float(risk_metrics, "stop_loss_price", 0)
            take_profit = self._extract_float(risk_metrics, "take_profit_price", 0)
            
            if current_price and stop_loss and take_profit:
                stop_loss_pct = abs(current_price - stop_loss) / current_price
                take_profit_pct = abs(take_profit - current_price) / current_price
                risk_reward_ratio = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0
                
                if risk_reward_ratio > 2.0:
                    bull_score += 8
                    bull_args.append(f"Favorable risk-reward ratio ({risk_reward_ratio:.1f}:1)")
                elif risk_reward_ratio < 1.0:
                    bear_score += 8
                    bear_args.append(f"Poor risk-reward ratio ({risk_reward_ratio:.1f}:1)")

        except Exception as e:
            logger.error(f"Error processing risk signals: {e}")

        return bull_score, bear_score, bull_args, bear_args

    def _calculate_price_action_signals(self, price_df: pd.DataFrame, current_price: float) -> Tuple[float, List[str]]:
        """Calculate price action and trend signals"""
        bull_score = 0.0
        bear_score = 0.0
        bull_args = []
        bear_args = []

        if price_df is None or len(price_df) < 20:
            return bull_score, bear_score, bull_args, bear_args

        try:
            close_prices = price_df["Close"].astype(float)
            
            # Moving Average Analysis
            if len(close_prices) >= 50:
                sma_20 = close_prices.tail(20).mean()
                sma_50 = close_prices.tail(50).mean()
                
                if current_price > sma_20 > sma_50:
                    bull_score += 15
                    bull_args.append("Strong uptrend: price above rising moving averages")
                elif current_price < sma_20 < sma_50:
                    bear_score += 15
                    bear_args.append("Strong downtrend: price below falling moving averages")
                elif current_price > sma_20 and sma_20 > sma_50:
                    bull_score += 8
                    bull_args.append("Price in established uptrend")
                elif current_price < sma_20 and sma_20 < sma_50:
                    bear_score += 8
                    bear_args.append("Price in established downtrend")

            # Recent momentum (5-day vs 20-day performance)
            if len(close_prices) >= 20:
                recent_5d = (current_price - close_prices.iloc[-5]) / close_prices.iloc[-5]
                recent_20d = (current_price - close_prices.iloc[-20]) / close_prices.iloc[-20]
                
                if recent_5d > 0.02 and recent_5d > recent_20d:  # Strong recent momentum
                    bull_score += 10
                    bull_args.append(f"Strong recent momentum (+{recent_5d:.1%} in 5 days)")
                elif recent_5d < -0.02 and recent_5d < recent_20d:  # Weak recent momentum
                    bear_score += 10
                    bear_args.append(f"Weak recent momentum ({recent_5d:.1%} in 5 days)")

            # Volume analysis (if available)
            if "Volume" in price_df.columns:
                volume = price_df["Volume"].astype(float)
                if len(volume) >= 10:
                    avg_volume = volume.tail(10).mean()
                    current_volume = volume.iloc[-1]
                    if current_volume > avg_volume * 1.5 and current_price > close_prices.iloc[-2]:
                        bull_score += 8
                        bull_args.append("High volume on up move - bullish confirmation")
                    elif current_volume > avg_volume * 1.5 and current_price < close_prices.iloc[-2]:
                        bear_score += 8
                        bear_args.append("High volume on down move - bearish confirmation")

        except Exception as e:
            logger.error(f"Error calculating price action signals: {e}")

        return bull_score, bear_score, bull_args, bear_args

    def _llm_summary(self, ticker: str, bull_args: List[str], bear_args: List[str], 
                    bull_strength: float, bear_strength: float, 
                    consensus_action: str, consensus_conf: float) -> str:
        """Generate LLM summary with fallback"""
        if not self.llm:
            return self._create_basic_summary(ticker, bull_args, bear_args, bull_strength, bear_strength, consensus_action, consensus_conf)

        prompt = f"""
You are a professional financial analyst providing a balanced debate summary.

Ticker: {ticker}

BULL CASE (Strength: {bull_strength:.1f}):
{chr(10).join(f'• {arg}' for arg in bull_args)}

BEAR CASE (Strength: {bear_strength:.1f}):
{chr(10).join(f'• {arg}' for arg in bear_args)}

CONSENSUS: {consensus_action} with {consensus_conf:.1f}% confidence

Write a comprehensive 150-200 word analysis that:
1. Summarizes the key bullish and bearish arguments
2. Explains the consensus recommendation
3. Highlights the most compelling evidence from both sides
4. Maintains professional, balanced tone

Use ONLY the information provided above. Do not invent additional data.
"""

        try:
            result = self.llm.ask(prompt)
            # Check if result is an error message
            if result.startswith("[LLM_ERROR]"):
                return self._create_basic_summary(ticker, bull_args, bear_args, bull_strength, bear_strength, consensus_action, consensus_conf)
            return str(result)
        except Exception as e:
            logger.warning(f"LLM summary failed: {e}")
            return self._create_basic_summary(ticker, bull_args, bear_args, bull_strength, bear_strength, consensus_action, consensus_conf)

    def _create_basic_summary(self, ticker: str, bull_args: List[str], bear_args: List[str],
                             bull_strength: float, bear_strength: float,
                             consensus_action: str, consensus_conf: float) -> str:
        """Create detailed fallback summary without LLM"""
        summary = f"COMPREHENSIVE ANALYSIS FOR {ticker.upper()}\n\n"
        
        summary += "BULLISH FACTORS:\n"
        if bull_args:
            for i, arg in enumerate(bull_args[:5], 1):
                summary += f"{i}. {arg}\n"
        else:
            summary += "No strong bullish signals detected\n"
        
        summary += f"\nBEARISH FACTORS:\n"
        if bear_args:
            for i, arg in enumerate(bear_args[:5], 1):
                summary += f"{i}. {arg}\n"
        else:
            summary += "No strong bearish signals detected\n"
        
        summary += f"\nSTRENGTH ANALYSIS:\n"
        summary += f"Bull Case Strength: {bull_strength:.1f}\n"
        summary += f"Bear Case Strength: {bear_strength:.1f}\n"
        
        summary += f"\nFINAL ASSESSMENT:\n"
        if consensus_action == "BUY":
            summary += f"STRONG BUY RECOMMENDATION with {consensus_conf:.1f}% confidence\n"
            summary += "Bullish factors significantly outweigh bearish concerns"
        elif consensus_action == "SELL":
            summary += f"STRONG SELL RECOMMENDATION with {consensus_conf:.1f}% confidence\n"
            summary += "Bearish factors significantly outweigh bullish arguments"
        else:
            summary += f"NEUTRAL/HOLD POSITION with {consensus_conf:.1f}% confidence\n"
            summary += "Market signals are mixed with no clear directional bias"
        
        return summary

    def debate(self, ticker: str, technical_result: Dict, risk_metrics: Dict, 
               price_data=None, sentiment_score: float = 50.0) -> Dict[str, Any]:
        """
        Main debate method that integrates technical, risk, and price analysis
        """
        # Extract price data
        price_df = price_data.get("df") if isinstance(price_data, dict) else price_data
        current_price = self._extract_float(technical_result, "latest_close", 0)

        # Get signals from all analysis types
        tech_bull, tech_bear, tech_bull_args, tech_bear_args = self._extract_technical_signals(technical_result)
        risk_bull, risk_bear, risk_bull_args, risk_bear_args = self._extract_risk_signals(risk_metrics, current_price)
        price_bull, price_bear, price_bull_args, price_bear_args = self._calculate_price_action_signals(price_df, current_price)

        # Combine all signals
        total_bull = tech_bull + risk_bull + price_bull
        total_bear = tech_bear + risk_bear + price_bear
        
        all_bull_args = tech_bull_args + risk_bull_args + price_bull_args
        all_bear_args = tech_bear_args + risk_bear_args + price_bear_args

        # Add sentiment if available
        if sentiment_score >= 70:
            total_bull += 10
            all_bull_args.append(f"Very positive market sentiment ({sentiment_score})")
        elif sentiment_score >= 60:
            total_bull += 5
            all_bull_args.append(f"Positive market sentiment ({sentiment_score})")
        elif sentiment_score <= 30:
            total_bear += 10
            all_bear_args.append(f"Very negative market sentiment ({sentiment_score})")
        elif sentiment_score <= 40:
            total_bear += 5
            all_bear_args.append(f"Negative market sentiment ({sentiment_score})")

        # Calculate consensus
        total_strength = total_bull + total_bear
        bull_pct = (total_bull / total_strength) * 100 if total_strength > 0 else 50
        bear_pct = (total_bear / total_strength) * 100 if total_strength > 0 else 50

        # Determine consensus action and confidence
        strength_diff = total_bull - total_bear
        
        if strength_diff > 25:
            action = "BUY"
            confidence = min(90, 60 + (strength_diff - 25) / 2)
        elif strength_diff > 10:
            action = "BUY"
            confidence = min(80, 55 + strength_diff / 2)
        elif strength_diff < -25:
            action = "SELL" 
            confidence = min(90, 60 + abs(strength_diff + 25) / 2)
        elif strength_diff < -10:
            action = "SELL"
            confidence = min(80, 55 + abs(strength_diff) / 2)
        else:
            action = "HOLD"
            confidence = max(40, 50 - abs(strength_diff) / 2)

        # Generate narrative
        narrative = self._llm_summary(
            ticker, all_bull_args, all_bear_args, total_bull, total_bear, action, confidence
        )

        return {
            "ticker": ticker,
            "analysis_timestamp": datetime.now().isoformat(),
            "bull_case": {
                "arguments": all_bull_args,
                "strength": round(total_bull, 1),
                "components": {
                    "technical": round(tech_bull, 1),
                    "risk": round(risk_bull, 1),
                    "price_action": round(price_bull, 1)
                }
            },
            "bear_case": {
                "arguments": all_bear_args,
                "strength": round(total_bear, 1),
                "components": {
                    "technical": round(tech_bear, 1),
                    "risk": round(risk_bear, 1),
                    "price_action": round(price_bear, 1)
                }
            },
            "consensus": {
                "action": action,
                "confidence": round(confidence, 1),
                "bull_pct": round(bull_pct, 1),
                "bear_pct": round(bear_pct, 1),
                "strength_difference": round(strength_diff, 1)
            },
            "narrative": narrative,
            "metadata": {
                "technical_indicators_used": len(tech_bull_args) + len(tech_bear_args),
                "risk_factors_considered": len(risk_bull_args) + len(risk_bear_args),
                "price_signals_evaluated": len(price_bull_args) + len(price_bear_args)
            }
        }