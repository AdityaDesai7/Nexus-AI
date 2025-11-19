# agent.py
import os
from groq import Groq
import json
from typing import Dict, List
import time

class GroqStockAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found!")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.1-70b-versatile"

        self.SOURCE_WEIGHTS = {
            'NewsAPI': 1.0, 'Reuters': 1.0, 'Bloomberg': 1.0,
            'Moneycontrol': 0.95, 'Economic Times': 0.9,
            'Yahoo Finance': 0.85, 'Google News': 0.8,
            'default': 0.7
        }

    def _call_llm(self, messages, max_retries=3):
        for i in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=0.3,
                    max_tokens=600,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                if i == max_retries - 1:
                    raise e
                time.sleep(2 ** i)
        return None

    def analyze_article_sentiment(self, title: str, content: str, source: str) -> Dict:
        prompt = f"""Analyze this stock news article.

Title: {title}
Source: {source}
Content: {content[:3000]}

Return EXACT JSON:
{{
    "sentiment": "POSITIVE|NEGATIVE|NEUTRAL",
    "confidence": 0.0-1.0,
    "reasoning": "2 sentence explanation",
    "key_points": ["fact1", "fact2"],
    "market_impact": "HIGH|MEDIUM|LOW",
    "investor_action": "BUY|SELL|HOLD|WATCH"
}}"""

        try:
            result = self._call_llm([
                {"role": "system", "content": "You are a senior equity analyst."},
                {"role": "user", "content": prompt}
            ])
            result['source_weight'] = self.SOURCE_WEIGHTS.get(source, 0.7)
            return result
        except:
            return {
                "sentiment": "NEUTRAL", "confidence": 0.0, "reasoning": "Failed",
                "key_points": [], "market_impact": "LOW", "investor_action": "HOLD",
                "source_weight": 0.5
            }

    def aggregate_analysis(self, analyses: List[Dict]) -> Dict:
        if not analyses:
            return {"trading_signal": "HOLD", "recommendation": "No data"}

        summary = "\n".join([
            f"- {a['sentiment']} ({a['confidence']:.0%} conf, weight: {a['source_weight']:.2f}) {a['reasoning']}"
            for a in analyses[:12]
        ])

        prompt = f"""You are a hedge fund CIO. Synthesize these {len(analyses)} analyses into a final call.

{summary}

Return JSON:
{{
    "overall_sentiment": "BULLISH|BEARISH|NEUTRAL",
    "confidence_level": 0.0-1.0,
    "trading_signal": "STRONG BUY|BUY|HOLD|SELL|STRONG SELL",
    "risk_level": "HIGH|MEDIUM|LOW",
    "recommendation": "2-3 sentence actionable advice",
    "key_catalysts": ["cat1", "cat2"],
    "key_risks": ["risk1", "risk2"],
    "price_outlook": "SHORT_TERM_UP|SHORT_TERM_DOWN|SIDEWAYS"
}}"""

        try:
            final = self._call_llm([
                {"role": "system", "content": "You are a risk-aware CIO."},
                {"role": "user", "content": prompt}
            ])

            # Weighted confidence
            weighted_conf = sum(a['confidence'] * a['source_weight'] for a in analyses) / sum(a['source_weight'] for a in analyses)
            final['weighted_confidence'] = round(weighted_conf, 3)
            return final
        except:
            return {
                "overall_sentiment": "NEUTRAL",
                "trading_signal": "HOLD",
                "recommendation": "Insufficient reliable data for recommendation.",
                "weighted_confidence": 0.0
            }