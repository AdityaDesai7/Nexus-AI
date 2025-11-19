# agents/sentiment_agent.py - COMPLETE FIXED VERSION

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Comprehensive keyword dictionaries
POSITIVE_KEYWORDS = [
    # Growth & Performance
    'growth', 'surge', 'rally', 'bull', 'bullish', 'gains', 'profit', 'profitability',
    'revenue', 'earnings', 'beat', 'outperform', 'surge', 'bounce', 'recovery',
    'breakthrough', 'innovation', 'expansion', 'upgrade', 'upside', 'strength',
    'momentum', 'positive', 'optimistic', 'strong', 'robust',
    
    # Market actions
    'buy', 'buying', 'accumulation', 'short covering', 'squeeze', 'rally',
    'breakout', 'rebound', 'bounce back', 'surge higher',
    
    # Company performance
    'record', 'highest', 'beat', 'exceed', 'outperform', 'solid',
    'new high', 'all time high', 'improved', 'improving',
    'better than expected', 'top performer', 'leader', 'market leader',
    
    # Sentiment positive
    'confidence', 'optimism', 'opportunity', 'potential', 'promising',
    'encouraging', 'positive outlook', 'bright', 'upbeat', 'enthusiasm',
    
    # Deal & partnership
    'acquisition', 'deal', 'partnership', 'collaboration', 'joint venture',
    'merger', 'strategic alliance', 'investment', 'funding', 'capital raise',
    
    # Analyst actions
    'upgrade', 'initiates coverage', 'raises target', 'buy rating', 'outperform',
    'recommended', 'strong buy', 'target raise', 'increase'
]

NEGATIVE_KEYWORDS = [
    # Decline & Loss
    'decline', 'fall', 'crash', 'plunge', 'slump', 'drop', 'loss', 'deficit',
    'bearish', 'bear', 'downside', 'weakness', 'weak', 'deteriorate', 'deterioration',
    'downtrend', 'selloff', 'correction', 'drawdown', 'miss',
    
    # Market actions
    'sell', 'selling', 'distribution', 'exit', 'dump', 'short', 'short sellers',
    'short pressure', 'margin call', 'liquidation', 'capitulation',
    
    # Company performance
    'worst', 'underperform', 'underperformance', 'missed', 'failed', 'failure',
    'declining', 'falling', 'lower than expected', 'disappointing', 'poor',
    
    # Sentiment negative
    'fear', 'fearful', 'pessimistic', 'pessimism', 'concern', 'risk',
    'uncertainty', 'volatile', 'volatility', 'panic', 'anxiety', 'gloomy', 'bleak',
    
    # Problems
    'bankruptcy', 'bankrupt', 'insolvency', 'default', 'credit risk',
    'investigation', 'lawsuit', 'scandal', 'fraud', 'accounting', 'restatement',
    'regulatory', 'violation', 'fine', 'penalty', 'recall',
    
    # Analyst actions
    'downgrade', 'cut', 'lower target', 'sell rating', 'underperform',
    'reduce', 'strong sell', 'negative', 'target cut'
]

NEUTRAL_KEYWORDS = [
    'announced', 'report', 'data', 'information', 'statement', 'confirm',
    'update', 'guidance', 'forecast', 'expect', 'expected', 'analyst',
    'meeting', 'conference', 'event', 'news', 'today', 'yesterday'
]


class SentimentAnalysisAgent:
    """Autonomous sentiment analysis - NO LLM DEPENDENCY"""
    
    def __init__(self):
        self.sentiment_history = []
        self.keyword_weights = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0
        }
    
    def analyze(self, 
                ticker: str, 
                news_data: List[Dict] = None,
                news_text: str = None) -> Dict:
        """
        Analyze sentiment from news data
        
        Args:
            ticker: Stock ticker
            news_data: List of news articles with 'title' and 'content'
            news_text: Single news text (alternative to news_data)
        
        Returns:
            Comprehensive sentiment analysis
        """
        
        if news_data is None:
            news_data = []
        
        if news_text:
            news_data = [{"title": "News", "content": news_text}]
        
        # ✅ FIXED: Handle empty news_data
        if not news_data or len(news_data) == 0:
            logger.warning(f"No news data for {ticker}, returning neutral")
            return self._neutral_result(ticker)
        
        try:
            # Analyze each article
            article_sentiments = []
            for article in news_data:
                try:
                    title = article.get('title', '')
                    content = article.get('content', '')
                    
                    # Skip empty articles
                    if not title and not content:
                        continue
                    
                    sentiment = self._analyze_single_article(title, content)
                    article_sentiments.append(sentiment)
                
                except Exception as e:
                    logger.warning(f"Error analyzing article: {e}")
                    continue
            
            # ✅ FIXED: If no articles parsed, return neutral
            if not article_sentiments:
                logger.warning(f"No articles parsed for {ticker}")
                return self._neutral_result(ticker)
            
            # Aggregate results
            aggregate = self._aggregate_sentiments(article_sentiments)
            
            # Generate summary
            summary = self._generate_summary(aggregate, len(article_sentiments))
            
            result = {
                "ticker": ticker,
                "overall_sentiment": aggregate['sentiment'],
                "overall_score": round(aggregate['score'], 3),
                "overall_confidence": round(aggregate['confidence'], 1),
                "article_count": len(article_sentiments),
                "positive_articles": sum(1 for s in article_sentiments if s['sentiment'] == 'positive'),
                "negative_articles": sum(1 for s in article_sentiments if s['sentiment'] == 'negative'),
                "neutral_articles": sum(1 for s in article_sentiments if s['sentiment'] == 'neutral'),
                "article_sentiments": article_sentiments,
                "summary": summary,
                "recommendation": self._sentiment_to_recommendation(aggregate['sentiment'], aggregate['score'])
            }
            
            logger.info(f"Sentiment analysis for {ticker}: {aggregate['sentiment']} ({aggregate['confidence']:.1f}%)")
            return result
        
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._neutral_result(ticker)
    
    def _analyze_single_article(self, title: str, content: str) -> Dict:
        """Analyze sentiment of a single article"""
        
        # Combine title and content (title weighted 2x)
        text = (title.lower() + " " + title.lower() + " " + content.lower())
        
        # Count keyword occurrences
        positive_count = self._count_keywords(text, POSITIVE_KEYWORDS)
        negative_count = self._count_keywords(text, NEGATIVE_KEYWORDS)
        neutral_count = self._count_keywords(text, NEUTRAL_KEYWORDS)
        
        total_keywords = positive_count + negative_count + neutral_count
        
        # ✅ FIXED: Better handling of low keyword counts
        if total_keywords == 0:
            # Fallback: use word length and structure as proxy
            if len(title) > 50 or len(content) > 200:
                return {
                    'sentiment': 'neutral',
                    'score': 0.0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'confidence': 30.0,
                    'text_preview': title[:100] if title else content[:100]
                }
            else:
                return {
                    'sentiment': 'neutral',
                    'score': 0.0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'confidence': 20.0,
                    'text_preview': title[:100] if title else content[:100]
                }
        
        # Calculate score: (positive - negative) / total
        score = (positive_count - negative_count) / total_keywords
        
        # Determine sentiment
        if score > 0.2:
            sentiment = 'positive'
        elif score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Confidence: how strongly keywords lean one way
        confidence = min(100, abs(score) * 100)
        
        return {
            'sentiment': sentiment,
            'score': round(score, 3),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'confidence': round(confidence, 1),
            'text_preview': title[:100] if title else content[:100]
        }
    
    def _count_keywords(self, text: str, keyword_list: List[str]) -> int:
        """Count occurrences of keywords in text"""
        count = 0
        for keyword in keyword_list:
            # Split text into words
            words = text.split()
            # Count keyword occurrences
            count += words.count(keyword)
        return count
    
    def _aggregate_sentiments(self, article_sentiments: List[Dict]) -> Dict:
        """Aggregate multiple article sentiments"""
        
        if not article_sentiments:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 50.0}
        
        # ✅ FIXED: Better aggregation
        # Calculate average score (weighted by confidence)
        weighted_scores = []
        confidences = []
        
        for s in article_sentiments:
            weighted_score = s['score'] * (s['confidence'] / 100)
            weighted_scores.append(weighted_score)
            confidences.append(s['confidence'])
        
        avg_score = np.mean(weighted_scores) if weighted_scores else 0.0
        avg_confidence = np.mean(confidences) if confidences else 50.0
        
        # Determine overall sentiment
        if avg_score > 0.2:
            overall_sentiment = 'positive'
        elif avg_score < -0.2:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'sentiment': overall_sentiment,
            'score': avg_score,
            'confidence': avg_confidence
        }
    
    def _generate_summary(self, aggregate: Dict, article_count: int) -> str:
        """Generate human-readable sentiment summary"""
        
        sentiment = aggregate['sentiment'].upper()
        score = aggregate['score']
        confidence = aggregate['confidence']
        
        score_emoji = "📈" if score > 0.1 else "📉" if score < -0.1 else "➡️"
        
        # ✅ FIXED: Better summary generation
        if sentiment == "POSITIVE":
            if confidence > 80:
                summary = f"{score_emoji} **STRONGLY POSITIVE**: Bullish market sentiment. {article_count} articles with strong positive signals detected."
            elif confidence > 60:
                summary = f"{score_emoji} **POSITIVE**: Bullish outlook. {article_count} articles show positive bias overall."
            else:
                summary = f"{score_emoji} **SLIGHTLY POSITIVE**: Mild bullish sentiment. {article_count} articles analyzed."
        
        elif sentiment == "NEGATIVE":
            if confidence > 80:
                summary = f"{score_emoji} **STRONGLY NEGATIVE**: Bearish market sentiment. {article_count} articles with strong negative signals detected."
            elif confidence > 60:
                summary = f"{score_emoji} **NEGATIVE**: Bearish outlook. {article_count} articles show negative bias overall."
            else:
                summary = f"{score_emoji} **SLIGHTLY NEGATIVE**: Mild bearish sentiment. {article_count} articles analyzed."
        
        else:  # NEUTRAL
            summary = f"{score_emoji} **NEUTRAL**: Mixed market sentiment. {article_count} articles with balanced signals - no clear trend."
        
        return summary
    
    def _sentiment_to_recommendation(self, sentiment: str, score: float) -> Dict:
        """Convert sentiment to trading recommendation"""
        
        if sentiment == "positive":
            if score > 0.5:
                return {"action": "STRONG_BUY", "confidence": min(90, (score + 1) * 45)}
            elif score > 0.2:
                return {"action": "BUY", "confidence": min(85, (score + 1) * 40)}
            else:
                return {"action": "HOLD", "confidence": 60}
        
        elif sentiment == "negative":
            if score < -0.5:
                return {"action": "STRONG_SELL", "confidence": min(90, (abs(score) + 1) * 45)}
            elif score < -0.2:
                return {"action": "SELL", "confidence": min(85, (abs(score) + 1) * 40)}
            else:
                return {"action": "HOLD", "confidence": 60}
        
        else:
            return {"action": "HOLD", "confidence": 50}
    
    def _neutral_result(self, ticker: str) -> Dict:
        """Return neutral result when no news"""
        
        return {
            "ticker": ticker,
            "overall_sentiment": "neutral",
            "overall_score": 0.0,
            "overall_confidence": 0.0,
            "article_count": 0,
            "positive_articles": 0,
            "negative_articles": 0,
            "neutral_articles": 0,
            "article_sentiments": [],
            "summary": "No news data available for analysis",
            "recommendation": {"action": "HOLD", "confidence": 50}
        }
    
    def get_sentiment_score(self, sentiment: str) -> float:
        """Convert sentiment string to numerical score (-1 to +1)"""
        mapping = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0,
            'strongly_positive': 0.8,
            'strongly_negative': -0.8
        }
        return mapping.get(sentiment.lower(), 0.0)
    
    def get_sentiment_breakdown(self, analysis_result: Dict) -> Dict:
        """Get detailed breakdown of sentiment"""
        
        total = analysis_result['article_count']
        
        if total == 0:
            return {
                'positive_pct': 0,
                'negative_pct': 0,
                'neutral_pct': 0,
                'dominant_sentiment': 'none'
            }
        
        pos_pct = (analysis_result['positive_articles'] / total) * 100
        neg_pct = (analysis_result['negative_articles'] / total) * 100
        neu_pct = (analysis_result['neutral_articles'] / total) * 100
        
        return {
            'positive_pct': round(pos_pct, 1),
            'negative_pct': round(neg_pct, 1),
            'neutral_pct': round(neu_pct, 1),
            'dominant_sentiment': analysis_result['overall_sentiment']
        }
