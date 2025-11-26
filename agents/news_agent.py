# agents/sentiment_agent.py - STREAMLINED WITH FinBERT ONLY

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)

# ==================== FinBERT INTEGRATION ====================

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    
    class FinBERTAnalyzer:
        """Professional FinBERT sentiment analyzer for financial news"""
        
        def __init__(self):
            logger.info("🔄 Loading FinBERT AI Model...")
            self.model_name = "ProsusAI/finbert"
            
            # Use GPU if available
            self.device = 0 if torch.cuda.is_available() else -1
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline for easy use
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
        def analyze_sentiment(self, text: str, title: str = "") -> Dict:
            """Analyze sentiment using FinBERT"""
            if not text or len(text.strip()) < 10:
                return self._default_sentiment()
            
            try:
                # Combine title and text for better context
                full_text = f"{title}. {text}" if title else text
                
                # Truncate if too long (FinBERT has 512 token limit)
                if len(full_text) > 2000:
                    full_text = full_text[:2000]
                
                # Get prediction
                result = self.classifier(full_text)[0]
                
                return {
                    'sentiment': result['label'].lower(),  # 'positive', 'negative', or 'neutral'
                    'confidence': result['score'],
                    'model': 'FinBERT'
                }
                
            except Exception as e:
                logger.error(f"FinBERT analysis failed: {e}")
                return self._default_sentiment()
        
        def _default_sentiment(self) -> Dict:
            """Return default neutral sentiment"""
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'model': 'FinBERT'
            }
    
    # Initialize FinBERT
    finbert_analyzer = FinBERTAnalyzer()
    ML_AVAILABLE = True
    
except Exception as e:
    logger.warning(f"FinBERT not available: {e}")
    ML_AVAILABLE = False
    finbert_analyzer = None

class SentimentAnalysisAgent:
    """AI-Powered Sentiment Analysis using FinBERT"""
    
    def __init__(self, use_ml: bool = True):
        self.sentiment_history = []
        self.use_ml = use_ml and ML_AVAILABLE
        self.finbert_analyzer = finbert_analyzer if self.use_ml else None
        
        if self.use_ml:
            logger.info("🤖 FinBERT AI model enabled for sentiment analysis")
        else:
            logger.warning("❌ FinBERT not available - agent cannot function without ML model")
    
    def analyze(self, 
                ticker: str, 
                news_data: List[Dict] = None,
                news_text: str = None) -> Dict:
        """
        Analyze sentiment from news data using FinBERT
        
        Args:
            ticker: Stock ticker
            news_data: List of news articles with 'title' and 'content'
            news_text: Single news text (alternative to news_data)
        
        Returns:
            Comprehensive sentiment analysis
        """
        
        if not ML_AVAILABLE:
            logger.error("FinBERT not available - cannot perform sentiment analysis")
            return self._error_result(ticker, "FinBERT model not available")
        
        if news_data is None:
            news_data = []
        
        if news_text:
            news_data = [{"title": "News", "content": news_text}]
        
        # Handle empty news_data
        if not news_data or len(news_data) == 0:
            logger.warning(f"No news data for {ticker}, returning neutral")
            return self._neutral_result(ticker)
        
        try:
            # Analyze each article with FinBERT
            article_sentiments = []
            for i, article in enumerate(news_data):
                try:
                    title = article.get('title', '')
                    content = article.get('content', '')
                    
                    # Skip empty articles
                    if not title and not content:
                        continue
                    
                    # Analyze with FinBERT
                    sentiment = self._analyze_with_finbert(title, content)
                    article_sentiments.append(sentiment)
                    
                    # Small delay to be respectful to the model
                    if i < len(news_data) - 1:
                        time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing article: {e}")
                    continue
            
            # If no articles parsed, return neutral
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
                "recommendation": self._sentiment_to_recommendation(aggregate['sentiment'], aggregate['score']),
                "analysis_method": "FinBERT AI",
                "model_used": "ProsusAI/finbert"
            }
            
            logger.info(f"Sentiment analysis for {ticker}: {aggregate['sentiment']} "
                       f"({aggregate['confidence']:.1f}%) - {len(article_sentiments)} articles analyzed")
            return result
        
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._error_result(ticker, str(e))
    
    def _analyze_with_finbert(self, title: str, content: str) -> Dict:
        """Analyze sentiment using FinBERT AI model"""
        analysis = self.finbert_analyzer.analyze_sentiment(content, title)
        
        # Convert FinBERT output to match our format with score
        sentiment_score = self._sentiment_to_score(analysis['sentiment'], analysis['confidence'])
        
        return {
            'sentiment': analysis['sentiment'],
            'score': sentiment_score,
            'confidence': analysis['confidence'] * 100,  # Convert to percentage
            'text_preview': title[:100] if title else content[:100],
            'model': analysis['model'],
            'raw_confidence': analysis['confidence']
        }
    
    def _sentiment_to_score(self, sentiment: str, confidence: float) -> float:
        """Convert sentiment and confidence to numerical score (-1 to +1)"""
        base_scores = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0
        }
        
        base_score = base_scores.get(sentiment, 0.0)
        # Adjust score by confidence
        return base_score * confidence
    
    def _aggregate_sentiments(self, article_sentiments: List[Dict]) -> Dict:
        """Aggregate multiple article sentiments"""
        
        if not article_sentiments:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 50.0}
        
        # Calculate weighted average score (weighted by confidence)
        weighted_scores = []
        confidences = []
        
        for sentiment in article_sentiments:
            weighted_score = sentiment['score'] * (sentiment['confidence'] / 100)
            weighted_scores.append(weighted_score)
            confidences.append(sentiment['confidence'])
        
        avg_score = np.mean(weighted_scores) if weighted_scores else 0.0
        avg_confidence = np.mean(confidences) if confidences else 50.0
        
        # Determine overall sentiment based on average score
        if avg_score > 0.1:
            overall_sentiment = 'positive'
        elif avg_score < -0.1:
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
        
        if sentiment == "POSITIVE":
            if confidence > 80:
                strength = "STRONGLY POSITIVE"
                description = "Bullish market sentiment with high confidence"
            elif confidence > 60:
                strength = "POSITIVE" 
                description = "Bullish outlook with good confidence"
            else:
                strength = "SLIGHTLY POSITIVE"
                description = "Mild bullish sentiment"
        
        elif sentiment == "NEGATIVE":
            if confidence > 80:
                strength = "STRONGLY NEGATIVE"
                description = "Bearish market sentiment with high confidence"
            elif confidence > 60:
                strength = "NEGATIVE"
                description = "Bearish outlook with good confidence"
            else:
                strength = "SLIGHTLY NEGATIVE"
                description = "Mild bearish sentiment"
        
        else:  # NEUTRAL
            strength = "NEUTRAL"
            description = "Mixed or balanced market sentiment"
        
        return f"{score_emoji} **{strength}**: {description}. {article_count} articles analyzed with {confidence:.1f}% confidence."
    
    def _sentiment_to_recommendation(self, sentiment: str, score: float) -> Dict:
        """Convert sentiment to trading recommendation"""
        
        if sentiment == "positive":
            if score > 0.5:
                return {"action": "STRONG_BUY", "confidence": min(95, (score + 1) * 45)}
            elif score > 0.2:
                return {"action": "BUY", "confidence": min(85, (score + 1) * 40)}
            else:
                return {"action": "HOLD", "confidence": 65}
        
        elif sentiment == "negative":
            if score < -0.5:
                return {"action": "STRONG_SELL", "confidence": min(95, (abs(score) + 1) * 45)}
            elif score < -0.2:
                return {"action": "SELL", "confidence": min(85, (abs(score) + 1) * 40)}
            else:
                return {"action": "HOLD", "confidence": 65}
        
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
            "recommendation": {"action": "HOLD", "confidence": 50},
            "analysis_method": "FinBERT AI",
            "model_used": "ProsusAI/finbert"
        }
    
    def _error_result(self, ticker: str, error_msg: str) -> Dict:
        """Return error result"""
        return {
            "ticker": ticker,
            "overall_sentiment": "error",
            "overall_score": 0.0,
            "overall_confidence": 0.0,
            "article_count": 0,
            "positive_articles": 0,
            "negative_articles": 0,
            "neutral_articles": 0,
            "article_sentiments": [],
            "summary": f"Analysis failed: {error_msg}",
            "recommendation": {"action": "HOLD", "confidence": 50},
            "analysis_method": "FinBERT AI",
            "model_used": "ProsusAI/finbert",
            "error": error_msg
        }
    
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

# Usage example:
if __name__ == "__main__":
    # Initialize the agent
    agent = SentimentAnalysisAgent()
    
    # Example news data
    news_data = [
        {"title": "Company XYZ reports strong earnings growth", "content": "XYZ Corporation announced record profits..."},
        {"title": "Analysts upgrade XYZ stock to buy", "content": "Several analysts have raised their ratings..."}
    ]
    
    # Analyze sentiment
    result = agent.analyze("XYZ", news_data)
    print(f"Sentiment: {result['overall_sentiment']}")
    print(f"Score: {result['overall_score']}")
    print(f"Recommendation: {result['recommendation']}")