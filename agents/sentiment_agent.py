# # agents/sentiment_agent.py - STREAMLINED WITH FinBERT ONLY

# import pandas as pd
# import numpy as np
# from typing import Dict, List, Optional
# import logging
# import time

# logger = logging.getLogger(__name__)

# # ==================== FinBERT INTEGRATION ====================

# try:
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
#     import torch
    
#     class FinBERTAnalyzer:
#         """Professional FinBERT sentiment analyzer for financial news"""
        
#         def __init__(self):
#             logger.info("🔄 Loading FinBERT AI Model...")
#             self.model_name = "ProsusAI/finbert"
            
#             # Use GPU if available
#             self.device = 0 if torch.cuda.is_available() else -1
            
#             # Load model and tokenizer
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#             self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
#             # Create pipeline for easy use
#             self.classifier = pipeline(
#                 "sentiment-analysis",
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 device=self.device
#             )
            
#         def analyze_sentiment(self, text: str, title: str = "") -> Dict:
#             """Analyze sentiment using FinBERT"""
#             if not text or len(text.strip()) < 10:
#                 return self._default_sentiment()
            
#             try:
#                 # Combine title and text for better context
#                 full_text = f"{title}. {text}" if title else text
                
#                 # Truncate if too long (FinBERT has 512 token limit)
#                 if len(full_text) > 2000:
#                     full_text = full_text[:2000]
                
#                 # Get prediction
#                 result = self.classifier(full_text)[0]
                
#                 return {
#                     'sentiment': result['label'].lower(),  # 'positive', 'negative', or 'neutral'
#                     'confidence': result['score'],
#                     'model': 'FinBERT'
#                 }
                
#             except Exception as e:
#                 logger.error(f"FinBERT analysis failed: {e}")
#                 return self._default_sentiment()
        
#         def _default_sentiment(self) -> Dict:
#             """Return default neutral sentiment"""
#             return {
#                 'sentiment': 'neutral',
#                 'confidence': 0.5,
#                 'model': 'FinBERT'
#             }
    
#     # Initialize FinBERT
#     finbert_analyzer = FinBERTAnalyzer()
#     ML_AVAILABLE = True
    
# except Exception as e:
#     logger.warning(f"FinBERT not available: {e}")
#     ML_AVAILABLE = False
#     finbert_analyzer = None

# class SentimentAnalysisAgent:
#     """AI-Powered Sentiment Analysis using FinBERT"""
    
#     def __init__(self, use_ml: bool = True):
#         self.sentiment_history = []
#         self.use_ml = use_ml and ML_AVAILABLE
#         self.finbert_analyzer = finbert_analyzer if self.use_ml else None
        
#         if self.use_ml:
#             logger.info("🤖 FinBERT AI model enabled for sentiment analysis")
#         else:
#             logger.warning("❌ FinBERT not available - agent cannot function without ML model")
    
#     def analyze(self, 
#                 ticker: str, 
#                 news_data: List[Dict] = None,
#                 news_text: str = None) -> Dict:
#         """
#         Analyze sentiment from news data using FinBERT
        
#         Args:
#             ticker: Stock ticker
#             news_data: List of news articles with 'title' and 'content'
#             news_text: Single news text (alternative to news_data)
        
#         Returns:
#             Comprehensive sentiment analysis
#         """
        
#         if not ML_AVAILABLE:
#             logger.error("FinBERT not available - cannot perform sentiment analysis")
#             return self._error_result(ticker, "FinBERT model not available")
        
#         if news_data is None:
#             news_data = []
        
#         if news_text:
#             news_data = [{"title": "News", "content": news_text}]
        
#         # Handle empty news_data
#         if not news_data or len(news_data) == 0:
#             logger.warning(f"No news data for {ticker}, returning neutral")
#             return self._neutral_result(ticker)
        
#         try:
#             # Analyze each article with FinBERT
#             article_sentiments = []
#             for i, article in enumerate(news_data):
#                 try:
#                     title = article.get('title', '')
#                     content = article.get('content', '')
                    
#                     # Skip empty articles
#                     if not title and not content:
#                         continue
                    
#                     # Analyze with FinBERT
#                     sentiment = self._analyze_with_finbert(title, content)
#                     article_sentiments.append(sentiment)
                    
#                     # Small delay to be respectful to the model
#                     if i < len(news_data) - 1:
#                         time.sleep(0.1)
                    
#                 except Exception as e:
#                     logger.warning(f"Error analyzing article: {e}")
#                     continue
            
#             # If no articles parsed, return neutral
#             if not article_sentiments:
#                 logger.warning(f"No articles parsed for {ticker}")
#                 return self._neutral_result(ticker)
            
#             # Aggregate results
#             aggregate = self._aggregate_sentiments(article_sentiments)
            
#             # Generate summary
#             summary = self._generate_summary(aggregate, len(article_sentiments))
            
#             result = {
#                 "ticker": ticker,
#                 "overall_sentiment": aggregate['sentiment'],
#                 "overall_score": round(aggregate['score'], 3),
#                 "overall_confidence": round(aggregate['confidence'], 1),
#                 "article_count": len(article_sentiments),
#                 "positive_articles": sum(1 for s in article_sentiments if s['sentiment'] == 'positive'),
#                 "negative_articles": sum(1 for s in article_sentiments if s['sentiment'] == 'negative'),
#                 "neutral_articles": sum(1 for s in article_sentiments if s['sentiment'] == 'neutral'),
#                 "article_sentiments": article_sentiments,
#                 "summary": summary,
#                 "recommendation": self._sentiment_to_recommendation(aggregate['sentiment'], aggregate['score']),
#                 "analysis_method": "FinBERT AI",
#                 "model_used": "ProsusAI/finbert"
#             }
            
#             logger.info(f"Sentiment analysis for {ticker}: {aggregate['sentiment']} "
#                        f"({aggregate['confidence']:.1f}%) - {len(article_sentiments)} articles analyzed")
#             return result
        
#         except Exception as e:
#             logger.error(f"Error in sentiment analysis: {e}")
#             return self._error_result(ticker, str(e))
    
#     def _analyze_with_finbert(self, title: str, content: str) -> Dict:
#         """Analyze sentiment using FinBERT AI model"""
#         analysis = self.finbert_analyzer.analyze_sentiment(content, title)
        
#         # Convert FinBERT output to match our format with score
#         sentiment_score = self._sentiment_to_score(analysis['sentiment'], analysis['confidence'])
        
#         return {
#             'sentiment': analysis['sentiment'],
#             'score': sentiment_score,
#             'confidence': analysis['confidence'] * 100,  # Convert to percentage
#             'text_preview': title[:100] if title else content[:100],
#             'model': analysis['model'],
#             'raw_confidence': analysis['confidence']
#         }
    
#     def _sentiment_to_score(self, sentiment: str, confidence: float) -> float:
#         """Convert sentiment and confidence to numerical score (-1 to +1)"""
#         base_scores = {
#             'positive': 1.0,
#             'negative': -1.0,
#             'neutral': 0.0
#         }
        
#         base_score = base_scores.get(sentiment, 0.0)
#         # Adjust score by confidence
#         return base_score * confidence
    
#     def _aggregate_sentiments(self, article_sentiments: List[Dict]) -> Dict:
#         """Aggregate multiple article sentiments"""
        
#         if not article_sentiments:
#             return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 50.0}
        
#         # Calculate weighted average score (weighted by confidence)
#         weighted_scores = []
#         confidences = []
        
#         for sentiment in article_sentiments:
#             weighted_score = sentiment['score'] * (sentiment['confidence'] / 100)
#             weighted_scores.append(weighted_score)
#             confidences.append(sentiment['confidence'])
        
#         avg_score = np.mean(weighted_scores) if weighted_scores else 0.0
#         avg_confidence = np.mean(confidences) if confidences else 50.0
        
#         # Determine overall sentiment based on average score
#         if avg_score > 0.1:
#             overall_sentiment = 'positive'
#         elif avg_score < -0.1:
#             overall_sentiment = 'negative'
#         else:
#             overall_sentiment = 'neutral'
        
#         return {
#             'sentiment': overall_sentiment,
#             'score': avg_score,
#             'confidence': avg_confidence
#         }
    
#     def _generate_summary(self, aggregate: Dict, article_count: int) -> str:
#         """Generate human-readable sentiment summary"""
        
#         sentiment = aggregate['sentiment'].upper()
#         score = aggregate['score']
#         confidence = aggregate['confidence']
        
#         score_emoji = "📈" if score > 0.1 else "📉" if score < -0.1 else "➡️"
        
#         if sentiment == "POSITIVE":
#             if confidence > 80:
#                 strength = "STRONGLY POSITIVE"
#                 description = "Bullish market sentiment with high confidence"
#             elif confidence > 60:
#                 strength = "POSITIVE" 
#                 description = "Bullish outlook with good confidence"
#             else:
#                 strength = "SLIGHTLY POSITIVE"
#                 description = "Mild bullish sentiment"
        
#         elif sentiment == "NEGATIVE":
#             if confidence > 80:
#                 strength = "STRONGLY NEGATIVE"
#                 description = "Bearish market sentiment with high confidence"
#             elif confidence > 60:
#                 strength = "NEGATIVE"
#                 description = "Bearish outlook with good confidence"
#             else:
#                 strength = "SLIGHTLY NEGATIVE"
#                 description = "Mild bearish sentiment"
        
#         else:  # NEUTRAL
#             strength = "NEUTRAL"
#             description = "Mixed or balanced market sentiment"
        
#         return f"{score_emoji} **{strength}**: {description}. {article_count} articles analyzed with {confidence:.1f}% confidence."
    
#     def _sentiment_to_recommendation(self, sentiment: str, score: float) -> Dict:
#         """Convert sentiment to trading recommendation"""
        
#         if sentiment == "positive":
#             if score > 0.5:
#                 return {"action": "STRONG_BUY", "confidence": min(95, (score + 1) * 45)}
#             elif score > 0.2:
#                 return {"action": "BUY", "confidence": min(85, (score + 1) * 40)}
#             else:
#                 return {"action": "HOLD", "confidence": 65}
        
#         elif sentiment == "negative":
#             if score < -0.5:
#                 return {"action": "STRONG_SELL", "confidence": min(95, (abs(score) + 1) * 45)}
#             elif score < -0.2:
#                 return {"action": "SELL", "confidence": min(85, (abs(score) + 1) * 40)}
#             else:
#                 return {"action": "HOLD", "confidence": 65}
        
#         else:
#             return {"action": "HOLD", "confidence": 50}
    
#     def _neutral_result(self, ticker: str) -> Dict:
#         """Return neutral result when no news"""
#         return {
#             "ticker": ticker,
#             "overall_sentiment": "neutral",
#             "overall_score": 0.0,
#             "overall_confidence": 0.0,
#             "article_count": 0,
#             "positive_articles": 0,
#             "negative_articles": 0,
#             "neutral_articles": 0,
#             "article_sentiments": [],
#             "summary": "No news data available for analysis",
#             "recommendation": {"action": "HOLD", "confidence": 50},
#             "analysis_method": "FinBERT AI",
#             "model_used": "ProsusAI/finbert"
#         }
    
#     def _error_result(self, ticker: str, error_msg: str) -> Dict:
#         """Return error result"""
#         return {
#             "ticker": ticker,
#             "overall_sentiment": "error",
#             "overall_score": 0.0,
#             "overall_confidence": 0.0,
#             "article_count": 0,
#             "positive_articles": 0,
#             "negative_articles": 0,
#             "neutral_articles": 0,
#             "article_sentiments": [],
#             "summary": f"Analysis failed: {error_msg}",
#             "recommendation": {"action": "HOLD", "confidence": 50},
#             "analysis_method": "FinBERT AI",
#             "model_used": "ProsusAI/finbert",
#             "error": error_msg
#         }
    
#     def get_sentiment_breakdown(self, analysis_result: Dict) -> Dict:
#         """Get detailed breakdown of sentiment"""
        
#         total = analysis_result['article_count']
        
#         if total == 0:
#             return {
#                 'positive_pct': 0,
#                 'negative_pct': 0,
#                 'neutral_pct': 0,
#                 'dominant_sentiment': 'none'
#             }
        
#         pos_pct = (analysis_result['positive_articles'] / total) * 100
#         neg_pct = (analysis_result['negative_articles'] / total) * 100
#         neu_pct = (analysis_result['neutral_articles'] / total) * 100
        
#         return {
#             'positive_pct': round(pos_pct, 1),
#             'negative_pct': round(neg_pct, 1),
#             'neutral_pct': round(neu_pct, 1),
#             'dominant_sentiment': analysis_result['overall_sentiment']
#         }

# # Usage example:
# if __name__ == "__main__":
#     # Initialize the agent
#     agent = SentimentAnalysisAgent()
    
#     # Example news data
#     news_data = [
#         {"title": "Company XYZ reports strong earnings growth", "content": "XYZ Corporation announced record profits..."},
#         {"title": "Analysts upgrade XYZ stock to buy", "content": "Several analysts have raised their ratings..."}
#     ]
    
#     # Analyze sentiment
#     result = agent.analyze("XYZ", news_data)
#     print(f"Sentiment: {result['overall_sentiment']}")
#     print(f"Score: {result['overall_score']}")
#     print(f"Recommendation: {result['recommendation']}")

# # agents/sentiment_agent.py - STREAMLINED WITH FinBERT ONLY + YFINANCE + WEB SCRAPING

# import pandas as pd
# import numpy as np
# from typing import Dict, List, Optional
# import logging
# import time
# import yfinance as yf
# import requests
# from datetime import datetime, timedelta
# from bs4 import BeautifulSoup
# import re
# import random

# logger = logging.getLogger(__name__)

# # ==================== YFINANCE + WEB SCRAPING NEWS FETCHING ====================

# class NewsFetcher:
#     """Fetch financial news using yfinance + web scraping fallback"""
    
#     def __init__(self):
#         self.source = "Yahoo Finance (yfinance) + Web Scraping"
#         self.headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#         }
        
#     def fetch_news_yfinance(self, ticker: str) -> List[Dict]:
#         """Fetch news using yfinance library"""
#         try:
#             logger.info(f"📰 Fetching news for {ticker} using yfinance...")
            
#             # Create ticker object
#             stock = yf.Ticker(ticker)
            
#             # Get news using yfinance
#             news = stock.news
            
#             if not news:
#                 logger.warning(f"No news found for {ticker} via yfinance")
#                 return []
            
#             articles = []
#             for news_item in news:
#                 try:
#                     title = news_item.get('title', '')
#                     link = news_item.get('link', '')
#                     publisher = news_item.get('publisher', '')
#                     published_date = news_item.get('providerPublishTime', '')
                    
#                     # Convert timestamp to readable date if available
#                     if published_date:
#                         published_date = datetime.fromtimestamp(published_date).strftime('%Y-%m-%d %H:%M:%S')
                    
#                     article = {
#                         'title': title,
#                         'content': title,  # Using title as content since full content isn't available
#                         'source': publisher or 'Yahoo Finance',
#                         'published': published_date,
#                         'url': link,
#                     }
                    
#                     # Only add meaningful articles
#                     if title and len(title) > 10:
#                         articles.append(article)
                        
#                 except Exception as e:
#                     logger.warning(f"Error processing news item: {e}")
#                     continue
            
#             logger.info(f"✅ Found {len(articles)} news articles from yfinance")
#             return articles
            
#         except Exception as e:
#             logger.error(f"Error fetching yfinance news for {ticker}: {e}")
#             return []
    
#     def fetch_news_moneycontrol(self, ticker: str) -> List[Dict]:
#         """Fetch news from Money Control (Indian stocks)"""
#         try:
#             # Remove .NS suffix for Indian stocks
#             base_ticker = ticker.replace('.NS', '')
            
#             # Money Control search URL
#             url = f"https://www.moneycontrol.com/news/tags/{base_ticker.lower()}.html"
            
#             response = requests.get(url, headers=self.headers, timeout=10)
            
#             if response.status_code != 200:
#                 return []
            
#             soup = BeautifulSoup(response.content, 'html.parser')
#             articles = []
            
#             # Find news items
#             news_items = soup.find_all('li', class_='clearfix')[:10]  # Limit to 10 articles
            
#             for item in news_items:
#                 try:
#                     title_elem = item.find('h2')
#                     link_elem = item.find('a')
#                     desc_elem = item.find('p')
                    
#                     if title_elem and link_elem:
#                         title = title_elem.get_text().strip()
#                         link = link_elem.get('href', '')
#                         description = desc_elem.get_text().strip() if desc_elem else title
                        
#                         # Make absolute URL
#                         if link and not link.startswith('http'):
#                             link = f"https://www.moneycontrol.com{link}"
                        
#                         article = {
#                             'title': title,
#                             'content': description,
#                             'source': 'Money Control',
#                             'published': '',
#                             'url': link
#                         }
                        
#                         if title and len(title) > 10:
#                             articles.append(article)
                            
#                 except Exception as e:
#                     continue
            
#             logger.info(f"✅ Found {len(articles)} news articles from Money Control")
#             return articles
            
#         except Exception as e:
#             logger.error(f"Error fetching Money Control news: {e}")
#             return []
    
#     def fetch_news_reuters(self, ticker: str) -> List[Dict]:
#         """Fetch news from Reuters"""
#         try:
#             # Remove .NS suffix
#             base_ticker = ticker.replace('.NS', '')
            
#             # Reuters search
#             url = f"https://www.reuters.com/search/news?blob={base_ticker}"
            
#             response = requests.get(url, headers=self.headers, timeout=10)
            
#             if response.status_code != 200:
#                 return []
            
#             soup = BeautifulSoup(response.content, 'html.parser')
#             articles = []
            
#             # Find news items (Reuters structure may vary)
#             news_items = soup.find_all('article', class_='story')[:10]
            
#             for item in news_items:
#                 try:
#                     title_elem = item.find('h3') or item.find('a')
#                     link_elem = item.find('a')
                    
#                     if title_elem and link_elem:
#                         title = title_elem.get_text().strip()
#                         link = link_elem.get('href', '')
                        
#                         # Make absolute URL
#                         if link and not link.startswith('http'):
#                             link = f"https://www.reuters.com{link}"
                        
#                         # Try to get description
#                         desc_elem = item.find('p')
#                         description = desc_elem.get_text().strip() if desc_elem else title
                        
#                         article = {
#                             'title': title,
#                             'content': description,
#                             'source': 'Reuters',
#                             'published': '',
#                             'url': link
#                         }
                        
#                         if title and len(title) > 10:
#                             articles.append(article)
                            
#                 except Exception as e:
#                     continue
            
#             logger.info(f"✅ Found {len(articles)} news articles from Reuters")
#             return articles
            
#         except Exception as e:
#             logger.error(f"Error fetching Reuters news: {e}")
#             return []
    
#     def fetch_news_bloomberg(self, ticker: str) -> List[Dict]:
#         """Fetch news from Bloomberg"""
#         try:
#             # Remove .NS suffix and format for Bloomberg
#             base_ticker = ticker.replace('.NS', '').replace('.', ':')
            
#             # Bloomberg search (simplified - may not work consistently)
#             url = f"https://www.bloomberg.com/search?query={base_ticker}"
            
#             response = requests.get(url, headers=self.headers, timeout=10)
            
#             if response.status_code != 200:
#                 return []
            
#             soup = BeautifulSoup(response.content, 'html.parser')
#             articles = []
            
#             # Bloomberg structure is complex, try multiple selectors
#             selectors = [
#                 'article[data-type="article"]',
#                 '.story-list-story',
#                 '.search-result-story'
#             ]
            
#             for selector in selectors:
#                 news_items = soup.select(selector)[:5]
#                 if news_items:
#                     break
            
#             for item in news_items[:5]:
#                 try:
#                     title_elem = item.find('h1') or item.find('h2') or item.find('h3')
#                     link_elem = item.find('a')
                    
#                     if title_elem and link_elem:
#                         title = title_elem.get_text().strip()
#                         link = link_elem.get('href', '')
                        
#                         # Make absolute URL
#                         if link and not link.startswith('http'):
#                             link = f"https://www.bloomberg.com{link}"
                        
#                         article = {
#                             'title': title,
#                             'content': title,  # Bloomberg content is usually behind paywall
#                             'source': 'Bloomberg',
#                             'published': '',
#                             'url': link
#                         }
                        
#                         if title and len(title) > 10:
#                             articles.append(article)
                            
#                 except Exception as e:
#                     continue
            
#             logger.info(f"✅ Found {len(articles)} news articles from Bloomberg")
#             return articles
            
#         except Exception as e:
#             logger.error(f"Error fetching Bloomberg news: {e}")
#             return []
    
#     def fetch_news_google_finance(self, ticker: str) -> List[Dict]:
#         """Fetch news from Google Finance"""
#         try:
#             # Format for Google Finance
#             formatted_ticker = ticker.replace('.NS', ':NSE')
            
#             url = f"https://www.google.com/finance/quote/{formatted_ticker}"
            
#             response = requests.get(url, headers=self.headers, timeout=10)
            
#             if response.status_code != 200:
#                 return []
            
#             soup = BeautifulSoup(response.content, 'html.parser')
#             articles = []
            
#             # Google Finance news section
#             news_container = soup.find('div', class_='news')
#             if news_container:
#                 news_items = news_container.find_all('div', class_='yY3Lee')[:10]
                
#                 for item in news_items:
#                     try:
#                         title_elem = item.find('div', class_='Yfwt5')
#                         link_elem = item.find('a')
                        
#                         if title_elem and link_elem:
#                             title = title_elem.get_text().strip()
#                             link = link_elem.get('href', '')
                            
#                             # Make absolute URL
#                             if link and not link.startswith('http'):
#                                 link = f"https://www.google.com{link}"
                            
#                             # Try to get snippet
#                             snippet_elem = item.find('div', class_='sfyJob')
#                             snippet = snippet_elem.get_text().strip() if snippet_elem else title
                            
#                             article = {
#                                 'title': title,
#                                 'content': snippet,
#                                 'source': 'Google Finance',
#                                 'published': '',
#                                 'url': link
#                             }
                            
#                             if title and len(title) > 10:
#                                 articles.append(article)
                                
#                     except Exception as e:
#                         continue
            
#             logger.info(f"✅ Found {len(articles)} news articles from Google Finance")
#             return articles
            
#         except Exception as e:
#             logger.error(f"Error fetching Google Finance news: {e}")
#             return []
    
#     def get_news(self, ticker: str, use_scraping: bool = True) -> List[Dict]:
#         """
#         Get news for ticker using yfinance + web scraping fallback
        
#         Args:
#             ticker: Stock ticker symbol
#             use_scraping: Whether to use web scraping if yfinance fails
#         """
#         all_articles = []
        
#         logger.info(f"📰 Fetching news for {ticker}...")
        
#         # Step 1: Try yfinance first
#         yfinance_articles = self.fetch_news_yfinance(ticker)
#         all_articles.extend(yfinance_articles)
        
#         # Step 2: If yfinance returns no articles and scraping is enabled, try web scraping
#         if (not yfinance_articles or len(yfinance_articles) == 0) and use_scraping:
#             logger.info("🔄 yfinance returned no articles, trying web scraping...")
            
#             # Try multiple trusted financial news sources
#             scraping_sources = [
#                 self.fetch_news_moneycontrol,
#                 self.fetch_news_google_finance,
#                 self.fetch_news_reuters,
#             ]
            
#             for source_func in scraping_sources:
#                 try:
#                     articles = source_func(ticker)
#                     if articles:
#                         all_articles.extend(articles)
#                         logger.info(f"✅ Found {len(articles)} articles from {source_func.__name__}")
#                         # Don't break, try to get from multiple sources
#                 except Exception as e:
#                     logger.warning(f"Scraping failed for {source_func.__name__}: {e}")
#                     continue
        
#         # Remove duplicates based on title similarity
#         unique_articles = self._remove_duplicate_articles(all_articles)
        
#         logger.info(f"📊 Total unique articles found: {len(unique_articles)}")
#         return unique_articles
    
#     def _remove_duplicate_articles(self, articles: List[Dict]) -> List[Dict]:
#         """Remove duplicate articles based on title similarity"""
#         unique_articles = []
#         seen_titles = set()
        
#         for article in articles:
#             title = article['title'].lower().strip()
            
#             # Create a simplified version for comparison
#             simple_title = re.sub(r'[^a-zA-Z0-9]', '', title)
            
#             if (title and len(title) > 10 and 
#                 simple_title not in seen_titles and
#                 len(simple_title) > 5):
                
#                 seen_titles.add(simple_title)
#                 unique_articles.append(article)
        
#         return unique_articles

# # ==================== FinBERT INTEGRATION ====================

# try:
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
#     import torch
    
#     class FinBERTAnalyzer:
#         """Professional FinBERT sentiment analyzer for financial news"""
        
#         def __init__(self):
#             logger.info("🔄 Loading FinBERT AI Model...")
#             self.model_name = "ProsusAI/finbert"
            
#             # Use GPU if available
#             self.device = 0 if torch.cuda.is_available() else -1
            
#             # Load model and tokenizer
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#             self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
#             # Create pipeline for easy use
#             self.classifier = pipeline(
#                 "sentiment-analysis",
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 device=self.device
#             )
            
#         def analyze_sentiment(self, text: str, title: str = "") -> Dict:
#             """Analyze sentiment using FinBERT"""
#             if not text or len(text.strip()) < 10:
#                 return self._default_sentiment()
            
#             try:
#                 # Combine title and text for better context
#                 full_text = f"{title}. {text}" if title else text
                
#                 # Truncate if too long (FinBERT has 512 token limit)
#                 if len(full_text) > 2000:
#                     full_text = full_text[:2000]
                
#                 # Get prediction
#                 result = self.classifier(full_text)[0]
                
#                 return {
#                     'sentiment': result['label'].lower(),  # 'positive', 'negative', or 'neutral'
#                     'confidence': result['score'],
#                     'model': 'FinBERT'
#                 }
                
#             except Exception as e:
#                 logger.error(f"FinBERT analysis failed: {e}")
#                 return self._default_sentiment()
        
#         def _default_sentiment(self) -> Dict:
#             """Return default neutral sentiment"""
#             return {
#                 'sentiment': 'neutral',
#                 'confidence': 0.5,
#                 'model': 'FinBERT'
#             }
    
#     # Initialize FinBERT
#     finbert_analyzer = FinBERTAnalyzer()
#     ML_AVAILABLE = True
    
# except Exception as e:
#     logger.warning(f"FinBERT not available: {e}")
#     ML_AVAILABLE = False
#     finbert_analyzer = None

# # ==================== ENHANCED SENTIMENT AGENT ====================

# class SentimentAnalysisAgent:
#     """AI-Powered Sentiment Analysis with Built-in News Fetching + Web Scraping"""
    
#     def __init__(self, use_ml: bool = True):
#         self.sentiment_history = []
#         self.use_ml = use_ml and ML_AVAILABLE
#         self.finbert_analyzer = finbert_analyzer if self.use_ml else None
        
#         # Initialize news fetcher with web scraping capability
#         self.news_fetcher = NewsFetcher()
        
#         if self.use_ml:
#             logger.info("🤖 FinBERT AI model enabled for sentiment analysis")
#         else:
#             logger.warning("❌ FinBERT not available - agent cannot function without ML model")
        
#         logger.info("📰 News fetcher initialized (yfinance + web scraping)")
    
#     def analyze(self, 
#                 ticker: str, 
#                 news_data: List[Dict] = None,
#                 news_text: str = None,
#                 fetch_news: bool = True,
#                 use_scraping: bool = True) -> Dict:
#         """
#         Analyze sentiment - can use provided news or fetch automatically
        
#         Args:
#             ticker: Stock ticker
#             news_data: Pre-fetched news articles
#             news_text: Single news text
#             fetch_news: Whether to automatically fetch news if none provided
#             use_scraping: Whether to use web scraping if yfinance fails
#         """
        
#         if not ML_AVAILABLE:
#             logger.error("FinBERT not available - cannot perform sentiment analysis")
#             return self._error_result(ticker, "FinBERT model not available")
        
#         # If no news data provided but fetch_news is True, fetch news automatically
#         if (not news_data or len(news_data) == 0) and fetch_news and not news_text:
#             logger.info(f"🔄 No news data provided, fetching news for {ticker}...")
#             news_data = self.news_fetcher.get_news(ticker, use_scraping=use_scraping)
        
#         if news_data is None:
#             news_data = []
        
#         if news_text:
#             news_data = [{"title": "News", "content": news_text}]
        
#         # Handle empty news_data
#         if not news_data or len(news_data) == 0:
#             logger.warning(f"No news data for {ticker}, returning neutral")
#             return self._neutral_result(ticker)
        
#         try:
#             # Analyze each article with FinBERT
#             article_sentiments = []
#             for i, article in enumerate(news_data):
#                 try:
#                     title = article.get('title', '')
#                     content = article.get('content', '')
                    
#                     # Skip empty articles
#                     if not title and not content:
#                         continue
                    
#                     # For news where content is same as title, use only title to avoid duplication
#                     if content == title:
#                         content = ""
                    
#                     # Analyze with FinBERT
#                     sentiment = self._analyze_with_finbert(title, content)
#                     article_sentiments.append(sentiment)
                    
#                     # Small delay to be respectful to the model
#                     if i < len(news_data) - 1:
#                         time.sleep(0.1)
                    
#                 except Exception as e:
#                     logger.warning(f"Error analyzing article: {e}")
#                     continue
            
#             # If no articles parsed, return neutral
#             if not article_sentiments:
#                 logger.warning(f"No articles parsed for {ticker}")
#                 return self._neutral_result(ticker)
            
#             # Aggregate results
#             aggregate = self._aggregate_sentiments(article_sentiments)
            
#             # Generate summary
#             summary = self._generate_summary(aggregate, len(article_sentiments))
            
#             result = {
#                 "ticker": ticker,
#                 "overall_sentiment": aggregate['sentiment'],
#                 "overall_score": round(aggregate['score'], 3),
#                 "overall_confidence": round(aggregate['confidence'], 1),
#                 "article_count": len(article_sentiments),
#                 "positive_articles": sum(1 for s in article_sentiments if s['sentiment'] == 'positive'),
#                 "negative_articles": sum(1 for s in article_sentiments if s['sentiment'] == 'negative'),
#                 "neutral_articles": sum(1 for s in article_sentiments if s['sentiment'] == 'neutral'),
#                 "article_sentiments": article_sentiments,
#                 "summary": summary,
#                 "recommendation": self._sentiment_to_recommendation(aggregate['sentiment'], aggregate['score']),
#                 "analysis_method": "FinBERT AI",
#                 "model_used": "ProsusAI/finbert",
#                 "news_source": "yfinance + web scraping"
#             }
            
#             logger.info(f"✅ Sentiment analysis for {ticker}: {aggregate['sentiment']} "
#                        f"({aggregate['confidence']:.1f}%) - {len(article_sentiments)} articles analyzed")
#             return result
        
#         except Exception as e:
#             logger.error(f"Error in sentiment analysis: {e}")
#             return self._error_result(ticker, str(e))
    
#     def _analyze_with_finbert(self, title: str, content: str) -> Dict:
#         """Analyze sentiment using FinBERT AI model"""
#         analysis = self.finbert_analyzer.analyze_sentiment(content, title)
        
#         # Convert FinBERT output to match our format with score
#         sentiment_score = self._sentiment_to_score(analysis['sentiment'], analysis['confidence'])
        
#         return {
#             'sentiment': analysis['sentiment'],
#             'score': sentiment_score,
#             'confidence': analysis['confidence'] * 100,  # Convert to percentage
#             'text_preview': title[:100] if title else content[:100],
#             'model': analysis['model'],
#             'raw_confidence': analysis['confidence']
#         }
    
#     def _sentiment_to_score(self, sentiment: str, confidence: float) -> float:
#         """Convert sentiment and confidence to numerical score (-1 to +1)"""
#         base_scores = {
#             'positive': 1.0,
#             'negative': -1.0,
#             'neutral': 0.0
#         }
        
#         base_score = base_scores.get(sentiment, 0.0)
#         # Adjust score by confidence
#         return base_score * confidence
    
#     def _aggregate_sentiments(self, article_sentiments: List[Dict]) -> Dict:
#         """Aggregate multiple article sentiments"""
        
#         if not article_sentiments:
#             return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 50.0}
        
#         # Calculate weighted average score (weighted by confidence)
#         weighted_scores = []
#         confidences = []
        
#         for sentiment in article_sentiments:
#             weighted_score = sentiment['score'] * (sentiment['confidence'] / 100)
#             weighted_scores.append(weighted_score)
#             confidences.append(sentiment['confidence'])
        
#         avg_score = np.mean(weighted_scores) if weighted_scores else 0.0
#         avg_confidence = np.mean(confidences) if confidences else 50.0
        
#         # Determine overall sentiment based on average score
#         if avg_score > 0.1:
#             overall_sentiment = 'positive'
#         elif avg_score < -0.1:
#             overall_sentiment = 'negative'
#         else:
#             overall_sentiment = 'neutral'
        
#         return {
#             'sentiment': overall_sentiment,
#             'score': avg_score,
#             'confidence': avg_confidence
#         }
    
#     def _generate_summary(self, aggregate: Dict, article_count: int) -> str:
#         """Generate human-readable sentiment summary"""
        
#         sentiment = aggregate['sentiment'].upper()
#         score = aggregate['score']
#         confidence = aggregate['confidence']
        
#         score_emoji = "📈" if score > 0.1 else "📉" if score < -0.1 else "➡️"
        
#         if sentiment == "POSITIVE":
#             if confidence > 80:
#                 strength = "STRONGLY POSITIVE"
#                 description = "Bullish market sentiment with high confidence"
#             elif confidence > 60:
#                 strength = "POSITIVE" 
#                 description = "Bullish outlook with good confidence"
#             else:
#                 strength = "SLIGHTLY POSITIVE"
#                 description = "Mild bullish sentiment"
        
#         elif sentiment == "NEGATIVE":
#             if confidence > 80:
#                 strength = "STRONGLY NEGATIVE"
#                 description = "Bearish market sentiment with high confidence"
#             elif confidence > 60:
#                 strength = "NEGATIVE"
#                 description = "Bearish outlook with good confidence"
#             else:
#                 strength = "SLIGHTLY NEGATIVE"
#                 description = "Mild bearish sentiment"
        
#         else:  # NEUTRAL
#             strength = "NEUTRAL"
#             description = "Mixed or balanced market sentiment"
        
#         return f"{score_emoji} **{strength}**: {description}. {article_count} articles analyzed with {confidence:.1f}% confidence."
    
#     def _sentiment_to_recommendation(self, sentiment: str, score: float) -> Dict:
#         """Convert sentiment to trading recommendation"""
        
#         if sentiment == "positive":
#             if score > 0.5:
#                 return {"action": "STRONG_BUY", "confidence": min(95, (score + 1) * 45)}
#             elif score > 0.2:
#                 return {"action": "BUY", "confidence": min(85, (score + 1) * 40)}
#             else:
#                 return {"action": "HOLD", "confidence": 65}
        
#         elif sentiment == "negative":
#             if score < -0.5:
#                 return {"action": "STRONG_SELL", "confidence": min(95, (abs(score) + 1) * 45)}
#             elif score < -0.2:
#                 return {"action": "SELL", "confidence": min(85, (abs(score) + 1) * 40)}
#             else:
#                 return {"action": "HOLD", "confidence": 65}
        
#         else:
#             return {"action": "HOLD", "confidence": 50}
    
#     def _neutral_result(self, ticker: str) -> Dict:
#         """Return neutral result when no news"""
#         return {
#             "ticker": ticker,
#             "overall_sentiment": "neutral",
#             "overall_score": 0.0,
#             "overall_confidence": 0.0,
#             "article_count": 0,
#             "positive_articles": 0,
#             "negative_articles": 0,
#             "neutral_articles": 0,
#             "article_sentiments": [],
#             "summary": "No news data available for analysis",
#             "recommendation": {"action": "HOLD", "confidence": 50},
#             "analysis_method": "FinBERT AI",
#             "model_used": "ProsusAI/finbert",
#             "news_source": "yfinance + web scraping"
#         }
    
#     def _error_result(self, ticker: str, error_msg: str) -> Dict:
#         """Return error result"""
#         return {
#             "ticker": ticker,
#             "overall_sentiment": "error",
#             "overall_score": 0.0,
#             "overall_confidence": 0.0,
#             "article_count": 0,
#             "positive_articles": 0,
#             "negative_articles": 0,
#             "neutral_articles": 0,
#             "article_sentiments": [],
#             "summary": f"Analysis failed: {error_msg}",
#             "recommendation": {"action": "HOLD", "confidence": 50},
#             "analysis_method": "FinBERT AI",
#             "model_used": "ProsusAI/finbert",
#             "news_source": "yfinance + web scraping",
#             "error": error_msg
#         }
    
#     def get_sentiment_breakdown(self, analysis_result: Dict) -> Dict:
#         """Get detailed breakdown of sentiment"""
        
#         total = analysis_result['article_count']
        
#         if total == 0:
#             return {
#                 'positive_pct': 0,
#                 'negative_pct': 0,
#                 'neutral_pct': 0,
#                 'dominant_sentiment': 'none'
#             }
        
#         pos_pct = (analysis_result['positive_articles'] / total) * 100
#         neg_pct = (analysis_result['negative_articles'] / total) * 100
#         neu_pct = (analysis_result['neutral_articles'] / total) * 100
        
#         return {
#             'positive_pct': round(pos_pct, 1),
#             'negative_pct': round(neg_pct, 1),
#             'neutral_pct': round(neu_pct, 1),
#             'dominant_sentiment': analysis_result['overall_sentiment']
#         }

# # Usage example:
# if __name__ == "__main__":
#     # Initialize the agent
#     agent = SentimentAnalysisAgent()
    
#     # Test with automatic news fetching + web scraping
#     print("🧪 Testing sentiment analysis with yfinance + web scraping...")
    
#     # Analyze NTPC.NS with automatic news fetching and web scraping fallback
#     result = agent.analyze("NTPC.NS", fetch_news=True, use_scraping=True)
    
#     print(f"Ticker: {result['ticker']}")
#     print(f"Overall Sentiment: {result['overall_sentiment']}")
#     print(f"Overall Score: {result['overall_score']}")
#     print(f"Confidence: {result['overall_confidence']}%")
#     print(f"Articles Analyzed: {result['article_count']}")
#     print(f"Summary: {result['summary']}")
#     print(f"Recommendation: {result['recommendation']}")
#     print(f"News Source: {result.get('news_source', 'N/A')}")
    
#     # Show breakdown if articles were found
#     if result['article_count'] > 0:
#         breakdown = agent.get_sentiment_breakdown(result)
#         print(f"Sentiment Breakdown: {breakdown}")
# agents/sentimental_agent.py - AI AGENT WITH GROQ LLM + FinBERT + NEWS
# agents/sentimental_agent.py - AI AGENT WITH GROQ LLM + FinBERT + NEWS
# agents/sentimental_agent.py - ENHANCED NEWS FETCHING

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import time
import yfinance as yf
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import json
import os

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# ==================== ENHANCED NEWS FETCHING ====================

class NewsFetcher:
    """Enhanced news fetcher with multiple reliable sources"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def fetch_news_yfinance_enhanced(self, ticker: str) -> List[Dict]:
        """Enhanced yfinance news fetching with multiple approaches"""
        try:
            logger.info(f"📰 Fetching news for {ticker} using enhanced yfinance...")
            
            stock = yf.Ticker(ticker)
            
            # Approach 1: Try direct news attribute
            news = getattr(stock, 'news', [])
            
            # Approach 2: Try getting info which might contain news
            if not news:
                try:
                    info = stock.info
                    # Some tickers have news in info
                    if 'companyNews' in info:
                        news = info['companyNews']
                except:
                    pass
            
            # Approach 3: Try alternative ticker format for Indian stocks
            if not news and ticker.endswith('.NS'):
                base_ticker = ticker.replace('.NS', '.BO')  # Try .BO format
                try:
                    alt_stock = yf.Ticker(base_ticker)
                    alt_news = getattr(alt_stock, 'news', [])
                    if alt_news:
                        news = alt_news
                        logger.info(f"✅ Found news using .BO format for {ticker}")
                except:
                    pass
            
            if not news:
                logger.warning(f"No news found for {ticker} via yfinance")
                return []
            
            articles = []
            for news_item in news[:15]:  # Limit to 15 articles
                try:
                    title = news_item.get('title', '') or news_item.get('headline', '')
                    link = news_item.get('link', '') or news_item.get('url', '')
                    publisher = news_item.get('publisher', '') or news_item.get('source', '')
                    
                    # Handle different timestamp formats
                    published_date = ''
                    for time_key in ['providerPublishTime', 'publishedAt', 'pubDate', 'timestamp']:
                        if time_key in news_item:
                            timestamp = news_item[time_key]
                            if timestamp:
                                if isinstance(timestamp, (int, float)) and timestamp > 1000000000:
                                    published_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                                    break
                    
                    # Get content - try multiple fields
                    content = title  # Default to title
                    for content_key in ['summary', 'description', 'content', 'text']:
                        if content_key in news_item and news_item[content_key]:
                            content = news_item[content_key]
                            break
                    
                    article = {
                        'title': title,
                        'content': content,
                        'source': publisher or 'Yahoo Finance',
                        'published': published_date,
                        'url': link,
                    }
                    
                    # Only add meaningful articles
                    if title and len(title) > 10:
                        articles.append(article)
                        
                except Exception as e:
                    logger.debug(f"Error processing news item: {e}")
                    continue
            
            logger.info(f"✅ Found {len(articles)} news articles from yfinance")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching enhanced yfinance news for {ticker}: {e}")
            return []
    
    def fetch_news_moneycontrol(self, ticker: str) -> List[Dict]:
        """Fetch news from Money Control for Indian stocks"""
        try:
            base_ticker = ticker.replace('.NS', '').replace('.BO', '')
            
            # Money Control search URL
            url = f"https://www.moneycontrol.com/rss/currentaffairs.xml"
            
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return []
            
            # Parse RSS feed
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')[:20]
            
            articles = []
            for item in items:
                try:
                    title = item.find('title').text if item.find('title') else ''
                    description = item.find('description').text if item.find('description') else ''
                    link = item.find('link').text if item.find('link') else ''
                    pub_date = item.find('pubDate').text if item.find('pubDate') else ''
                    
                    # Filter for relevant articles (contains ticker or company name)
                    if (base_ticker.lower() in title.lower() or 
                        base_ticker.lower() in description.lower() or
                        'infosys' in title.lower() or 'infosys' in description.lower()):
                        
                        article = {
                            'title': title,
                            'content': description,
                            'source': 'Money Control',
                            'published': pub_date,
                            'url': link
                        }
                        
                        if title and len(title) > 10:
                            articles.append(article)
                            
                except Exception as e:
                    continue
            
            logger.info(f"✅ Found {len(articles)} relevant articles from Money Control")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Money Control news: {e}")
            return []
    
    def fetch_news_google_rss(self, ticker: str) -> List[Dict]:
        """Fetch news from Google News RSS"""
        try:
            base_ticker = ticker.replace('.NS', '').replace('.BO', '')
            
            # Google News RSS
            url = f"https://news.google.com/rss/search?q={base_ticker}+stock&hl=en-IN&gl=IN&ceid=IN:en"
            
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')[:15]
            
            articles = []
            for item in items:
                try:
                    title = item.find('title').text if item.find('title') else ''
                    description = item.find('description').text if item.find('description') else ''
                    link = item.find('link').text if item.find('link') else ''
                    pub_date = item.find('pubDate').text if item.find('pubDate') else ''
                    source = item.find('source')
                    source_name = source.text if source else 'Google News'
                    
                    article = {
                        'title': title,
                        'content': description,
                        'source': source_name,
                        'published': pub_date,
                        'url': link
                    }
                    
                    if title and len(title) > 10:
                        articles.append(article)
                        
                except Exception as e:
                    continue
            
            logger.info(f"✅ Found {len(articles)} articles from Google News RSS")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Google News RSS: {e}")
            return []
    
    def fetch_news_manual_fallback(self, ticker: str) -> List[Dict]:
        """Manual fallback with sample news for testing"""
        base_ticker = ticker.replace('.NS', '')
        
        # Sample news articles for testing
        sample_news = [
            {
                'title': f"{base_ticker} Reports Strong Quarterly Earnings Growth",
                'content': f"{base_ticker} announced better-than-expected quarterly results with revenue growth of 15% year-over-year. The company's profit margins expanded due to improved operational efficiency.",
                'source': 'Financial Times',
                'published': datetime.now().strftime('%Y-%m-%d'),
                'url': f'https://example.com/news/{base_ticker}-earnings'
            },
            {
                'title': f"Analysts Raise Price Target for {base_ticker} Stock",
                'content': f"Several brokerage firms have increased their price targets for {base_ticker} citing strong growth prospects and market position. The stock has been upgraded to 'Buy' rating.",
                'source': 'Bloomberg',
                'published': datetime.now().strftime('%Y-%m-%d'),
                'url': f'https://example.com/news/{base_ticker}-upgrade'
            },
            {
                'title': f"{base_ticker} Expands Operations in European Markets",
                'content': f"The company announced new partnerships and expansion plans in European markets, which is expected to drive future revenue growth and market share gains.",
                'source': 'Reuters',
                'published': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'url': f'https://example.com/news/{base_ticker}-expansion'
            }
        ]
        
        logger.info(f"📋 Using sample news for {ticker} (fallback mode)")
        return sample_news
    
    def get_news(self, ticker: str, use_fallback: bool = True) -> List[Dict]:
        """
        Get news using multiple reliable sources with fallbacks
        """
        all_articles = []
        
        logger.info(f"🔍 Searching for news about {ticker}...")
        
        # Source 1: Enhanced yfinance
        yfinance_articles = self.fetch_news_yfinance_enhanced(ticker)
        all_articles.extend(yfinance_articles)
        
        # Source 2: Money Control (for Indian stocks)
        if ticker.endswith('.NS') and not yfinance_articles:
            moneycontrol_articles = self.fetch_news_moneycontrol(ticker)
            all_articles.extend(moneycontrol_articles)
        
        # Source 3: Google News RSS
        if not all_articles:
            google_articles = self.fetch_news_google_rss(ticker)
            all_articles.extend(google_articles)
        
        # Source 4: Manual fallback for testing
        if not all_articles and use_fallback:
            fallback_articles = self.fetch_news_manual_fallback(ticker)
            all_articles.extend(fallback_articles)
            logger.warning(f"⚠️ Using fallback sample news for {ticker}")
        
        # Remove duplicates based on title
        unique_articles = self._remove_duplicate_articles(all_articles)
        
        if unique_articles:
            logger.info(f"🎯 Successfully collected {len(unique_articles)} unique articles for {ticker}")
            for i, article in enumerate(unique_articles[:3]):  # Show first 3
                logger.info(f"   {i+1}. {article['title'][:80]}... ({article['source']})")
        else:
            logger.warning(f"❌ No news articles found for {ticker} from any source")
        
        return unique_articles
    
    def _remove_duplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article['title'].lower().strip()
            
            # Create a simplified version for comparison
            simple_title = re.sub(r'[^a-zA-Z0-9]', '', title)
            
            if (title and len(title) > 10 and 
                simple_title not in seen_titles and
                len(simple_title) > 5):
                
                seen_titles.add(simple_title)
                unique_articles.append(article)
        
        return unique_articles

# ==================== GROQ LLM INTEGRATION ====================

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    logger.warning("Groq package not available. Install with: pip install groq")
    GROQ_AVAILABLE = False

class GroqAIAgent:
    """AI Agent using Groq's LLM for advanced financial reasoning"""
    
    def __init__(self, api_key: str = None, model: str = "mixtral-8x7b-32768"):
        if not GROQ_AVAILABLE:
            raise ImportError("Groq package not available")
        
        self.api_key = api_key or GROQ_API_KEY
        if self.api_key == "YOUR_GROQ_API_KEY_HERE":
            raise ValueError("Please set your Groq API key in GROQ_API_KEY variable")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        
        logger.info(f"🤖 Groq AI Agent initialized with model: {model}")
    
    def analyze_sentiment_context(self, ticker: str, news_articles: List[Dict], 
                                finbert_results: Dict) -> Dict:
        """Use LLM for contextual sentiment analysis and reasoning"""
        
        if not news_articles:
            return self._fallback_analysis(finbert_results)
        
        prompt = self._build_sentiment_prompt(ticker, news_articles, finbert_results)
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=1024
            )
            
            analysis = response.choices[0].message.content
            return self._parse_llm_response(analysis, ticker)
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._fallback_analysis(finbert_results)
    
    def _build_sentiment_prompt(self, ticker: str, news_articles: List[Dict], 
                              finbert_results: Dict) -> str:
        """Build prompt for sentiment analysis"""
        
        articles_text = ""
        for i, article in enumerate(news_articles[:8]):  # Limit to 8 articles
            articles_text += f"{i+1}. {article.get('title', '')}\n"
            if article.get('content') and article['content'] != article.get('title', ''):
                articles_text += f"   Content: {article['content'][:150]}...\n"
            articles_text += f"   Source: {article.get('source', 'Unknown')}\n\n"
        
        finbert_summary = f"""
        FinBERT Technical Analysis:
        - Overall Sentiment: {finbert_results.get('overall_sentiment', 'neutral')}
        - Score: {finbert_results.get('overall_score', 0):.3f}
        - Confidence: {finbert_results.get('overall_confidence', 0):.1f}%
        - Articles Analyzed: {finbert_results.get('article_count', 0)}
        """
        
        prompt = f"""
        As a financial AI analyst, analyze sentiment for {ticker} stock.

        RECENT NEWS:
        {articles_text}

        TECHNICAL ANALYSIS:
        {finbert_summary}

        Provide JSON analysis with:
        - contextual_sentiment (bullish/bearish/neutral/mixed)
        - confidence_level (high/medium/low)
        - key_drivers (list of main factors)
        - market_impact (high/medium/low)
        - risk_factors (list of risks)
        - nuanced_analysis (brief explanation)
        - time_horizon (short/medium/long_term)

        Focus on factual, actionable insights.
        """
        
        return prompt
    
    def _parse_llm_response(self, response: str, ticker: str) -> Dict:
        """Parse LLM response for sentiment analysis"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "contextual_sentiment": "neutral",
                    "confidence_level": "medium",
                    "key_drivers": ["Limited news data available"],
                    "market_impact": "low",
                    "risk_factors": ["Insufficient information"],
                    "nuanced_analysis": "Analysis based on limited data",
                    "time_horizon": "short_term"
                }
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._fallback_analysis({})
    
    def _fallback_analysis(self, finbert_results: Dict) -> Dict:
        """Fallback analysis when LLM fails"""
        return {
            "contextual_sentiment": "neutral",
            "confidence_level": "medium",
            "key_drivers": ["Automated sentiment analysis"],
            "market_impact": "medium",
            "risk_factors": ["Standard market risks apply"],
            "nuanced_analysis": "Based on technical sentiment indicators",
            "time_horizon": "short_term"
        }

# ==================== FinBERT INTEGRATION ====================

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    
    class FinBERTAnalyzer:
        """Professional FinBERT sentiment analyzer"""
        
        def __init__(self):
            logger.info("🔄 Loading FinBERT AI Model...")
            self.model_name = "ProsusAI/finbert"
            self.device = 0 if torch.cuda.is_available() else -1
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
        def analyze_sentiment(self, text: str, title: str = "") -> Dict:
            if not text or len(text.strip()) < 10:
                return self._default_sentiment()
            
            try:
                full_text = f"{title}. {text}" if title else text
                if len(full_text) > 2000:
                    full_text = full_text[:2000]
                
                result = self.classifier(full_text)[0]
                return {
                    'sentiment': result['label'].lower(),
                    'confidence': result['score'],
                    'model': 'FinBERT'
                }
            except Exception as e:
                logger.error(f"FinBERT analysis failed: {e}")
                return self._default_sentiment()
        
        def _default_sentiment(self) -> Dict:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'model': 'FinBERT'}
    
    finbert_analyzer = FinBERTAnalyzer()
    ML_AVAILABLE = True
    
except Exception as e:
    logger.warning(f"FinBERT not available: {e}")
    ML_AVAILABLE = False
    finbert_analyzer = None

# ==================== AI SENTIMENT AGENT ====================

class SentimentAnalysisAgent:
    """AI-Powered Sentiment Analysis Agent"""
    
    def __init__(self, groq_api_key: str = None, use_ml: bool = True):
        self.use_ml = use_ml and ML_AVAILABLE
        self.finbert_analyzer = finbert_analyzer if self.use_ml else None
        
        # Initialize enhanced news fetcher
        self.news_fetcher = NewsFetcher()
        
        # Initialize Groq AI Agent
        self.groq_agent = None
        if GROQ_AVAILABLE:
            try:
                api_key = groq_api_key or GROQ_API_KEY
                if api_key and api_key != "YOUR_GROQ_API_KEY_HERE":
                    self.groq_agent = GroqAIAgent(api_key=api_key)
                    logger.info("🚀 Groq AI Agent integrated")
                else:
                    logger.warning("❌ Groq API key not provided")
            except Exception as e:
                logger.error(f"Failed to initialize Groq agent: {e}")
        
        logger.info(f"🎯 AI Sentiment Agent ready - LLM: {'✅' if self.groq_agent else '❌'}")
    
    def analyze(self, 
                ticker: str, 
                news_data: List[Dict] = None,
                fetch_news: bool = True,
                use_llm_reasoning: bool = True) -> Dict:
        """
        Advanced sentiment analysis with AI reasoning
        """
        
        if not ML_AVAILABLE:
            return self._error_result(ticker, "FinBERT model not available")
        
        # Fetch news if not provided
        if fetch_news and not news_data:
            logger.info(f"🔄 Fetching news for {ticker}...")
            news_data = self.news_fetcher.get_news(ticker)
        
        if not news_data:
            logger.warning(f"❌ No news data available for {ticker}")
            return self._neutral_result(ticker)
        
        try:
            # Step 1: Technical analysis with FinBERT
            technical_result = self._technical_analysis(ticker, news_data)
            
            # Step 2: AI Reasoning with Groq LLM
            ai_insights = {}
            if use_llm_reasoning and self.groq_agent:
                logger.info("🧠 Using Groq LLM for contextual analysis...")
                ai_insights = self.groq_agent.analyze_sentiment_context(
                    ticker, news_data, technical_result
                )
            
            # Step 3: Combine analyses
            final_result = self._combine_analyses(
                ticker, technical_result, ai_insights, news_data
            )
            
            logger.info(f"✅ AI Analysis complete for {ticker}")
            return final_result
        
        except Exception as e:
            logger.error(f"Error in AI sentiment analysis: {e}")
            return self._error_result(ticker, str(e))
    
    def _technical_analysis(self, ticker: str, news_data: List[Dict]) -> Dict:
        """Perform technical sentiment analysis with FinBERT"""
        article_sentiments = []
        
        for i, article in enumerate(news_data):
            try:
                title = article.get('title', '')
                content = article.get('content', '')
                
                if not title and not content:
                    continue
                
                sentiment = self._analyze_with_finbert(title, content)
                article_sentiments.append(sentiment)
                
                if i < len(news_data) - 1:
                    time.sleep(0.05)  # Reduced delay
                    
            except Exception as e:
                continue
        
        if not article_sentiments:
            return self._neutral_result(ticker)
        
        aggregate = self._aggregate_sentiments(article_sentiments)
        
        return {
            "ticker": ticker,
            "overall_sentiment": aggregate['sentiment'],
            "overall_score": round(aggregate['score'], 3),
            "overall_confidence": round(aggregate['confidence'], 1),
            "article_count": len(article_sentiments),
            "positive_articles": sum(1 for s in article_sentiments if s['sentiment'] == 'positive'),
            "negative_articles": sum(1 for s in article_sentiments if s['sentiment'] == 'negative'),
            "neutral_articles": sum(1 for s in article_sentiments if s['sentiment'] == 'neutral'),
            "article_sentiments": article_sentiments,
            "analysis_method": "FinBERT AI",
            "model_used": "ProsusAI/finbert"
        }
    
    def _combine_analyses(self, ticker: str, technical: Dict, 
                         ai_insights: Dict, news_data: List[Dict]) -> Dict:
        """Combine all analyses into final result"""
        
        # Determine final sentiment
        final_sentiment = technical['overall_sentiment']
        if ai_insights and 'contextual_sentiment' in ai_insights:
            contextual = ai_insights['contextual_sentiment']
            if contextual in ['bullish', 'bearish']:
                final_sentiment = 'positive' if contextual == 'bullish' else 'negative'
        
        summary = self._generate_summary(technical, ai_insights)
        
        result = {
            **technical,
            "overall_sentiment": final_sentiment,
            "summary": summary,
            "recommendation": self._sentiment_to_recommendation(final_sentiment, technical['overall_score']),
            "news_source": "Multiple Sources",
            "ai_agent_used": bool(self.groq_agent),
            "timestamp": datetime.now().isoformat()
        }
        
        if ai_insights:
            result["ai_contextual_analysis"] = ai_insights
        
        return result
    
    def _generate_summary(self, technical: Dict, ai_insights: Dict) -> str:
        """Generate comprehensive summary"""
        
        sentiment_emoji = {
            'positive': '📈', 
            'negative': '📉', 
            'neutral': '➡️'
        }
        
        emoji = sentiment_emoji.get(technical['overall_sentiment'], '➡️')
        base_summary = f"{emoji} {technical['overall_sentiment'].upper()} sentiment "
        base_summary += f"with {technical['overall_confidence']}% confidence "
        base_summary += f"based on {technical['article_count']} articles."
        
        if ai_insights and 'nuanced_analysis' in ai_insights:
            ai_analysis = ai_insights['nuanced_analysis']
            if len(ai_analysis) > 30:
                base_summary += f" AI Analysis: {ai_analysis[:200]}..."
        
        return base_summary
    
    def _analyze_with_finbert(self, title: str, content: str) -> Dict:
        """Analyze sentiment using FinBERT"""
        analysis = self.finbert_analyzer.analyze_sentiment(content, title)
        sentiment_score = self._sentiment_to_score(analysis['sentiment'], analysis['confidence'])
        
        return {
            'sentiment': analysis['sentiment'],
            'score': sentiment_score,
            'confidence': analysis['confidence'] * 100,
            'text_preview': title[:100] if title else content[:100],
            'model': analysis['model']
        }
    
    def _sentiment_to_score(self, sentiment: str, confidence: float) -> float:
        base_scores = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
        return base_scores.get(sentiment, 0.0) * confidence
    
    def _aggregate_sentiments(self, article_sentiments: List[Dict]) -> Dict:
        if not article_sentiments:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 50.0}
        
        weighted_scores = [s['score'] * (s['confidence'] / 100) for s in article_sentiments]
        confidences = [s['confidence'] for s in article_sentiments]
        
        avg_score = np.mean(weighted_scores) if weighted_scores else 0.0
        avg_confidence = np.mean(confidences) if confidences else 50.0
        
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
    
    def _sentiment_to_recommendation(self, sentiment: str, score: float) -> Dict:
        if sentiment == "positive":
            if score > 0.5:
                return {"action": "STRONG_BUY", "confidence": 85}
            elif score > 0.2:
                return {"action": "BUY", "confidence": 75}
            else:
                return {"action": "HOLD", "confidence": 60}
        elif sentiment == "negative":
            if score < -0.5:
                return {"action": "STRONG_SELL", "confidence": 85}
            elif score < -0.2:
                return {"action": "SELL", "confidence": 75}
            else:
                return {"action": "HOLD", "confidence": 60}
        else:
            return {"action": "HOLD", "confidence": 50}
    
    def _neutral_result(self, ticker: str) -> Dict:
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
            "model_used": "ProsusAI/finbert",
            "news_source": "Multiple Sources",
            "ai_agent_used": bool(self.groq_agent)
        }
    
    def _error_result(self, ticker: str, error_msg: str) -> Dict:
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
            "news_source": "Multiple Sources",
            "ai_agent_used": bool(self.groq_agent),
            "error": error_msg
        }

# ==================== TEST FUNCTION ====================

def test_sentiment_analysis():
    """Test the sentiment analysis with multiple stocks"""
    
    print("🧪 Testing Enhanced AI Sentiment Agent...")
    
    # Initialize agent
    agent = SentimentAnalysisAgent()
    
    test_stocks = ["INFY.NS", "NTPC.NS", "RELIANCE.NS", "TCS.NS"]
    
    for stock in test_stocks:
        print(f"\n{'='*60}")
        print(f"🔍 Analyzing: {stock}")
        print(f"{'='*60}")
        
        try:
            result = agent.analyze(stock, fetch_news=True, use_llm_reasoning=True)
            
            print(f"✅ Ticker: {result['ticker']}")
            print(f"✅ Sentiment: {result['overall_sentiment'].upper()}")
            print(f"✅ Score: {result['overall_score']:.3f}")
            print(f"✅ Confidence: {result['overall_confidence']}%")
            print(f"✅ Articles: {result['article_count']}")
            print(f"✅ AI Used: {result.get('ai_agent_used', False)}")
            print(f"✅ Summary: {result['summary']}")
            print(f"✅ Recommendation: {result['recommendation']}")
            
            if 'ai_contextual_analysis' in result:
                ai = result['ai_contextual_analysis']
                print(f"\n🤖 AI Analysis:")
                print(f"   - Context: {ai.get('contextual_sentiment', 'N/A')}")
                print(f"   - Confidence: {ai.get('confidence_level', 'N/A')}")
                print(f"   - Key Drivers: {', '.join(ai.get('key_drivers', []))}")
                
        except Exception as e:
            print(f"❌ Error analyzing {stock}: {e}")

if __name__ == "__main__":
    test_sentiment_analysis()