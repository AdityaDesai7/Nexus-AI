import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote, urljoin, quote_plus
import feedparser
import time
from datetime import datetime, timedelta
import warnings
import os
from typing import List, Dict
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

class NewsAggregator:
    """
    Multi-source news aggregator for stock market analysis
    Priority: NewsAPI → RSS Feeds → Web Scraping with fallbacks
    """
    
    def __init__(self, news_api_key: str = None):
        """
        Initialize news aggregator
        
        Args:
            news_api_key: NewsAPI key (or use NEWS_API_KEY env var)
        """
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY')
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        print("✅ NewsAggregator initialized")
    
    # ==================== METHOD 1: NewsAPI (MOST RELIABLE) ====================
    def fetch_from_newsapi(self, stock_name: str, days: int = 7) -> List[Dict]:
        """
        Fetch news from NewsAPI (requires API key)
        Best for: Quick, reliable results
        """
        print(f"📡 [Method 1] Fetching from NewsAPI...")
        
        if not self.news_api_key:
            print("   ⚠️  NewsAPI key not found")
            return []
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"{stock_name} stock OR {stock_name} shares OR {stock_name} trade",
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": self.news_api_key,
                "pageSize": 20,
                "searchIn": "title,description"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            articles_data = response.json().get('articles', [])
            
            articles = []
            for article in articles_data[:15]:
                content = article.get('description') or article.get('content') or ""
                
                articles.append({
                    'title': article.get('title', 'N/A'),
                    'content': content[:2000],
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', 'NewsAPI'),
                    'published_at': article.get('publishedAt', ''),
                    'image': article.get('urlToImage', ''),
                    'author': article.get('author', 'Unknown')
                })
            
            print(f"   ✅ NewsAPI: Found {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"   ⚠️  NewsAPI error: {str(e)[:50]}")
            return []
    
    # ==================== METHOD 2: RSS FEEDS ====================
    def fetch_from_rss_feeds(self, stock_name: str) -> List[Dict]:
        """
        Fetch from RSS feeds (fast, no API key needed)
        Best for: Indian stocks, tech stocks
        """
        print(f"📡 [Method 2] Fetching from RSS feeds...")
        
        feeds = [
            # Google News RSS
            f"https://news.google.com/rss/search?q={quote(stock_name)}+stock&hl=en-IN&gl=IN&ceid=IN:en",
            # Economic Times
            f"https://economictimes.indiatimes.com/topic/{stock_name.lower()}/rss",
            # Business Standard
            "https://www.business-standard.com/rss/stock-market-106.rss",
            # Moneycontrol Top News
            "https://www.moneycontrol.com/rss/buzzingstocks.xml",
            # FinTwit / Market News
            f"https://feeds.bloomberg.com/markets/news.rss",
        ]
        
        articles = []
        
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:8]:
                    title = entry.get('title', '')
                    
                    # Filter: only relevant entries
                    if stock_name.lower() not in title.lower():
                        continue
                    
                    summary = entry.get('summary', '') or entry.get('description', '')
                    
                    # Clean HTML from summary
                    summary = BeautifulSoup(summary, 'html.parser').get_text()
                    
                    articles.append({
                        'title': title,
                        'content': summary[:2000],
                        'url': entry.get('link', ''),
                        'source': feed.feed.get('title', 'RSS Feed'),
                        'published_at': entry.get('published', ''),
                        'image': '',
                        'author': entry.get('author', 'Unknown')
                    })
                
                time.sleep(0.3)  # Be polite to RSS feeds
                
            except Exception as e:
                continue
        
        print(f"   ✅ RSS Feeds: Found {len(articles)} articles")
        return articles
    
    # ==================== METHOD 3: MONEYCONTROL SCRAPING ====================
    def fetch_from_moneycontrol(self, stock_name: str) -> List[Dict]:
        """
        Scrape Moneycontrol (Indian stock market focus)
        Best for: Indian stocks like TCS, Reliance, HDFC
        """
        print(f"📡 [Method 3] Fetching from Moneycontrol...")
        
        try:
            # Search for company
            search_url = f"https://www.moneycontrol.com/stocks/cptmarket/compsearchnew.php?search_data=&cid=&mbsearch_str={stock_name}&topsearch_type=1"
            
            resp = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Find first company link
            company_link = soup.find('a', href=re.compile(r'/india/stockpricequote/'))
            
            if not company_link:
                print("   ⚠️  Company not found on Moneycontrol")
                return []
            
            # Get company page
            company_url = urljoin("https://www.moneycontrol.com", company_link['href'])
            company_resp = self.session.get(company_url, timeout=10)
            company_soup = BeautifulSoup(company_resp.content, 'html.parser')
            
            articles = []
            
            # Find news section (might vary by page structure)
            news_section = company_soup.find('div', class_=re.compile('news|article', re.I))
            
            if news_section:
                news_links = news_section.find_all('a', href=True)
                
                for link in news_links[:10]:
                    title = link.get_text(strip=True)
                    url = urljoin("https://www.moneycontrol.com", link['href'])
                    
                    if len(title) > 20:
                        # Try to fetch content
                        try:
                            content_resp = self.session.get(url, timeout=8)
                            content_soup = BeautifulSoup(content_resp.content, 'html.parser')
                            
                            # Extract content
                            content_div = content_soup.find('div', class_=re.compile('content|story', re.I))
                            content = content_div.get_text(strip=True)[:2000] if content_div else ""
                            
                            articles.append({
                                'title': title,
                                'content': content,
                                'url': url,
                                'source': 'Moneycontrol',
                                'published_at': '',
                                'image': '',
                                'author': 'Moneycontrol'
                            })
                        except:
                            pass
                
                time.sleep(0.5)
            
            print(f"   ✅ Moneycontrol: Found {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"   ⚠️  Moneycontrol error: {str(e)[:50]}")
            return []
    
    # ==================== METHOD 4: ECONOMIC TIMES SCRAPING ====================
    def fetch_from_economic_times(self, stock_name: str) -> List[Dict]:
        """
        Scrape Economic Times (Indian business news)
        Best for: Indian market insights
        """
        print(f"📡 [Method 4] Fetching from Economic Times...")
        
        try:
            # Try company-specific URL
            urls_to_try = [
                f"https://economictimes.indiatimes.com/{stock_name.lower()}-ltd/stocks",
                f"https://economictimes.indiatimes.com/topic/{stock_name.lower()}",
                f"https://economictimes.indiatimes.com/markets/stocks/news",
            ]
            
            articles = []
            
            for url in urls_to_try:
                try:
                    resp = self.session.get(url, timeout=10)
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    
                    # Find article links
                    links = soup.find_all('a', href=re.compile(r'articleshow|news', re.I))
                    
                    for link in links[:10]:
                        title = link.get_text(strip=True)
                        
                        if len(title) > 20 and (stock_name.lower() in title.lower() or len(articles) == 0):
                            article_url = urljoin("https://economictimes.indiatimes.com", link['href'])
                            
                            # Check if already added
                            if any(a['url'] == article_url for a in articles):
                                continue
                            
                            # Fetch article content
                            try:
                                article_resp = self.session.get(article_url, timeout=8)
                                article_soup = BeautifulSoup(article_resp.content, 'html.parser')
                                
                                # Extract content
                                article_div = article_soup.find('div', class_=re.compile('article|story|content', re.I))
                                content = article_div.get_text(strip=True)[:2000] if article_div else title
                                
                                articles.append({
                                    'title': title,
                                    'content': content,
                                    'url': article_url,
                                    'source': 'Economic Times',
                                    'published_at': '',
                                    'image': '',
                                    'author': 'ET'
                                })
                            except:
                                articles.append({
                                    'title': title,
                                    'content': title,
                                    'url': article_url,
                                    'source': 'Economic Times',
                                    'published_at': '',
                                    'image': '',
                                    'author': 'ET'
                                })
                    
                    if articles:
                        break
                    
                except:
                    continue
                
                time.sleep(0.5)
            
            print(f"   ✅ Economic Times: Found {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"   ⚠️  Economic Times error: {str(e)[:50]}")
            return []
    
    # ==================== METHOD 5: YAHOO FINANCE ====================
    def fetch_from_yahoo_finance(self, stock_name: str) -> List[Dict]:
        """
        Fetch from Yahoo Finance (global stocks)
        Best for: US stocks, international markets
        """
        print(f"📡 [Method 5] Fetching from Yahoo Finance...")
        
        try:
            url = f"https://finance.yahoo.com/quote/{stock_name}/news"
            
            resp = self.session.get(url, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            articles = []
            
            # Find news items
            news_items = soup.find_all(['h3', 'h2'], class_=re.compile('title|headline', re.I))
            
            for item in news_items[:10]:
                link = item.find('a', href=True)
                
                if link:
                    title = link.get_text(strip=True)
                    article_url = link['href']
                    
                    if not article_url.startswith('http'):
                        article_url = urljoin("https://finance.yahoo.com", article_url)
                    
                    articles.append({
                        'title': title,
                        'content': title,  # Yahoo blocks content scraping
                        'url': article_url,
                        'source': 'Yahoo Finance',
                        'published_at': '',
                        'image': '',
                        'author': 'Yahoo Finance'
                    })
            
            print(f"   ✅ Yahoo Finance: Found {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"   ⚠️  Yahoo Finance error: {str(e)[:50]}")
            return []
    
    # ==================== MAIN AGGREGATION ====================
    def fetch_all_news(self, stock_name: str) -> List[Dict]:
        """
        Fetch news from all available sources with fallback strategy
        
        Args:
            stock_name: Stock ticker (e.g., "TCS", "AAPL", "RELIANCE")
        
        Returns:
            List of articles with title, content, url, source, etc.
        """
        
        print("\\n" + "="*80)
        print(f"🔍 FETCHING NEWS FOR: {stock_name.upper()}")
        print("="*80 + "\\n")
        
        all_articles = []
        
        # Priority order - try each method
        fetchers = [
            ('NewsAPI', self.fetch_from_newsapi(stock_name)),
            ('RSS Feeds', self.fetch_from_rss_feeds(stock_name)),
            ('Moneycontrol', self.fetch_from_moneycontrol(stock_name)),
            ('Economic Times', self.fetch_from_economic_times(stock_name)),
            ('Yahoo Finance', self.fetch_from_yahoo_finance(stock_name)),
        ]
        
        for source_name, articles in fetchers:
            all_articles.extend(articles)
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_articles = []
        
        for article in all_articles:
            url = article['url']
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        print("\\n" + "="*80)
        print(f"✅ TOTAL UNIQUE ARTICLES FOUND: {len(unique_articles)}")
        print("="*80 + "\\n")
        
        # Return top articles (most recent/relevant first)
        return unique_articles[:15]
    
    def get_articles_json(self, stock_name: str) -> List[Dict]:
        """
        Get articles as JSON-ready format
        Perfect for API responses
        """
        articles = self.fetch_all_news(stock_name)
        
        return {
            'stock': stock_name,
            'timestamp': datetime.now().isoformat(),
            'total_articles': len(articles),
            'articles': articles
        }


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    
    print("\\n" + "="*80)
    print("📰 ENHANCED NEWS FETCHER FOR STOCK SENTIMENT ANALYSIS")
    print("="*80)
    
    # Initialize
    fetcher = NewsAggregator()
    
    # Fetch news
    stock = input("\\n💼 Enter stock ticker (TCS, RELIANCE, AAPL): ").strip().upper()
    
    if stock:
        articles = fetcher.fetch_all_news(stock)
        
        print(f"\\n📰 First 3 articles:\\n")
        for i, article in enumerate(articles[:3], 1):
            print(f"[{i}] {article['title'][:70]}...")
            print(f"    Source: {article['source']}")
            print(f"    Content: {article['content'][:100]}...\\n")
        
        # Also save as JSON
        import json
        output = fetcher.get_articles_json(stock)
        filename = f"{stock}_articles.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Articles saved to: {filename}")
    else:
        print("❌ No ticker provided!")