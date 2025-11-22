# ==================== FILE: trading_bot/agents/professional_news_logic.py ====================

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote_plus, urljoin
import json
import feedparser
import warnings
import time
from typing import List, Dict, Any

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# CORE LOGIC: News Fetching and Content Extraction
# ------------------------------------------------------------

class InvestorGradeNewsFetcher:
    """Professional-grade news fetcher using multiple fallback strategies"""
    
    def __init__(self, stock_name):
        self.stock = stock_name.upper()
        self.stock_lower = stock_name.lower()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    # --- News Source Methods (kept as provided by user) ---
    def get_rss_feeds(self):
        # ... (user's get_rss_feeds implementation) ...
        feeds = [
            f"https://news.google.com/rss/search?q={quote_plus(self.stock)}+stock+when:7d&hl=en-IN&gl=IN&ceid=IN:en",
            f"https://economictimes.indiatimes.com/topic/{self.stock_lower}/rss",
            f"https://www.moneycontrol.com/rss/buzzingstocks.xml",
            f"https://www.business-standard.com/rss/stock-market-106.rss",
        ]
        articles = []
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]:
                    if self.stock.lower() in entry.title.lower() or self.stock.get('summary', '').lower():
                        articles.append({
                            'title': entry.title,
                            'url': entry.link,
                            'published': entry.get('published', 'N/A'),
                            'source': 'RSS Feed'
                        })
            except:
                continue
        return articles

    def get_moneycontrol_news(self):
        # ... (user's get_moneycontrol_news implementation) ...
        try:
            search_url = f"https://www.moneycontrol.com/stocks/cptmarket/compsearchnew.php?search_data=&cid=&mbsearch_str={self.stock}&topsearch_type=1"
            resp = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            company_link = soup.find('a', href=re.compile(r'/india/stockpricequote/'))
            if company_link:
                company_url = urljoin("https://www.moneycontrol.com", company_link['href'])
                news_resp = self.session.get(company_url, timeout=10)
                news_soup = BeautifulSoup(news_resp.content, 'html.parser')
                articles = []
                news_items = news_soup.find_all('li', class_=re.compile('news_list'))
                for item in news_items[:5]:
                    link = item.find('a', href=True)
                    if link:
                        articles.append({
                            'title': link.get_text(strip=True),
                            'url': urljoin("https://www.moneycontrol.com", link['href']),
                            'source': 'MoneyControl'
                        })
                return articles
        except Exception as e:
            return []
        return []
    
    def get_economictimes_news(self):
        # ... (user's get_economictimes_news implementation) ...
        urls_to_try = [
            f"https://economictimes.indiatimes.com/{self.stock_lower}-ltd/stocks/companyid-{self.stock_lower}.cms",
            f"https://economictimes.indiatimes.com/topic/{self.stock_lower}",
            f"https://economictimes.indiatimes.com/markets/stocks/news"
        ]
        articles = []
        for url in urls_to_try:
            try:
                resp = self.session.get(url, timeout=10)
                soup = BeautifulSoup(resp.content, 'html.parser')
                links = soup.find_all('a', href=re.compile(r'articleshow|news'))
                for link in links[:10]:
                    title = link.get_text(strip=True)
                    if len(title) > 20 and self.stock.lower() in title.lower():
                        full_url = urljoin("https://economictimes.indiatimes.com", link['href'])
                        if full_url not in [a['url'] for a in articles]:
                            articles.append({
                                'title': title,
                                'url': full_url,
                                'source': 'Economic Times'
                            })
                if articles:
                    break
            except:
                continue
        return articles

    def get_yahoo_finance_news(self):
        # ... (user's get_yahoo_finance_news implementation) ...
        try:
            url = f"https://finance.yahoo.com/quote/{self.stock}/news"
            resp = self.session.get(url, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            articles = []
            news_items = soup.find_all('h3', class_=re.compile('Mb'))
            for item in news_items[:5]:
                link = item.find('a', href=True)
                if link:
                    articles.append({
                        'title': link.get_text(strip=True),
                        'url': urljoin("https://finance.yahoo.com", link['href']),
                        'source': 'Yahoo Finance'
                    })
            return articles
        except:
            return []
    
    def get_reuters_news(self):
        # ... (user's get_reuters_news implementation) ...
        try:
            url = f"https://www.reuters.com/site-search/?query={quote_plus(self.stock)}%20stock"
            resp = self.session.get(url, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            articles = []
            items = soup.find_all('li', class_=re.compile('search-result'))
            for item in items[:5]:
                link = item.find('a', href=True)
                if link and 'markets' in link['href']:
                    articles.append({
                        'title': link.get_text(strip=True),
                        'url': urljoin("https://www.reuters.com", link['href']),
                        'source': 'Reuters'
                    })
            return articles
        except:
            return []

    def get_investing_com_news(self):
        # ... (user's get_investing_com_news implementation) ...
        try:
            url = f"https://www.investing.com/search/?q={quote_plus(self.stock)}"
            resp = self.session.get(url, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            articles = []
            items = soup.find_all('a', class_=re.compile('article'))[:5]
            for item in items:
                if item.get('href'):
                    articles.append({
                        'title': item.get_text(strip=True),
                        'url': urljoin("https://www.investing.com", item['href']),
                        'source': 'Investing.com'
                    })
            return articles
        except:
            return []
    
    def fetch_all_sources(self):
        """Fetch from all sources with intelligent fallback"""
        all_articles = []
        
        fetchers = [
            self.get_rss_feeds,
            self.get_moneycontrol_news,
            self.get_yahoo_finance_news,
            self.get_economictimes_news,
            self.get_reuters_news,
            self.get_investing_com_news,
        ]
        
        for fetcher in fetchers:
            try:
                articles = fetcher()
                all_articles.extend(articles)
                time.sleep(0.5)
            except Exception:
                continue
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        return unique_articles[:10] # Limit to top 10 articles


class SmartContentExtractor:
    """Intelligent article content extraction with multiple strategies"""
    
    @staticmethod
    def extract_content(url, title=""):
        # ... (user's extract_content implementation) ...
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'form']):
                tag.decompose()
            
            article = soup.find('article')
            if article:
                text = article.get_text(separator=' ', strip=True)
                if len(text) > 200:
                    return text[:8000]
            
            main = soup.find(['main', 'div'], class_=re.compile(r'article|content|story|post-body|entry'))
            if main:
                text = main.get_text(separator=' ', strip=True)
                if len(text) > 200:
                    return text[:8000]
            
            paragraphs = soup.find_all('p')
            if len(paragraphs) >= 3:
                text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                if len(text) > 200:
                    return text[:8000]
            
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                return f"{title}. {meta_desc.get('content', '')}"
            
            return ""
            
        except Exception:
            return ""

# ------------------------------------------------------------
# WRAPPER FOR AGENT EXECUTION
# ------------------------------------------------------------

class ProfessionalSentimentLogic:
    """
    Combines fetching and extraction to prepare data for the LLM agent.
    """
    def get_analyzable_articles(self, stock_name: str) -> List[Dict[str, Any]]:
        fetcher = InvestorGradeNewsFetcher(stock_name)
        raw_articles = fetcher.fetch_all_sources()
        
        analyzable_data = []
        extractor = SmartContentExtractor()

        for article in raw_articles:
            content = extractor.extract_content(article['url'], article['title'])
            
            # Only include articles with usable content for LLM analysis
            if len(content) > 100:
                analyzable_data.append({
                    'title': article['title'],
                    'url': article['url'],
                    'source': article.get('source', 'Unknown'),
                    'content': content
                })
        
        return analyzable_data[:5] # Limit content extraction overhead