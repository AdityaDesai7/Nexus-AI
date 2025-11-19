# new_news_fetcher.py
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TAVILY_URL = "https://api.tavily.com/search"
NEWSAPI_URL = "https://newsapi.org/v2/everything"


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    fmts = ("%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d")
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


class NewNewsFetcher:
    """
    Fetch-only class. Combines Tavily + NewsAPI results, normalizes and dedupes.
    No LLM, no summaries — just clean article data.
    """

    def __init__(self,
                 tavily_api_key: Optional[str] = None,
                 news_api_key: Optional[str] = None,
                 timeout_seconds: int = 15):
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        # Support both NEWS_API_KEY and NEWSAPI_KEY env names
        self.news_api_key = news_api_key or os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")
        self.timeout = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NewNewsFetcher/1.0"})

        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY missing — Tavily fetch disabled.")
        if not self.news_api_key:
            logger.warning("NEWS_API_KEY/NEWSAPI_KEY missing — NewsAPI fetch disabled.")

    def get_news(self,
                 company_name: str,
                 ticker: Optional[str] = None,
                 max_results: int = 16,
                 days: int = 7,
                 dedupe: bool = True) -> List[Dict]:
        query = f"{company_name} {ticker} latest news" if ticker else f"{company_name} latest news"

        articles: List[Dict] = []
        tavily_json = self._fetch_tavily(query=query, max_results=max_results, days=days) if self.tavily_api_key else None
        newsapi_json = self._fetch_newsapi(query=company_name, max_results=max_results, days=days) if self.news_api_key else None

        if tavily_json:
            articles.extend(self._normalize_tavily(tavily_json))
        if newsapi_json:
            articles.extend(self._normalize_newsapi(newsapi_json))

        if dedupe:
            articles = self._dedupe(articles)

        # Sort newest first if possible
        articles.sort(key=lambda a: (_parse_date(a.get("publishedAt")) or datetime.min), reverse=True)
        return articles

    # --------------------
    # External fetchers
    # --------------------
    def _fetch_tavily(self, query: str, max_results: int, days: int) -> Optional[Dict]:
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "topic": "news",
            "include_answer": True,
            "max_results": max_results,
            "days": days
        }
        try:
            r = self.session.post(TAVILY_URL, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.error(f"Tavily error: {e}")
            return None

    def _fetch_newsapi(self, query: str, max_results: int, days: int) -> Optional[Dict]:
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "q": query,
            "apiKey": self.news_api_key,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": max_results,
            "from": from_date
        }
        try:
            r = self.session.get(NEWSAPI_URL, params=params, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.error(f"NewsAPI error: {e}")
            return None

    # --------------------
    # Normalizers
    # --------------------
    def _normalize_tavily(self, data: Dict) -> List[Dict]:
        results = data.get("results", []) if isinstance(data, dict) else []
        norm: List[Dict] = []
        for r in results:
            norm.append({
                "title": r.get("title") or "N/A",
                "content": r.get("content") or "",
                "description": r.get("content") or "",
                "source": r.get("source") or "Unknown",
                "publishedAt": r.get("published_date") or "Unknown",
                "url": r.get("url") or ""
            })
        return norm

    def _normalize_newsapi(self, data: Dict) -> List[Dict]:
        arts = data.get("articles", []) if isinstance(data, dict) else []
        norm: List[Dict] = []
        for a in arts:
            src = (a.get("source") or {}).get("name", "Unknown")
            norm.append({
                "title": a.get("title") or "N/A",
                "content": a.get("content") or "",
                "description": a.get("description") or "",
                "source": src,
                "publishedAt": a.get("publishedAt") or "Unknown",
                "url": a.get("url") or ""
            })
        return norm

    def _dedupe(self, articles: List[Dict]) -> List[Dict]:
        seen_urls = set()
        seen_titles = set()
        unique: List[Dict] = []
        for a in articles:
            url = (a.get("url") or "").strip()
            title = (a.get("title") or "").strip().lower()
            if url and url in seen_urls:
                continue
            if title and title in seen_titles:
                continue
            unique.append(a)
            if url:
                seen_urls.add(url)
            if title:
                seen_titles.add(title)
        return unique