# new_news_agent.py
import os
import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

try:
    import numpy as np
except ImportError:
    np = None

try:
    from groq import Groq
except ImportError:
    Groq = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Keyword dictionaries (extended)
POSITIVE_KEYWORDS = [
    'growth', 'surge', 'rally', 'bull', 'bullish', 'gains', 'profit', 'profitability',
    'revenue', 'earnings', 'beat', 'outperform', 'bounce', 'recovery',
    'breakthrough', 'innovation', 'expansion', 'upgrade', 'upside', 'strength',
    'momentum', 'positive', 'optimistic', 'strong', 'robust',
    'buy', 'buying', 'accumulation', 'short covering', 'squeeze',
    'breakout', 'rebound', 'bounce back', 'surge higher',
    'record', 'highest', 'exceed', 'solid', 'new high', 'all time high', 'improved', 'improving',
    'better than expected', 'top performer', 'leader', 'market leader',
    'confidence', 'optimism', 'opportunity', 'potential', 'promising', 'encouraging',
    'positive outlook', 'bright', 'upbeat', 'enthusiasm',
    'acquisition', 'deal', 'partnership', 'collaboration', 'joint venture',
    'merger', 'strategic alliance', 'investment', 'funding', 'capital raise',
    'initiates coverage', 'raises target', 'buy rating', 'recommended', 'strong buy', 'target raise', 'increase'
]

NEGATIVE_KEYWORDS = [
    'decline', 'fall', 'crash', 'plunge', 'slump', 'drop', 'loss', 'deficit',
    'bearish', 'bear', 'downside', 'weakness', 'weak', 'deteriorate', 'deterioration',
    'downtrend', 'selloff', 'correction', 'drawdown', 'miss',
    'sell', 'selling', 'distribution', 'exit', 'dump', 'short', 'short sellers',
    'short pressure', 'margin call', 'liquidation', 'capitulation',
    'worst', 'underperform', 'underperformance', 'missed', 'failed', 'failure',
    'declining', 'falling', 'lower than expected', 'disappointing', 'poor',
    'fear', 'fearful', 'pessimistic', 'pessimism', 'concern', 'risk',
    'uncertainty', 'volatile', 'volatility', 'panic', 'anxiety', 'gloomy', 'bleak',
    'bankruptcy', 'bankrupt', 'insolvency', 'default', 'credit risk',
    'investigation', 'lawsuit', 'scandal', 'fraud', 'accounting', 'restatement',
    'regulatory', 'violation', 'fine', 'penalty', 'recall',
    'downgrade', 'cut', 'lower target', 'sell rating', 'reduce', 'strong sell', 'negative', 'target cut'
]

NEUTRAL_KEYWORDS = [
    'announced', 'report', 'data', 'information', 'statement', 'confirm',
    'update', 'guidance', 'forecast', 'expect', 'expected', 'analyst',
    'meeting', 'conference', 'event', 'news', 'today', 'yesterday'
]


def _try_parse_date(value: str) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    fmts = ("%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d")
    for fmt in fmts:
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            continue
    return None


class NewNews:
    """
    NewNews Agent
    - Deterministic keyword-based sentiment (fast)
    - Groq-powered executive summary (optional)
    - Grounded chat over provided articles (citations like [n])

    Usage:
        agent = NewNews(groq_api_key=os.getenv("GROQ_API_KEY"))
        agent.set_context(ticker="AAPL", company_name="Apple", news_data=articles)
        analysis = agent.analyze(ticker="AAPL")
        summary = agent.groq_summary()
        resp = agent.chat("What are key risks?")
    """

    def __init__(self,
                 groq_api_key: Optional[str] = None,
                 model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.4,
                 max_tokens: int = 900):
        self.groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.groq_key) if (self.groq_key and Groq) else None
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.ticker: Optional[str] = None
        self.company_name: Optional[str] = None
        self.articles: List[Dict[str, Any]] = []
        self.last_analysis: Optional[Dict[str, Any]] = None
        self.chat_history: List[Dict[str, str]] = []

        # Precompile patterns (handles multi-word phrases with word boundaries)
        self._pos_patterns = [re.compile(rf"\b{re.escape(k.lower())}\b") for k in POSITIVE_KEYWORDS]
        self._neg_patterns = [re.compile(rf"\b{re.escape(k.lower())}\b") for k in NEGATIVE_KEYWORDS]
        self._neu_patterns = [re.compile(rf"\b{re.escape(k.lower())}\b") for k in NEUTRAL_KEYWORDS]

    # Public API
    def set_context(self, ticker: str, company_name: str, news_data: List[Dict[str, Any]]) -> None:
        self.ticker = ticker
        self.company_name = company_name
        self.articles = self._normalize_articles(news_data)
        self.reset_chat()

    def analyze(self,
                ticker: str,
                news_data: Optional[List[Dict[str, Any]]] = None,
                news_text: Optional[str] = None) -> Dict[str, Any]:
        if news_text:
            data = [{"title": "News", "content": news_text, "source": "User", "publishedAt": None, "url": ""}]
        else:
            data = self._normalize_articles(news_data or self.articles)

        if not data:
            res = self._neutral_result(ticker)
            self.last_analysis = res
            return res

        article_sentiments: List[Dict[str, Any]] = []
        for art in data:
            try:
                title = art.get("title") or ""
                content = art.get("content") or art.get("description") or ""
                s = self._analyze_single_article(title, content)
                s.update({
                    "title": title[:160],
                    "source": art.get("source", "Unknown"),
                    "publishedAt": art.get("publishedAt", "Unknown"),
                    "url": art.get("url", "")
                })
                article_sentiments.append(s)
            except Exception as e:
                logger.warning(f"Analyze article error: {e}")

        if not article_sentiments:
            res = self._neutral_result(ticker)
            self.last_analysis = res
            return res

        aggregate = self._aggregate(article_sentiments)
        summary = self._summary(aggregate, len(article_sentiments))
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
            "recommendation": self._to_reco(aggregate['sentiment'], aggregate['score'])
        }
        self.last_analysis = result
        return result

    def groq_summary(self, top_k: int = 10) -> Optional[str]:
        if not self.client:
            return None
        if not self.articles:
            return None

        picked, idx_map = self._pick_articles(self.articles, top_k=top_k)
        articles_text = "\n\n".join([
            f"[{i}] Title: {a['title']}\n"
            f"Source: {a['source']}\n"
            f"Date: {a.get('publishedAt','Unknown')}\n"
            f"URL: {a.get('url','')}\n"
            f"Content: {a.get('content','') or a.get('description','')}"
            for i, a in picked
        ])

        analysis_hint = ""
        if self.last_analysis:
            la = self.last_analysis
            analysis_hint = f"Keyword snapshot: overall={la['overall_sentiment']}, score={la['overall_score']}, conf={la['overall_confidence']}%, pos/neg/neu={la['positive_articles']}/{la['negative_articles']}/{la['neutral_articles']}."

        prompt = f"""
You are a professional financial news analyst. Use only the Provided Articles. Be concise and actionable.
- Do not invent facts. If not covered, say so.
- Cite with [n] where n is the article index.

Company: {self.company_name or ''} ({self.ticker or ''})
{analysis_hint}

Provide:
1) Executive Summary (2-3 sentences)
2) Key Points (3-5 bullets)
3) Sentiment (Positive/Negative/Neutral) with 1-line rationale
4) Impact Areas
5) Watchlist / Action Items

Provided Articles:
{articles_text}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq summary error: {e}")
            return None

    def chat(self,
             user_message: str,
             max_articles: int = 12,
             temperature: float = 0.2,
             return_citations: bool = True) -> Dict[str, Any]:
        if not self.client:
            return {"answer": "Groq not configured (set GROQ_API_KEY).", "citations": []}
        if not self.articles:
            return {"answer": "No articles loaded. Call set_context(...).", "citations": []}

        picked, idx_map = self._pick_articles(self.articles, top_k=max_articles)
        articles_text = "\n\n".join([
            f"[{i}] Title: {a['title']}\n"
            f"Source: {a['source']}\n"
            f"Date: {a.get('publishedAt','Unknown')}\n"
            f"URL: {a.get('url','')}\n"
            f"Content: {a.get('content','') or a.get('description','')}"
            for i, a in picked
        ])

        guardrails = f"""
You are a financial assistant. Answer ONLY using the Provided Articles below.
- If not in the articles, say: "Not covered in the provided articles."
- Use [n] citations.
Company: {self.company_name or ''} ({self.ticker or ''})

Provided Articles:
{articles_text}
"""

        conversation = self._build_history(guardrails, user_message)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=conversation,
                max_tokens=self.max_tokens,
                temperature=temperature
            )
            answer = resp.choices[0].message.content
            citations = self._extract_citations(answer, picked) if return_citations else []
            self.chat_history.append({"role": "assistant", "content": answer})
            return {"answer": answer, "citations": citations}
        except Exception as e:
            logger.error(f"Groq chat error: {e}")
            return {"answer": "Sorry, I had an issue generating the response.", "citations": []}

    def reset_chat(self) -> None:
        self.chat_history = [{
            "role": "system",
            "content": "You are a precise, citation-first financial assistant. Ground answers in provided articles and cite as [n]."
        }]

    # Internals
    def _normalize_articles(self, articles: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not articles:
            return []
        norm: List[Dict[str, Any]] = []
        for a in articles:
            if not isinstance(a, dict):
                continue
            title = a.get("title") or ""
            content = a.get("content") or a.get("description") or ""
            src = a.get("source")
            src_name = src.get("name") if isinstance(src, dict) else src
            norm.append({
                "title": str(title),
                "content": str(content),
                "source": str(src_name or "Unknown"),
                "publishedAt": a.get("publishedAt") or a.get("published_date") or a.get("date") or "Unknown",
                "url": a.get("url") or ""
            })
        return norm

    def _analyze_single_article(self, title: str, content: str) -> Dict[str, Any]:
        text = f"{title.lower()} {title.lower()} {content.lower()}"
        pos = sum(len(p.findall(text)) for p in self._pos_patterns)
        neg = sum(len(p.findall(text)) for p in self._neg_patterns)
        neu = sum(len(p.findall(text)) for p in self._neu_patterns)
        total = pos + neg + neu

        if total == 0:
            conf = 30.0 if (len(title) > 50 or len(content) > 200) else 20.0
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'confidence': conf,
                'text_preview': (title or content)[:100]
            }

        score = (pos - neg) / max(1, total)
        sentiment = 'positive' if score > 0.2 else 'negative' if score < -0.2 else 'neutral'
        confidence = min(100, abs(score) * 100)

        return {
            'sentiment': sentiment,
            'score': round(score, 3),
            'positive_count': pos,
            'negative_count': neg,
            'neutral_count': neu,
            'confidence': round(confidence, 1),
            'text_preview': (title or content)[:100]
        }

    def _aggregate(self, sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not sentiments:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 50.0}
        weighted = [s['score'] * (s['confidence'] / 100) for s in sentiments]
        confs = [s['confidence'] for s in sentiments]
        avg_score = (float(np.mean(weighted)) if np else sum(weighted) / len(weighted)) if weighted else 0.0
        avg_conf = (float(np.mean(confs)) if np else sum(confs) / len(confs)) if confs else 50.0
        overall = 'positive' if avg_score > 0.2 else 'negative' if avg_score < -0.2 else 'neutral'
        return {'sentiment': overall, 'score': avg_score, 'confidence': avg_conf}

    def _summary(self, agg: Dict[str, Any], n: int) -> str:
        sent = agg['sentiment'].upper()
        score = agg['score']
        conf = agg['confidence']
        emoji = "📈" if score > 0.1 else "📉" if score < -0.1 else "➡️"
        if sent == "POSITIVE":
            if conf > 80: return f"{emoji} STRONGLY POSITIVE: Bullish sentiment. {n} articles with strong positive signals."
            if conf > 60: return f"{emoji} POSITIVE: Bullish outlook. {n} articles show positive bias."
            return f"{emoji} SLIGHTLY POSITIVE: Mild bullish sentiment. {n} articles analyzed."
        if sent == "NEGATIVE":
            if conf > 80: return f"{emoji} STRONGLY NEGATIVE: Bearish sentiment. {n} articles with strong negative signals."
            if conf > 60: return f"{emoji} NEGATIVE: Bearish outlook. {n} articles show negative bias."
            return f"{emoji} SLIGHTLY NEGATIVE: Mild bearish sentiment. {n} articles analyzed."
        return f"{emoji} NEUTRAL: Mixed sentiment. {n} articles with balanced signals."

    def _to_reco(self, sentiment: str, score: float) -> Dict[str, Any]:
        if sentiment == "positive":
            if score > 0.5: return {"action": "STRONG_BUY", "confidence": min(90, (score + 1) * 45)}
            if score > 0.2: return {"action": "BUY", "confidence": min(85, (score + 1) * 40)}
            return {"action": "HOLD", "confidence": 60}
        if sentiment == "negative":
            if score < -0.5: return {"action": "STRONG_SELL", "confidence": min(90, (abs(score) + 1) * 45)}
            if score < -0.2: return {"action": "SELL", "confidence": min(85, (abs(score) + 1) * 40)}
            return {"action": "HOLD", "confidence": 60}
        return {"action": "HOLD", "confidence": 50}

    def _pick_articles(self, articles: List[Dict[str, Any]], top_k: int = 10) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[int, Dict[str, Any]]]:
        def sort_key(a: Dict[str, Any]):
            dt = _try_parse_date(a.get("publishedAt"))
            return (0 if dt else 1, -(dt.timestamp()) if dt else 0)
        sorted_arts = sorted(articles, key=sort_key)
        picked: List[Tuple[int, Dict[str, Any]]] = []
        idx_map: Dict[int, Dict[str, Any]] = {}
        for i, a in enumerate(sorted_arts[:top_k], start=1):
            picked.append((i, a))
            idx_map[i] = a
        return picked, idx_map

    def _build_history(self, guardrails: str, user_message: str) -> List[Dict[str, str]]:
        trimmed = self.chat_history[-12:] if len(self.chat_history) > 12 else self.chat_history[:]
        convo = trimmed + [{"role": "system", "content": guardrails}]
        if self.last_analysis:
            la = self.last_analysis
            snapshot = json.dumps({
                "overall_sentiment": la["overall_sentiment"],
                "overall_score": la["overall_score"],
                "overall_confidence": la["overall_confidence"],
                "pos_neg_neu": [la["positive_articles"], la["negative_articles"], la["neutral_articles"]]
            })
            convo.append({"role": "system", "content": f"Sentiment Snapshot JSON: {snapshot}"})
        convo.append({"role": "user", "content": user_message})
        self.chat_history.append({"role": "user", "content": user_message})
        return convo

    def _extract_citations(self, answer: str, picked: List[Tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        indices = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
        idx_dict = {i: a for i, a in picked}
        cites: List[Dict[str, Any]] = []
        for i in sorted(indices):
            art = idx_dict.get(i)
            if art:
                cites.append({
                    "index": i,
                    "title": art.get("title", "N/A"),
                    "url": art.get("url", ""),
                    "source": art.get("source", "Unknown"),
                    "publishedAt": art.get("publishedAt", "Unknown"),
                })
        return cites

    # Neutral result helper
    def _neutral_result(self, ticker: str) -> Dict[str, Any]:
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

# For compatibility if you want:
NewNewsAgent = NewNews