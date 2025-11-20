# trading_bot/agents/news_agent.py
import logging
from typing import Dict, Any, List
import pandas as pd
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class NewsAgent(BaseAgent):
    """
    REAL AI NEWS AGENT
    - Multi-source news fetch
    - Cleaning + ranking
    - Sentiment scoring (rule-based or LLM if available)
    - Structured output
    """

    def __init__(self):
        super().__init__("news_agent")

    # ---------------------------------------------------
    # Utility: Clean text
    # ---------------------------------------------------
    def _clean(self, text: str) -> str:
        if not text:
            return ""
        return (
            text.replace("\n", " ")
                .replace("\t", " ")
                .strip()
        )

    # ---------------------------------------------------
    # Utility: Rule-based sentiment as fallback
    # ---------------------------------------------------
    def _fallback_sentiment(self, text: str) -> str:
        if not text:
            return "neutral"
        t = text.lower()

        positive = ["gain", "growth", "profit", "upgrade", "beat", "strong"]
        negative = ["loss", "down", "fall", "fraud", "weak", "downgrade"]

        score = 0
        for w in positive:
            if w in t:
                score += 1
        for w in negative:
            if w in t:
                score -= 1

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------
    # AI SENTIMENT via LLM (optional)
    # ---------------------------------------------------
    def _llm_sentiment(self, title: str, summary: str):
        if not self.llm:
            return None

        prompt = f"""
        Classify sentiment for this news headline:

        Title: {title}
        Summary: {summary}

        Respond with one word: positive, neutral, or negative.
        """

        try:
            out = self.llm.run(prompt)
            text = str(out).lower()

            if "positive" in text:
                return "positive"
            if "negative" in text:
                return "negative"
            return "neutral"
        except Exception:
            return None  # fallback will be used

    # ---------------------------------------------------
    # Core agent logic
    # ---------------------------------------------------
    def act(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ticker = params.get("ticker")
        limit = params.get("limit", 15)

        # Fetch news using our universal tool
        try:
            raw_news = self.tools["fetch_news"](ticker, limit)
        except Exception as e:
            logger.error(f"News fetch failed: {e}")
            return {
                "status": "ERROR",
                "error": f"News fetch failed: {e}",
            }

        articles = raw_news.get("articles", [])
        if not articles:
            return {
                "status": "OK",
                "articles": [],
                "summary": "No news available for this ticker.",
                "sentiment": "neutral",
            }

        processed = []

        for art in articles[:limit]:
            title = self._clean(art.get("title", ""))
            summary = self._clean(art.get("summary", ""))

            # Try LLM sentiment → fallback
            sent = self._llm_sentiment(title, summary)
            if not sent:
                sent = self._fallback_sentiment(title + " " + summary)

            processed.append({
                "title": title,
                "summary": summary,
                "source": art.get("source", "unknown"),
                "link": art.get("link", ""),
                "sentiment": sent,
            })

        # -----------------------------
        # Rank by relevance
        # simple heuristic: positive/negative before neutral
        # -----------------------------
        rank_order = {"positive": 1, "negative": 1, "neutral": 2}
        processed.sort(key=lambda x: rank_order.get(x["sentiment"], 3))

        # -----------------------------
        # Build summary
        # -----------------------------
        pos = sum(1 for a in processed if a["sentiment"] == "positive")
        neg = sum(1 for a in processed if a["sentiment"] == "negative")

        if pos > neg:
            overall = "positive"
        elif neg > pos:
            overall = "negative"
        else:
            overall = "neutral"

        return {
            "status": "OK",
            "articles": processed,
            "sentiment": overall,
            "counts": {
                "positive": pos,
                "negative": neg,
                "neutral": len(processed) - pos - neg
            }
        }
