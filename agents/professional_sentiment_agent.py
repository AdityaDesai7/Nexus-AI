# ==================== INSERT INTO trading_bot/agents/wrappers.py ====================

# New Imports needed:
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import json
import logging
import re
from trading_bot.agents.base_agent import BaseAgent

# Assuming ProfessionalSentimentLogic is correctly imported from a sibling file
try:
    from agents.professional_news_logic import ProfessionalSentimentLogic
except ImportError:
    # Adjust path if needed based on your file structure
    from trading_bot.agents.professional_news_logic import ProfessionalSentimentLogic 

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# LLM-POWERED SENTIMENT AGENT WRAPPER (NEW)
# ------------------------------------------------------------
class ProfessionalSentimentAgent(BaseAgent):
    """
    Uses InvestorGradeNewsFetcher for data and an LLM for sentiment scoring 
    and qualitative synthesis.
    """
    def __init__(self, name="sentiment", tools=None, llm=None):
        # BaseAgent handles tools and LLM
        super().__init__(name=name, tools=tools or [], llm=llm) 
        self.logic = ProfessionalSentimentLogic()
        self.max_articles = 5 # Limit LLM calls to 5 articles

    def plan(self, inp):
        # Simple plan: just need the ticker
        return {"action": "analyze_sentiment", "ticker": inp.get("ticker")}

    def _generate_sentiment_visualization(self, breakdown: dict, ticker: str) -> str:
        """Generates and saves a Bar chart of the sentiment breakdown."""
        labels = list(breakdown.keys())
        counts = [breakdown[k]['count'] for k in labels]
        
        if sum(counts) == 0:
            return ""

        colors = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'yellow'}
        
        # Filter out zero counts for cleaner visualization
        viz_data = [(l, c) for l, c in zip(labels, counts) if c > 0]
        viz_labels = [d[0] for d in viz_data]
        viz_counts = [d[1] for d in viz_data]
        viz_colors = [colors.get(l, 'gray') for l in viz_labels]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(viz_labels, viz_counts, color=viz_colors)
        
        plt.title(f"LLM Sentiment Breakdown for {ticker.upper()}", fontsize=14)
        plt.ylabel("Number of Articles Analyzed")
        
        # Add labels on bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), ha='center', va='bottom', fontsize=10)

        filename = f"{ticker.upper()}_sentiment_visual.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        return filename

    def act(self, plan):
        ticker = plan.get("ticker")
        if not ticker:
            return {"status": "ERROR", "agent": "sentiment", "error": "Ticker not provided."}

        logger.info(f"Sentiment Agent: Fetching and extracting articles for {ticker}...")
        
        # 1. Fetch Articles and Extract Content
        try:
            analyzable_articles = self.logic.get_analyzable_articles(ticker)
        except Exception as e:
            logger.error(f"News logic failed: {e}")
            return {"status": "ERROR", "agent": "sentiment", "error": f"News fetching failed: {e}"}

        if not analyzable_articles:
            return {"status": "NO_DATA", "ticker": ticker, "message": "No suitable articles found for LLM analysis."}

        # 2. LLM Analysis Loop
        sentiment_tallies = []
        detailed_analysis = []
        
        logger.info(f"Sentiment Agent: Sending {len(analyzable_articles)} articles to LLM for scoring...")

        for i, article in enumerate(analyzable_articles):
            # Focus the LLM on the title and the first 2000 characters of content
            content_snippet = article['content'][:2000] 
            
            llm_prompt = f"""
            TASK: Analyze the sentiment of the following news article about {ticker}.
            
            ---
            Article Title: {article['title']}
            ---
            Article Snippet: {content_snippet}
            ---
            
            Based *only* on the text provided, respond with a JSON object.
            
            1.  Determine the sentiment (POSITIVE, NEGATIVE, or NEUTRAL).
            2.  Provide a short, 1-2 sentence **Summary** of the article's core impact on the stock.
            
            JSON format: {{"sentiment": "SENTIMENT_HERE", "summary": "SUMMARY_HERE"}}
            """
            
            try:
                # Use your existing LLM wrapper (self.llm.ask)
                llm_response_text = self.llm.ask(llm_prompt)
                
                # Robustly extract JSON from the LLM's raw text response
                match = re.search(r'\{.*\}', llm_response_text, re.DOTALL)
                if match:
                    llm_data = json.loads(match.group(0))
                    sentiment = llm_data.get('sentiment', 'NEUTRAL').upper()
                    summary = llm_data.get('summary', 'LLM did not provide a clear summary.')
                else:
                    sentiment = 'NEUTRAL'
                    summary = f"LLM failed to return valid JSON. Raw response: {llm_response_text[:100]}..."

            except Exception as e:
                logger.error(f"LLM sentiment analysis failed for article {i+1}: {e}")
                sentiment = 'NEUTRAL'
                summary = "LLM analysis failed."

            sentiment_tallies.append(sentiment)
            detailed_analysis.append({
                "source": article['source'],
                "title": article['title'],
                "sentiment": sentiment,
                "llm_summary": summary,
                "url": article['url']
            })

        # 3. Aggregate Results and Generate Final Report
        sent_counts = Counter(sentiment_tallies)
        total = len(sentiment_tallies)
        
        breakdown = {
            'POSITIVE': {'count': sent_counts.get('POSITIVE', 0), 'percentage': f"{(sent_counts.get('POSITIVE', 0) / total) * 100:.1f}%"},
            'NEGATIVE': {'count': sent_counts.get('NEGATIVE', 0), 'percentage': f"{(sent_counts.get('NEGATIVE', 0) / total) * 100:.1f}%"},
            'NEUTRAL': {'count': sent_counts.get('NEUTRAL', 0), 'percentage': f"{(sent_counts.get('NEUTRAL', 0) / total) * 100:.1f}%"}
        }
        
        net_score = sent_counts.get('POSITIVE', 0) - sent_counts.get('NEGATIVE', 0)

        # 4. Generate Visualization
        viz_path = self._generate_sentiment_visualization(breakdown, ticker)
        
        # 5. LLM Synthesis (Final Qualitative Report)
        final_prompt = (
            f"Synthesize a final, holistic market sentiment report for {ticker} based on the following analysis of {total} news articles. "
            "Focus on the net mood and a final trading signal (Bullish, Neutral, or Bearish). "
            f"Breakdown: {json.dumps(breakdown)}. Net Score: {net_score}. "
            f"Top 3 articles: {json.dumps(detailed_analysis[:3])}"
        )
        final_summary = self.llm.ask(final_prompt)

        return {
            "status": "OK",
            "ticker": ticker,
            "total_articles_analyzed": total,
            "sentiment_breakdown": breakdown,
            "net_sentiment_score": net_score,
            "visualization_path": viz_path,
            "llm_qualitative_summary": final_summary,
            "detailed_article_analysis": detailed_analysis
        }