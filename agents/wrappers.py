# trading_bot/agents/wrappers.py
"""
Wrappers for pure-logic agents. Each wrapper inherits BaseAgent and calls tools via self.call_tool(...).
This module exports create_wrapped_agents(tools) which the UI expects.
"""

import logging
from typing import Dict, Any, Optional
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Flexible imports to support both running as package and as top-level script
try:
    from agents.base_agent import BaseAgent
    from agents.technical_agent import TechnicalAnalysisAgent as TechnicalLogic
    from agents.risk_agent import RiskManagementAgent as RiskLogic
    from agents.portfolio_agent import PortfolioManagerAgent as PortfolioLogic
    from agents.debate_agent import DebateAgent as DebateLogic
    from agents.master_agent import MasterAgent as MasterLogic
    from agents.news_agent import NewsAgent as NewsLogic
    # tools registry if available
    try:
        from tools.toolbox import TOOLS
    except Exception:
        TOOLS = {}
except Exception:
    from trading_bot.agents.base_agent import BaseAgent
    from trading_bot.agents.technical_agent import TechnicalAnalysisAgent as TechnicalLogic
    from trading_bot.agents.risk_agent import RiskManagementAgent as RiskLogic
    from trading_bot.agents.portfolio_agent import PortfolioManagerAgent as PortfolioLogic
    from trading_bot.agents.debate_agent import DebateAgent as DebateLogic
    from trading_bot.agents.master_agent import MasterAgent as MasterLogic
    from trading_bot.agents.news_agent import NewsAgent as NewsLogic
    try:
        from trading_bot.tools.toolbox import TOOLS
    except Exception:
        TOOLS = {}

# Helper to normalize tool outputs
def _extract_df(maybe_df_or_dict):
    import pandas as pd
    if maybe_df_or_dict is None:
        return None
    if isinstance(maybe_df_or_dict, pd.DataFrame):
        return maybe_df_or_dict
    if isinstance(maybe_df_or_dict, dict):
        if "df" in maybe_df_or_dict and isinstance(maybe_df_or_dict["df"], pd.DataFrame):
            return maybe_df_or_dict["df"]
        # Some tools return list/dict - not df
    return None

# -------------------------
# Agent wrappers
# -------------------------
class TechnicalAgent(BaseAgent):
    def __init__(self, name: str="technical", tools: Optional[Dict[str, Any]] = None, llm: Any = None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        self.logic = TechnicalLogic()

    def plan(self, user_input):
        return {"action": "analyze", **user_input}

    def act(self, plan):
        ticker = plan.get("ticker")
        start = plan.get("start")
        end = plan.get("end")
        # fetch price for auditability
        try:
            df = self.call_tool("fetch_price", ticker, start, end)
        except Exception as e:
            logger.warning("TechnicalAgent: fetch_price failed: %s", e)
            df = None
        tech_out = self.logic.analyze(ticker, start_date=start, end_date=end)
        return {"ticker": ticker, "technical": tech_out if isinstance(tech_out, dict) else dict(tech_out)}

class RiskAgent(BaseAgent):
    def __init__(self, name: str="risk", tools: Optional[Dict[str, Any]] = None, llm: Any = None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        self.logic = RiskLogic()

    def plan(self, user_input):
        return {"action": "evaluate", **user_input}

    def act(self, plan):
        ticker = plan.get("ticker")
        start = plan.get("start")
        end = plan.get("end")
        current_price = plan.get("current_price")
        # attempt to fetch df if provided or via tools
        df = plan.get("df")
        if df is None:
            try:
                df = self.call_tool("fetch_price", ticker, start, end)
            except Exception as e:
                logger.warning("RiskAgent: fetch_price failed: %s", e)
                df = None
        try:
            risk_out = self.logic.evaluate(
                ticker=ticker,
                df=df,
                current_price=current_price or (df["Close"].iloc[-1] if _extract_df(df) is not None else None),
                technical_confidence=plan.get("technical_confidence", 50),
                sentiment_confidence=plan.get("sentiment_confidence", 50)
            )
        except Exception as e:
            logger.exception("RiskAgent logic failed")
            return {"ticker": ticker, "status": "ERROR", "agent": "risk", "error": str(e)}
        return {"ticker": ticker, "risk": risk_out}

class PortfolioAgent(BaseAgent):
    def __init__(self, name: str="portfolio", tools: Optional[Dict[str, Any]] = None, llm: Any = None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        self.logic = PortfolioLogic()

    def plan(self, user_input):
        return {"action": "decide", **user_input}

    def act(self, plan):
        ticker = plan.get("ticker")
        start = plan.get("start")
        end = plan.get("end")
        current_price = plan.get("current_price")
        df = plan.get("df")
        if df is None:
            try:
                df = self.call_tool("fetch_price", ticker, start, end)
            except Exception as e:
                logger.warning("PortfolioAgent: fetch_price failed: %s", e)
                df = None

        # if current_price still None, try extract from df
        try:
            if current_price is None and _extract_df(df) is not None:
                current_price = float(df["Close"].iloc[-1])
        except Exception:
            current_price = current_price or 0.0

        technical_signal = plan.get("technical_signal", {})
        risk_metrics = plan.get("risk_metrics", {})

        try:
            action, quantity, metadata = self.logic.decide(
                ticker=ticker,
                current_price=current_price,
                technical_signal=technical_signal,
                sentiment_signal=plan.get("sentiment_signal"),
                risk_metrics=risk_metrics,
                portfolio_state=plan.get("portfolio_state")
            )
        except Exception as e:
            logger.exception("PortfolioAgent logic failed")
            return {"ticker": ticker, "status": "ERROR", "agent": "portfolio", "error": str(e)}

        alloc_metrics = self.logic.get_allocation_metrics(action, quantity, current_price)
        result = {
            "ticker": ticker,
            "action": action,
            "quantity": int(quantity if quantity is not None else 0),
            "meta": metadata if isinstance(metadata, dict) else {"meta": str(metadata)},
            "allocation": alloc_metrics,
            "df": df if _extract_df(df) is not None else None
        }
        return result

class DebateAgent(BaseAgent):
    def __init__(self, name: str="debate", tools: Optional[Dict[str, Any]] = None, llm: Any = None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        self.logic = DebateLogic()

    def plan(self, user_input):
        return {"action": "debate", **user_input}

    def act(self, plan):
        ticker = plan.get("ticker")
        technical_result = plan.get("technical_result", {})
        risk_metrics = plan.get("risk_metrics", {})
        df = plan.get("df")
        if df is None:
            try:
                df = self.call_tool("fetch_price", ticker, plan.get("start"), plan.get("end"))
            except Exception:
                df = None
        try:
            debate_out = self.logic.debate(ticker=ticker, technical_result=technical_result, risk_metrics=risk_metrics, price_data=df, sentiment_score=plan.get("sentiment_score", 50))
        except Exception as e:
            logger.exception("DebateAgent logic failed")
            return {"ticker": ticker, "status": "ERROR", "agent": "debate", "error": str(e)}
        return {"ticker": ticker, "debate": debate_out}

class MasterAgent(BaseAgent):
    def __init__(self, name: str="master", tools: Optional[Dict[str, Any]] = None, llm: Any = None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        self.logic = MasterLogic()

    def plan(self, user_input):
        return {"action": "synthesize", **user_input}

    def act(self, plan):
        ticker = plan.get("ticker")
        technical_result = plan.get("technical_result", {})
        sentiment_result = plan.get("sentiment_result", {})
        risk_metrics = plan.get("risk_metrics", {})
        portfolio_metrics = plan.get("portfolio_metrics", {})
        current_price = plan.get("current_price")
        if current_price is None:
            try:
                df = self.call_tool("fetch_price", ticker, plan.get("start"), plan.get("end"))
                if isinstance(df, dict) and "df" in df:
                    current_price = float(df["df"]["Close"].iloc[-1])
                elif hasattr(df, "iloc"):
                    current_price = float(df["Close"].iloc[-1])
            except Exception:
                current_price = current_price or 0.0

        try:
            master_out = self.logic.synthesize(
                ticker=ticker,
                technical_result=technical_result,
                sentiment_result=sentiment_result,
                risk_metrics=risk_metrics,
                portfolio_metrics=portfolio_metrics,
                current_price=current_price
            )
        except Exception as e:
            logger.exception("MasterAgent logic failed")
            return {"ticker": ticker, "status": "ERROR", "agent": "master", "error": str(e)}

        return {"ticker": ticker, "master": master_out}

class NewsAgent(BaseAgent):
    def __init__(self, name: str="news", tools: Optional[Dict[str, Any]] = None, llm: Any = None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        self.logic = NewsLogic()

    def plan(self, user_input):
        return {"action": "fetch_news", **user_input}

    def act(self, plan):
        ticker = plan.get("ticker")
        limit = plan.get("limit", 15)
        try:
            articles = self.call_tool("fetch_news", ticker, limit)
        except Exception as e:
            logger.exception("NewsAgent fetch failed")
            articles = []
        # Normalize list of articles to common shape
        summaries = []
        for art in (articles or [])[:limit]:
            if not isinstance(art, dict):
                continue
            summaries.append({
                "title": art.get("title") or art.get("headline") or "",
                "url": art.get("url"),
                "summary": art.get("summary") or art.get("description") or "",
                "source": art.get("source") or art.get("provider") or "unknown"
            })
        return {"ticker": ticker, "count": len(articles), "articles": articles, "summaries": summaries}

def create_wrapped_agents(tools: Optional[Dict[str, Any]] = None, llm: Any = None) -> Dict[str, BaseAgent]:
    tools = tools or TOOLS
    inst = {
        "technical": TechnicalAgent(tools=tools, llm=llm),
        "risk": RiskAgent(tools=tools, llm=llm),
        "portfolio": PortfolioAgent(tools=tools, llm=llm),
        "debate": DebateAgent(tools=tools, llm=llm),
        "master": MasterAgent(tools=tools, llm=llm),
        "news": NewsAgent(tools=tools, llm=llm)
    }
    return inst
