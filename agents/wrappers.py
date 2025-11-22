# # trading_bot/agents/wrappers.py
# """
# Wrapper layer converting pure-logic agents into BaseAgent-compatible agents.
# This exposes create_wrapped_agents() which the UI and AgentRunner expect.
# """

# import logging
# from typing import Dict, Any, Optional

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# # ------------------------------------------------------------
# # Robust Imports (works in both top-level and package execution)
# # ------------------------------------------------------------
# try:
#     from agents.base_agent import BaseAgent
#     from agents.technical_agent import TechnicalAnalysisAgent as TechnicalLogic
#     from agents.risk_agent import RiskManagementAgent as RiskLogic
#     from agents.portfolio_agent import PortfolioManagerAgent as PortfolioLogic
#     from agents.debate_agent import DebateAgent as DebateLogic
#     from agents.master_agent import MasterAgent as MasterLogic
#     from agents.news_agent import NewsAgent as NewsLogic
#     from tools.toolbox import TOOLS
# except Exception:
#     from trading_bot.agents.base_agent import BaseAgent
#     from trading_bot.agents.technical_agent import TechnicalAnalysisAgent as TechnicalLogic
#     from trading_bot.agents.risk_agent import RiskManagementAgent as RiskLogic
#     from trading_bot.agents.portfolio_agent import PortfolioManagerAgent as PortfolioLogic
#     from trading_bot.agents.debate_agent import DebateAgent as DebateLogic
#     from trading_bot.agents.master_agent import MasterAgent as MasterLogic
#     from trading_bot.agents.news_agent import NewsAgent as NewsLogic
#     from trading_bot.tools.toolbox import TOOLS


# # ------------------------------------------------------------
# # Helper: Normalize df
# # ------------------------------------------------------------
# def _extract_df(data):
#     import pandas as pd
#     if data is None:
#         return None
#     if isinstance(data, pd.DataFrame):
#         return data
#     if isinstance(data, dict) and "df" in data and isinstance(data["df"], pd.DataFrame):
#         return data["df"]
#     return None


# # ------------------------------------------------------------
# # TECHNICAL AGENT WRAPPER
# # ------------------------------------------------------------
# class TechnicalAgent(BaseAgent):
#     def __init__(self, name="technical", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         self.logic = TechnicalLogic()

#     def plan(self, user_input):
#         return {"action": "run", **user_input}

#     def act(self, plan):
#         ticker = plan.get("ticker")
#         start = plan.get("start")
#         end = plan.get("end")

#         # Try fetching df for consistency
#         try:
#             df = self.call_tool("fetch_price", ticker, start, end)
#         except Exception:
#             df = None

#         try:
#             out = self.logic.analyze(ticker, start_date=start, end_date=end)
#         except Exception as e:
#             logger.exception("Technical logic failed")
#             return {"status": "ERROR", "agent": "technical", "error": str(e)}

#         return {"ticker": ticker, "technical": out, "df": _extract_df(df)}


# # ------------------------------------------------------------
# # RISK AGENT WRAPPER
# # ------------------------------------------------------------
# class RiskAgent(BaseAgent):
#     def __init__(self, name="risk", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         self.logic = RiskLogic()

#     def plan(self, user_input):
#         return {"action": "run", **user_input}

#     def act(self, plan):
#         ticker = plan.get("ticker")
#         start = plan.get("start")
#         end = plan.get("end")
#         df = plan.get("df")

#         if df is None:
#             try:
#                 df = self.call_tool("fetch_price", ticker, start, end)
#             except Exception:
#                 df = None

#         dp = _extract_df(df)

#         try:
#             current_price = float(dp["Close"].iloc[-1]) if dp is not None else plan.get("current_price")
#         except Exception:
#             current_price = plan.get("current_price", 0.0)

#         try:
#             out = self.logic.evaluate(
#                 ticker=ticker,
#                 df=dp,
#                 current_price=current_price,
#                 technical_confidence=plan.get("technical_confidence", 50),
#                 sentiment_confidence=plan.get("sentiment_confidence", 50)
#             )
#         except Exception as e:
#             logger.exception("Risk logic failed")
#             return {"status": "ERROR", "agent": "risk", "error": str(e)}

#         return {"ticker": ticker, "risk": out, "df": dp}


# # ------------------------------------------------------------
# # PORTFOLIO AGENT WRAPPER
# # ------------------------------------------------------------
# class PortfolioAgent(BaseAgent):
#     def __init__(self, name="portfolio", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         self.logic = PortfolioLogic()

#     def plan(self, user_input):
#         return {"action": "run", **user_input}

#     def act(self, plan):
#         ticker = plan.get("ticker")
#         df = _extract_df(plan.get("df"))

#         if df is None:
#             try:
#                 df = _extract_df(self.call_tool("fetch_price", ticker,
#                                                 plan.get("start"), plan.get("end")))
#             except Exception:
#                 df = None

#         try:
#             current_price = float(df["Close"].iloc[-1]) if df is not None else plan.get("current_price")
#         except Exception:
#             current_price = plan.get("current_price", 0.0)

#         try:
#             action, quantity, meta = self.logic.decide(
#                 ticker=ticker,
#                 current_price=current_price,
#                 technical_signal=plan.get("technical_signal", {}),
#                 sentiment_signal=plan.get("sentiment_signal"),
#                 risk_metrics=plan.get("risk_metrics", {}),
#                 portfolio_state=plan.get("portfolio_state")
#             )
#         except Exception as e:
#             logger.exception("Portfolio logic failed")
#             return {"status": "ERROR", "agent": "portfolio", "error": str(e)}

#         alloc = self.logic.get_allocation_metrics(action, quantity, current_price)

#         return {
#             "ticker": ticker,
#             "action": action,
#             "quantity": int(quantity or 0),
#             "allocation": alloc,
#             "meta": meta if isinstance(meta, dict) else {"raw": str(meta)},
#             "df": df,
#         }


# # ------------------------------------------------------------
# # DEBATE AGENT WRAPPER
# # ------------------------------------------------------------
# class DebateAgent(BaseAgent):
#     def __init__(self, name="debate", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         self.logic = DebateLogic(llm=llm)

#     def plan(self, inp):
#         return {"action": "debate", **inp}

#     def act(self, plan):
#         try:
#             out = self.logic.debate(
#                 ticker=plan.get("ticker"),
#                 technical_result=plan.get("technical_result", {}),
#                 risk_metrics=plan.get("risk_metrics", {}),
#                 price_data=plan.get("df"),
#                 sentiment_score=plan.get("sentiment_score", 50)
#             )
#         except Exception as e:
#             logger.exception("Debate logic failed")
#             return {"status": "ERROR", "agent": "debate", "error": str(e)}

#         return {"ticker": plan.get("ticker"), "debate": out}


# # ------------------------------------------------------------
# # MASTER AGENT WRAPPER
# # ------------------------------------------------------------
# class MasterAgent(BaseAgent):
#     def __init__(self, name="master", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         self.logic = MasterLogic(llm=llm)

#     def plan(self, inp):
#         return {"action": "synthesize", **inp}

#     def act(self, plan):
#         try:
#             out = self.logic.synthesize(
#                 ticker=plan.get("ticker"),
#                 technical_result=plan.get("technical_result", {}),
#                 sentiment_result=plan.get("sentiment_result", {}),
#                 risk_metrics=plan.get("risk_metrics", {}),
#                 portfolio_metrics=plan.get("portfolio_metrics", {}),
#                 current_price=plan.get("current_price", 0.0),
#             )
#         except Exception as e:
#             logger.exception("Master logic failed")
#             return {"status": "ERROR", "agent": "master", "error": str(e)}

#         return {"ticker": plan.get("ticker"), "master": out}


# # ------------------------------------------------------------
# # NEWS AGENT WRAPPER
# # ------------------------------------------------------------
# class NewsAgent(BaseAgent):
#     def __init__(self, name="news", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         self.logic = NewsLogic()

#     def plan(self, inp):
#         return {"action": "fetch", **inp}

#     def act(self, plan):
#         ticker = plan.get("ticker")
#         limit = plan.get("limit", 15)

#         try:
#             articles = self.call_tool("fetch_news", ticker, limit)
#         except Exception:
#             articles = []

#         summaries = []
#         for a in (articles or [])[:limit]:
#             if not isinstance(a, dict):
#                 continue
#             summaries.append({
#                 "title": a.get("title") or a.get("headline") or "",
#                 "url": a.get("url"),
#                 "summary": a.get("summary") or a.get("description") or "",
#                 "source": a.get("source") or a.get("provider") or "unknown",
#             })

#         return {"ticker": ticker, "articles": articles, "summaries": summaries}


# # ------------------------------------------------------------
# # MAIN FACTORY — used by AgentRunner + UI
# # ------------------------------------------------------------
# def create_wrapped_agents(tools=None, llm=None) -> Dict[str, BaseAgent]:
#     tools = tools or TOOLS
#     return {
#         "technical": TechnicalAgent(tools=tools, llm=llm),
#         "risk": RiskAgent(tools=tools, llm=llm),
#         "portfolio": PortfolioAgent(tools=tools, llm=llm),
#         "debate": DebateAgent(tools=tools, llm=llm),
#         "master": MasterAgent(tools=tools, llm=llm),
#         "news": NewsAgent(tools=tools, llm=llm),
#     }

"""
Wrapper layer converting pure-logic agents into BaseAgent-compatible agents.
This exposes create_wrapped_agents() which the UI and AgentRunner expect.
"""

import logging
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------
# ROBUST IMPORTS with better error handling
# ------------------------------------------------------------

def safe_import(module_name, class_name, fallback_module=None):
    """Safely import classes with comprehensive error handling"""
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        logger.warning(f"Primary import failed for {class_name}: {e}")
        if fallback_module:
            try:
                module = __import__(fallback_module, fromlist=[class_name])
                return getattr(module, class_name)
            except ImportError:
                logger.error(f"Fallback import also failed for {class_name}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error importing {class_name}: {e}")
        return None

# Import BaseAgent first as it's critical
try:
    from agents.base_agent import BaseAgent
except ImportError:
    try:
        from trading_bot.agents.base_agent import BaseAgent
    except ImportError:
        logger.critical("❌ Cannot import BaseAgent - this will break everything!")
        raise

# Import all agent logic classes with fallbacks
TechnicalLogic = safe_import('agents.technical_agent', 'TechnicalAnalysisAgent', 'trading_bot.agents.technical_agent')
RiskLogic = safe_import('agents.risk_agent', 'RiskManagementAgent', 'trading_bot.agents.risk_agent')
PortfolioLogic = safe_import('agents.portfolio_agent', 'PortfolioManagerAgent', 'trading_bot.agents.portfolio_agent')
DebateLogic = safe_import('agents.debate_agent', 'DebateAgent', 'trading_bot.agents.debate_agent')
MasterLogic = safe_import('agents.master_agent', 'MasterAgent', 'trading_bot.agents.master_agent')
NewsLogic = safe_import('agents.news_agent', 'NewsAgent', 'trading_bot.agents.news_agent')
SentimentLogic = safe_import('agents.professional_sentiment_agent', 'ProfessionalSentimentAgent', 'trading_bot.agents.professional_sentiment_agent')

try:
    from tools.toolbox import TOOLS
except ImportError:
    try:
        from trading_bot.tools.toolbox import TOOLS
    except ImportError:
        logger.critical("❌ Cannot import TOOLS - agent functionality will be limited")
        TOOLS = {}

# ------------------------------------------------------------
# Helper: Normalize df
# ------------------------------------------------------------
def _extract_df(data):
    import pandas as pd
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict) and "df" in data and isinstance(data["df"], pd.DataFrame):
        return data["df"]
    return None


# ------------------------------------------------------------
# TECHNICAL AGENT WRAPPER - FIXED with better error handling
# ------------------------------------------------------------
class TechnicalAgent(BaseAgent):
    def __init__(self, name="technical", tools=None, llm=None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        if TechnicalLogic:
            self.logic = TechnicalLogic()
        else:
            logger.error("❌ TechnicalLogic not available - creating dummy implementation")
            self.logic = None

    def plan(self, user_input):
        return {"action": "run", **user_input}

    def act(self, plan):
        ticker = plan.get("ticker")
        start = plan.get("start")
        end = plan.get("end")

        # Try fetching df for consistency
        try:
            df = self.call_tool("fetch_price", ticker, start, end)
        except Exception as e:
            logger.warning(f"fetch_price failed: {e}")
            df = None

        try:
            if self.logic:
                out = self.logic.analyze(ticker, start_date=start, end_date=end)
            else:
                # Fallback implementation
                out = {
                    "status": "FALLBACK",
                    "message": "Technical logic not available",
                    "indicators": {},
                    "signals": []
                }
        except Exception as e:
            logger.exception("Technical logic failed")
            return {
                "status": "ERROR", 
                "agent": "technical", 
                "error": str(e),
                "ticker": ticker
            }

        return {
            "ticker": ticker, 
            "technical": out, 
            "df": _extract_df(df),
            "status": "SUCCESS"
        }


# ------------------------------------------------------------
# RISK AGENT WRAPPER - FIXED with better error handling
# ------------------------------------------------------------
class RiskAgent(BaseAgent):
    def __init__(self, name="risk", tools=None, llm=None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        if RiskLogic:
            self.logic = RiskLogic()
        else:
            logger.error("❌ RiskLogic not available - creating dummy implementation")
            self.logic = None

    def plan(self, user_input):
        return {"action": "run", **user_input}

    def act(self, plan):
        ticker = plan.get("ticker")
        start = plan.get("start")
        end = plan.get("end")
        df = plan.get("df")

        if df is None:
            try:
                df = self.call_tool("fetch_price", ticker, start, end)
            except Exception:
                df = None

        dp = _extract_df(df)

        try:
            current_price = float(dp["Close"].iloc[-1]) if dp is not None else plan.get("current_price")
        except Exception:
            current_price = plan.get("current_price", 0.0)

        try:
            if self.logic:
                out = self.logic.evaluate(
                    ticker=ticker,
                    df=dp,
                    current_price=current_price,
                    technical_confidence=plan.get("technical_confidence", 50),
                    sentiment_confidence=plan.get("sentiment_confidence", 50)
                )
            else:
                # Fallback implementation
                out = {
                    "status": "FALLBACK",
                    "risk_level": "MEDIUM",
                    "message": "Risk logic not available"
                }
        except Exception as e:
            logger.exception("Risk logic failed")
            return {"status": "ERROR", "agent": "risk", "error": str(e)}

        return {
            "ticker": ticker, 
            "risk": out, 
            "df": dp,
            "status": "SUCCESS"
        }


# ------------------------------------------------------------
# PORTFOLIO AGENT WRAPPER - FIXED with better error handling
# ------------------------------------------------------------
class PortfolioAgent(BaseAgent):
    def __init__(self, name="portfolio", tools=None, llm=None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        if PortfolioLogic:
            self.logic = PortfolioLogic()
        else:
            logger.error("❌ PortfolioLogic not available - creating dummy implementation")
            self.logic = None

    def plan(self, user_input):
        return {"action": "run", **user_input}

    def act(self, plan):
        ticker = plan.get("ticker")
        df = _extract_df(plan.get("df"))

        if df is None:
            try:
                df = _extract_df(self.call_tool("fetch_price", ticker,
                                                plan.get("start"), plan.get("end")))
            except Exception:
                df = None

        try:
            current_price = float(df["Close"].iloc[-1]) if df is not None else plan.get("current_price")
        except Exception:
            current_price = plan.get("current_price", 0.0)

        try:
            if self.logic:
                action, quantity, meta = self.logic.decide(
                    ticker=ticker,
                    current_price=current_price,
                    technical_signal=plan.get("technical_signal", {}),
                    sentiment_signal=plan.get("sentiment_signal"),
                    risk_metrics=plan.get("risk_metrics", {}),
                    portfolio_state=plan.get("portfolio_state")
                )
                alloc = self.logic.get_allocation_metrics(action, quantity, current_price)
            else:
                # Fallback implementation
                action = "HOLD"
                quantity = 0
                meta = {"message": "Portfolio logic not available"}
                alloc = {}
        except Exception as e:
            logger.exception("Portfolio logic failed")
            return {"status": "ERROR", "agent": "portfolio", "error": str(e)}

        return {
            "ticker": ticker,
            "action": action,
            "quantity": int(quantity or 0),
            "allocation": alloc,
            "meta": meta if isinstance(meta, dict) else {"raw": str(meta)},
            "df": df,
            "status": "SUCCESS"
        }


# ------------------------------------------------------------
# DEBATE AGENT WRAPPER - FIXED with better error handling
# ------------------------------------------------------------
class DebateAgent(BaseAgent):
    def __init__(self, name="debate", tools=None, llm=None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        if DebateLogic:
            self.logic = DebateLogic(llm=llm)
        else:
            logger.error("❌ DebateLogic not available - creating dummy implementation")
            self.logic = None

    def plan(self, inp):
        return {"action": "debate", **inp}

    def act(self, plan):
        try:
            if self.logic:
                out = self.logic.debate(
                    ticker=plan.get("ticker"),
                    technical_result=plan.get("technical_result", {}),
                    risk_metrics=plan.get("risk_metrics", {}),
                    price_data=plan.get("df"),
                    sentiment_score=plan.get("sentiment_score", 50)
                )
            else:
                # Fallback implementation
                out = {
                    "status": "FALLBACK",
                    "consensus": "NEUTRAL",
                    "message": "Debate logic not available",
                    "arguments": []
                }
        except Exception as e:
            logger.exception("Debate logic failed")
            return {"status": "ERROR", "agent": "debate", "error": str(e)}

        return {
            "ticker": plan.get("ticker"), 
            "debate": out,
            "status": "SUCCESS"
        }


# ------------------------------------------------------------
# MASTER AGENT WRAPPER - FIXED with better error handling
# ------------------------------------------------------------
class MasterAgent(BaseAgent):
    def __init__(self, name="master", tools=None, llm=None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        if MasterLogic:
            self.logic = MasterLogic(llm=llm)
        else:
            logger.error("❌ MasterLogic not available - creating dummy implementation")
            self.logic = None

    def plan(self, inp):
        return {"action": "synthesize", **inp}

    def act(self, plan):
        try:
            if self.logic:
                out = self.logic.synthesize(
                    ticker=plan.get("ticker"),
                    technical_result=plan.get("technical_result", {}),
                    sentiment_result=plan.get("sentiment_result", {}),
                    risk_metrics=plan.get("risk_metrics", {}),
                    portfolio_metrics=plan.get("portfolio_metrics", {}),
                    current_price=plan.get("current_price", 0.0),
                )
            else:
                # Fallback implementation
                out = {
                    "status": "FALLBACK",
                    "action": "HOLD",
                    "confidence": 50,
                    "reasoning": "Master logic not available",
                    "risk_management": {}
                }
        except Exception as e:
            logger.exception("Master logic failed")
            return {
                "ticker": plan.get("ticker"), 
                "status": "ERROR", 
                "agent": "master", 
                "error": str(e)
            }

        return {
            "ticker": plan.get("ticker"), 
            "master": out,
            "status": "SUCCESS"
        }


# ------------------------------------------------------------
# NEWS AGENT WRAPPER - FIXED with better error handling
# ------------------------------------------------------------
class NewsAgent(BaseAgent):
    def __init__(self, name="news", tools=None, llm=None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        if NewsLogic:
            self.logic = NewsLogic()
        else:
            logger.error("❌ NewsLogic not available - creating dummy implementation")
            self.logic = None

    def plan(self, inp):
        return {"action": "fetch", **inp}

    def act(self, plan):
        ticker = plan.get("ticker")
        limit = plan.get("limit", 15)

        try:
            articles = self.call_tool("fetch_news", ticker, limit)
        except Exception as e:
            logger.warning(f"fetch_news failed: {e}")
            articles = []

        summaries = []
        for a in (articles or [])[:limit]:
            if not isinstance(a, dict):
                continue
            summaries.append({
                "title": a.get("title") or a.get("headline") or "",
                "url": a.get("url"),
                "summary": a.get("summary") or a.get("description") or "",
                "source": a.get("source") or a.get("provider") or "unknown",
            })

        return {
            "ticker": ticker, 
            "articles": articles, 
            "summaries": summaries,
            "status": "SUCCESS"
        }


# ------------------------------------------------------------
# SENTIMENT AGENT WRAPPER - FIXED with better error handling
# ------------------------------------------------------------
class ProfessionalSentimentAgent(BaseAgent):
    def __init__(self, name="sentiment", tools=None, llm=None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        if SentimentLogic:
            self.logic = SentimentLogic()
        else:
            logger.error("❌ SentimentLogic not available - creating dummy implementation")
            self.logic = None

    def plan(self, inp):
        return {"action": "analyze", **inp}

    def act(self, plan):
        ticker = plan.get("ticker")
        try:
            if self.logic:
                # Call the sentiment analysis logic
                result = self.logic.analyze(ticker)
            else:
                # Fallback implementation
                result = {
                    "status": "FALLBACK",
                    "sentiment": "NEUTRAL",
                    "confidence": 0.5,
                    "message": "Sentiment logic not available"
                }
        except Exception as e:
            logger.exception("Sentiment logic failed")
            return {"status": "ERROR", "agent": "sentiment", "error": str(e)}

        return {
            "ticker": ticker,
            "sentiment": result,
            "status": "SUCCESS"
        }


# ------------------------------------------------------------
# MAIN FACTORY — used by AgentRunner + UI
# ------------------------------------------------------------
def create_wrapped_agents(tools=None, llm=None) -> Dict[str, BaseAgent]:
    """
    Factory function that creates all wrapped agents.
    This is the main entry point for AgentRunner.
    """
    tools = tools or TOOLS
    logger.info("🏭 Creating wrapped agents...")
    
    agents = {}
    
    # Define all agent classes to create
    agent_classes = {
        "technical": TechnicalAgent,
        "risk": RiskAgent,
        "portfolio": PortfolioAgent,
        "debate": DebateAgent,
        "master": MasterAgent,
        "news": NewsAgent,
        "sentiment": ProfessionalSentimentAgent,
    }
    
    # Create each agent with error handling
    for name, agent_class in agent_classes.items():
        try:
            agent = agent_class(tools=tools, llm=llm)
            agents[name] = agent
            logger.info(f"   ✅ Created agent: {name}")
        except Exception as e:
            logger.error(f"   ❌ Failed to create agent {name}: {e}")
            # Don't add failed agents
    
    logger.info(f"🎯 Successfully created {len(agents)} agents: {list(agents.keys())}")
    return agents


# ------------------------------------------------------------
# ALTERNATIVE FACTORY for backward compatibility
# ------------------------------------------------------------
def create_agent_registry(tools=None, llm=None):
    """Alternative factory function name for compatibility"""
    return create_wrapped_agents(tools, llm)


# ------------------------------------------------------------
# DIRECT REGISTRATION FUNCTION
# ------------------------------------------------------------
def register_agents_directly(runner, tools=None, llm=None):
    """
    Directly register agents to an AgentRunner instance.
    Useful for manual registration scenarios.
    """
    agents = create_wrapped_agents(tools, llm)
    for name, agent in agents.items():
        try:
            runner.register(name, agent)
            logger.info(f"📋 Directly registered: {name}")
        except Exception as e:
            logger.error(f"Failed to directly register {name}: {e}")
    
    return list(agents.keys())