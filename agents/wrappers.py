
# """
# Wrapper layer converting pure-logic agents into BaseAgent-compatible agents.
# This exposes create_wrapped_agents() which the UI and AgentRunner expect.
# """

# import logging
# from typing import Dict, Any, Optional
# import json

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# # ------------------------------------------------------------
# # ROBUST IMPORTS with better error handling
# # ------------------------------------------------------------

# def safe_import(module_name, class_name, fallback_module=None):
#     """Safely import classes with comprehensive error handling"""
#     try:
#         module = __import__(module_name, fromlist=[class_name])
#         return getattr(module, class_name)
#     except ImportError as e:
#         logger.warning(f"Primary import failed for {class_name}: {e}")
#         if fallback_module:
#             try:
#                 module = __import__(fallback_module, fromlist=[class_name])
#                 return getattr(module, class_name)
#             except ImportError:
#                 logger.error(f"Fallback import also failed for {class_name}")
#         return None
#     except Exception as e:
#         logger.error(f"Unexpected error importing {class_name}: {e}")
#         return None

# # Import BaseAgent first as it's critical
# try:
#     from agents.base_agent import BaseAgent
# except ImportError:
#     try:
#         from trading_bot.agents.base_agent import BaseAgent
#     except ImportError:
#         logger.critical("❌ Cannot import BaseAgent - this will break everything!")
#         raise

# # Import all agent logic classes with fallbacks
# TechnicalLogic = safe_import('agents.technical_agent', 'TechnicalAnalysisAgent', 'trading_bot.agents.technical_agent')
# RiskLogic = safe_import('agents.risk_agent', 'RiskManagementAgent', 'trading_bot.agents.risk_agent')
# PortfolioLogic = safe_import('agents.portfolio_agent', 'PortfolioManagerAgent', 'trading_bot.agents.portfolio_agent')
# DebateLogic = safe_import('agents.debate_agent', 'DebateAgent', 'trading_bot.agents.debate_agent')
# MasterLogic = safe_import('agents.master_agent', 'MasterAgent', 'trading_bot.agents.master_agent')
# NewsLogic = safe_import('agents.news_agent', 'SentimentAnalysisAgent', 'trading_bot.agents.news_agent')
# SentimentLogic = safe_import('agents.sentiment_agent', 'SentimentAnalysisAgent', 'trading_bot.agents.sentiment_agent')

# try:
#     from tools.toolbox import TOOLS
# except ImportError:
#     try:
#         from trading_bot.tools.toolbox import TOOLS
#     except ImportError:
#         logger.critical("❌ Cannot import TOOLS - agent functionality will be limited")
#         TOOLS = {}

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
# # TECHNICAL AGENT WRAPPER - FIXED with better error handling
# # ------------------------------------------------------------
# class TechnicalAgent(BaseAgent):
#     def __init__(self, name="technical", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         print("technical agent init called from wrappers")
#         if TechnicalLogic:
#             self.logic = TechnicalLogic()
#         else:
#             logger.error("❌ TechnicalLogic not available - creating dummy implementation")
#             self.logic = None

#     def plan(self, user_input):
#         return {"action": "run", **user_input}

#     def act(self, plan):
#         ticker = plan.get("ticker")
#         start = plan.get("start")
#         end = plan.get("end")

#         # Try fetching df for consistency
#         try:
#             df = self.call_tool("fetch_price", ticker, start, end)
#         except Exception as e:
#             logger.warning(f"fetch_price failed: {e}")
#             df = None

#         try:
#             if self.logic:
#                 out = self.logic.analyze(ticker, start_date=start, end_date=end)
#             else:
#                 # Fallback implementation
#                 out = {
#                     "status": "FALLBACK",
#                     "message": "Technical logic not available",
#                     "indicators": {},
#                     "signals": []
#                 }
#         except Exception as e:
#             logger.exception("Technical logic failed")
#             return {
#                 "status": "ERROR", 
#                 "agent": "technical", 
#                 "error": str(e),
#                 "ticker": ticker
#             }

#         return {
#             "ticker": ticker, 
#             "technical": out, 
#             "df": _extract_df(df),
#             "status": "SUCCESS"
#         }


# # ------------------------------------------------------------
# # RISK AGENT WRAPPER - FIXED with better error handling
# # ------------------------------------------------------------
# class RiskAgent(BaseAgent):
#     def __init__(self, name="risk", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         if RiskLogic:
#             self.logic = RiskLogic()
#         else:
#             logger.error("❌ RiskLogic not available - creating dummy implementation")
#             self.logic = None

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
#             if self.logic:
#                 out = self.logic.evaluate(
#                     ticker=ticker,
#                     df=dp,
#                     current_price=current_price,
#                     technical_confidence=plan.get("technical_confidence", 50),
#                     sentiment_confidence=plan.get("sentiment_confidence", 50)
#                 )
#             else:
#                 # Fallback implementation
#                 out = {
#                     "status": "FALLBACK",
#                     "risk_level": "MEDIUM",
#                     "message": "Risk logic not available"
#                 }
#         except Exception as e:
#             logger.exception("Risk logic failed")
#             return {"status": "ERROR", "agent": "risk", "error": str(e)}

#         return {
#             "ticker": ticker, 
#             "risk": out, 
#             "df": dp,
#             "status": "SUCCESS"
#         }


# # ------------------------------------------------------------
# # PORTFOLIO AGENT WRAPPER - FIXED with better error handling
# # ------------------------------------------------------------
# class PortfolioAgent(BaseAgent):
#     def __init__(self, name="portfolio", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         if PortfolioLogic:
#             self.logic = PortfolioLogic()
#         else:
#             logger.error("❌ PortfolioLogic not available - creating dummy implementation")
#             self.logic = None

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
#             if self.logic:
#                 action, quantity, meta = self.logic.decide(
#                     ticker=ticker,
#                     current_price=current_price,
#                     technical_signal=plan.get("technical_signal", {}),
#                     sentiment_signal=plan.get("sentiment_signal"),
#                     risk_metrics=plan.get("risk_metrics", {}),
#                     portfolio_state=plan.get("portfolio_state")
#                 )
#                 alloc = self.logic.get_allocation_metrics(action, quantity, current_price)
#             else:
#                 # Fallback implementation
#                 action = "HOLD"
#                 quantity = 0
#                 meta = {"message": "Portfolio logic not available"}
#                 alloc = {}
#         except Exception as e:
#             logger.exception("Portfolio logic failed")
#             return {"status": "ERROR", "agent": "portfolio", "error": str(e)}

#         return {
#             "ticker": ticker,
#             "action": action,
#             "quantity": int(quantity or 0),
#             "allocation": alloc,
#             "meta": meta if isinstance(meta, dict) else {"raw": str(meta)},
#             "df": df,
#             "status": "SUCCESS"
#         }


# # ------------------------------------------------------------
# # DEBATE AGENT WRAPPER - FIXED with better error handling
# # ------------------------------------------------------------
# # In your wrappers.py - Debate Agent section
# class DebateAgent(BaseAgent):
#     def __init__(self, name="debate", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         if DebateLogic:
#             self.logic = DebateLogic(llm=llm)
#         else:
#             logger.error("❌ DebateLogic not available - creating dummy implementation")
#             self.logic = None

#     def plan(self, inp):
#         return {"action": "debate", **inp}

#     def act(self, plan):
#         try:
#             if self.logic:
#                 out = self.logic.debate(
#                     ticker=plan.get("ticker"),
#                     technical_result=plan.get("technical_result", {}),
#                     risk_metrics=plan.get("risk_metrics", {}),
#                     price_data=plan.get("df"),
#                     sentiment_score=plan.get("sentiment_score", 50)
#                 )
#             else:
#                 # Fallback implementation
#                 out = {
#                     "status": "FALLBACK",
#                     "consensus": "NEUTRAL",
#                     "message": "Debate logic not available",
#                     "arguments": []
#                 }
#         except Exception as e:
#             logger.exception("Debate logic failed")
#             return {"status": "ERROR", "agent": "debate", "error": str(e)}

#         return {
#             "ticker": plan.get("ticker"), 
#             "debate": out,
#             "status": "SUCCESS"
#         }


# # ------------------------------------------------------------
# # MASTER AGENT WRAPPER - FIXED with better error handling
# # ------------------------------------------------------------
# # In your wrappers.py - update the Master Agent section
# class MasterAgent(BaseAgent):
    
    
#     def __init__(self, tools=None, llm=None, name="master"):
#         # Provide default for tools if None
#         tools = tools or TOOLS if 'TOOLS' in globals() else {}
#         super().__init__(name=name, tools=tools, llm=llm)
        
#         print(f"🔧 MasterAgent init - llm provided: {llm is not None}")
#         print(f"🔧 MasterAgent init - MasterLogic available: {MasterLogic is not None}")
        
#         # Initialize MasterLogic if available
#         if MasterLogic:
#             try:
#                 print("🔧 Attempting to initialize MasterLogic with LLM...")
#                 self.logic = MasterLogic(llm=llm, min_confidence=60.0)
#                 print("✅ MasterLogic initialized successfully")
#             except Exception as e:
#                 logger.error(f"❌ Failed to initialize MasterLogic: {e}")
#                 print(f"❌ MasterLogic initialization failed: {e}")
#                 self.logic = None
#         else:
#             logger.error("❌ MasterLogic not available - creating dummy implementation")
#             print("❌ MasterLogic class not found/imported")
#             self.logic = None

#     def plan(self, inp):
#         return {"action": "synthesize", **inp}

#     def act(self, plan):
#         try:
#             ticker = plan.get("ticker")
#             if not ticker:
#                 return {
#                     "status": "ERROR", 
#                     "agent": "master", 
#                     "error": "Missing required 'ticker' in input"
#                 }

#             print(f"🔧 MasterAgent.act() - logic available: {self.logic is not None}")
#             print(f"🔧 MasterAgent.act() - llm available: {self.llm is not None}")

#             if self.logic:
#                 print("🚀 Using MasterLogic for synthesis...")
#                 out = self.logic.synthesize(
#                     ticker=ticker,
#                     technical_result=plan.get("technical_result", {}),
#                     sentiment_result=plan.get("sentiment_result", {}),
#                     risk_metrics=plan.get("risk_metrics", {}),
#                     portfolio_metrics=plan.get("portfolio_metrics", {}),
#                     current_price=plan.get("current_price", 0.0),
#                 )
#                 print(f"✅ MasterLogic synthesis completed: {out.get('status', 'UNKNOWN')}")
#             else:
#                 print("🔄 Falling back to fallback logic...")
#                 # Fallback implementation with basic analysis
#                 out = self._fallback_master_analysis(plan)

#             return {
#                 "ticker": ticker,
#                 "master": out,
#                 "status": "SUCCESS"
#             }

#         except Exception as e:
#             logger.exception("Master agent execution failed")
#             print(f"❌ Master agent execution failed: {e}")
#             return {
#                 "ticker": plan.get("ticker"), 
#                 "status": "ERROR", 
#                 "agent": "master", 
#                 "error": f"Master analysis failed: {str(e)}"
#             }

#     def _fallback_master_analysis(self, plan):
#         """Fallback analysis when MasterLogic is not available"""
#         ticker = plan.get("ticker")
#         current_price = plan.get("current_price", 0.0)
        
#         # Basic sentiment analysis from available data
#         technical_conf = plan.get("technical_result", {}).get("confidence", 50)
#         sentiment_conf = plan.get("sentiment_result", {}).get("confidence", 50)
#         risk_level = plan.get("risk_metrics", {}).get("risk_level", "MEDIUM")
        
#         # Simple decision logic
#         avg_confidence = (technical_conf + sentiment_conf) / 2
#         if avg_confidence > 60 and risk_level in ["LOW", "MEDIUM"]:
#             action = "BUY"
#         elif avg_confidence < 40:
#             action = "SELL"
#         else:
#             action = "HOLD"

#         return {
#             "status": "FALLBACK",
#             "action": action,
#             "confidence": avg_confidence,
#             "reasoning": f"Fallback analysis: Technical({technical_conf}%), Sentiment({sentiment_conf}%), Risk({risk_level})",
#             "current_price": current_price,
#             "risk_management": {
#                 "risk_level": risk_level,
#                 "suggested_stop_loss": current_price * 0.95 if current_price > 0 else 0,
#                 "suggested_take_profit": current_price * 1.08 if current_price > 0 else 0
#             },
#             "signals": {
#                 "technical": technical_conf / 100,
#                 "sentiment": sentiment_conf / 100,
#                 "composite": avg_confidence / 100
#             }
#         }
# # ------------------------------------------------------------
# # NEWS AGENT WRAPPER - FIXED with better error handling
# # ------------------------------------------------------------
# class NewsAgent(BaseAgent):
#     def __init__(self, name="news", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         if NewsLogic:
#             self.logic = NewsLogic()
#         else:
#             logger.error("❌ NewsLogic not available - creating dummy implementation")
#             self.logic = None

#     def plan(self, inp):
#         return {"action": "fetch", **inp}

#     def act(self, plan):
#         ticker = plan.get("ticker")
#         limit = plan.get("limit", 15)

#         try:
#             articles = self.call_tool("fetch_news", ticker, limit)
#         except Exception as e:
#             logger.warning(f"fetch_news failed: {e}")
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

#         return {
#             "ticker": ticker, 
#             "articles": articles, 
#             "summaries": summaries,
#             "status": "SUCCESS"
#         }


# # ------------------------------------------------------------
# # SENTIMENT AGENT WRAPPER - FIXED with better error handling
# # ------------------------------------------------------------
# class ProfessionalSentimentAgent(BaseAgent):
#     def __init__(self, name="sentiment", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         if SentimentLogic:
#             self.logic = SentimentLogic()
#         else:
#             logger.error("❌ SentimentLogic not available - creating dummy implementation")
#             self.logic = None

#     def plan(self, inp):
#         return {"action": "analyze", **inp}

#     def act(self, plan):
#         ticker = plan.get("ticker")
#         try:
#             if self.logic:
#                 # Call the sentiment analysis logic
#                 result = self.logic.analyze(ticker)
#             else:
#                 # Fallback implementation
#                 result = {
#                     "status": "FALLBACK",
#                     "sentiment": "NEUTRAL",
#                     "confidence": 0.5,
#                     "message": "Sentiment logic not available"
#                 }
#         except Exception as e:
#             logger.exception("Sentiment logic failed")
#             return {"status": "ERROR", "agent": "sentiment", "error": str(e)}

#         return {
#             "ticker": ticker,
#             "sentiment": result,
#             "status": "SUCCESS"
#         }


# # ------------------------------------------------------------
# # MAIN FACTORY — used by AgentRunner + UI
# # ------------------------------------------------------------
# def create_wrapped_agents(tools=None, llm=None) -> Dict[str, BaseAgent]:
#     """
#     Factory function that creates all wrapped agents.
#     This is the main entry point for AgentRunner.
#     """
#     tools = tools or TOOLS
#     logger.info("🏭 Creating wrapped agents...")
    
#     agents = {}
    
#     # Define all agent classes to create
#     agent_classes = {
#         "technical": TechnicalAgent,
#         "risk": RiskAgent,
#         "portfolio": PortfolioAgent,
#         "debate": DebateAgent,
#         "master": MasterAgent,
#         "news": NewsAgent,
#         "sentiment": ProfessionalSentimentAgent,
#     }
    
#     # Create each agent with error handling
#     for name, agent_class in agent_classes.items():
#         try:
#             agent = agent_class(tools=tools, llm=llm)
#             agents[name] = agent
#             logger.info(f"   ✅ Created agent: {name}")
#         except Exception as e:
#             logger.error(f"   ❌ Failed to create agent {name}: {e}")
#             # Don't add failed agents
    
#     logger.info(f"🎯 Successfully created {len(agents)} agents: {list(agents.keys())}")
#     return agents


# # ------------------------------------------------------------
# # ALTERNATIVE FACTORY for backward compatibility
# # ------------------------------------------------------------
# def create_agent_registry(tools=None, llm=None):
#     """Alternative factory function name for compatibility"""
#     return create_wrapped_agents(tools, llm)


# # ------------------------------------------------------------
# # DIRECT REGISTRATION FUNCTION
# # ------------------------------------------------------------
# def register_agents_directly(runner, tools=None, llm=None):
#     """
#     Directly register agents to an AgentRunner instance.
#     Useful for manual registration scenarios.
#     """
#     agents = create_wrapped_agents(tools, llm)
#     for name, agent in agents.items():
#         try:
#             runner.register(name, agent)
#             logger.info(f"📋 Directly registered: {name}")
#         except Exception as e:
#             logger.error(f"Failed to directly register {name}: {e}")
    
#     return list(agents.keys())



# """
# Wrapper layer converting pure-logic agents into BaseAgent-compatible agents.
# """

# import logging
# import os
# from typing import Dict, Any
# import json
# import datetime

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# # ------------------------------------------------------------
# # ROBUST IMPORTS with better error handling
# # ------------------------------------------------------------

# def safe_import(module_name, class_name, fallback_module=None):
#     """Safely import classes with comprehensive error handling"""
#     try:
#         module = __import__(module_name, fromlist=[class_name])
#         return getattr(module, class_name)
#     except ImportError as e:
#         logger.warning(f"Primary import failed for {class_name}: {e}")
#         if fallback_module:
#             try:
#                 module = __import__(fallback_module, fromlist=[class_name])
#                 return getattr(module, class_name)
#             except ImportError:
#                 logger.error(f"Fallback import also failed for {class_name}")
#         return None
#     except Exception as e:
#         logger.error(f"Unexpected error importing {class_name}: {e}")
#         return None

# # Import BaseAgent first as it's critical
# try:
#     from agents.base_agent import BaseAgent
# except ImportError:
#     try:
#         from trading_bot.agents.base_agent import BaseAgent
#     except ImportError:
#         logger.critical("❌ Cannot import BaseAgent - this will break everything!")
#         raise

# # Import all agent logic classes with fallbacks
# TechnicalLogic = safe_import('agents.technical_agent', 'TechnicalAnalysisAgent', 'trading_bot.agents.technical_agent')
# RiskLogic = safe_import('agents.risk_agent', 'RiskManagementAgent', 'trading_bot.agents.risk_agent')
# PortfolioLogic = safe_import('agents.portfolio_agent', 'PortfolioManagerAgent', 'trading_bot.agents.portfolio_agent')
# DebateLogic = safe_import('agents.debate_agent', 'DebateAgent', 'trading_bot.agents.debate_agent')
# MasterLogic = safe_import('agents.master_agent', 'MasterAgent', 'trading_bot.agents.master_agent')
# NewsLogic = safe_import('agents.news_agent', 'SentimentAnalysisAgent', 'trading_bot.agents.news_agent')
# SentimentLogic = safe_import('agents.sentiment_agent', 'SentimentAnalysisAgent', 'trading_bot.agents.sentiment_agent')

# try:
#     from tools.toolbox import TOOLS
# except ImportError:
#     try:
#         from trading_bot.tools.toolbox import TOOLS
#     except ImportError:
#         logger.critical("❌ Cannot import TOOLS - agent functionality will be limited")
#         TOOLS = {}

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
#         print("technical agent init called from wrappers")
#         if TechnicalLogic:
#             self.logic = TechnicalLogic()
#         else:
#             logger.error("❌ TechnicalLogic not available - creating dummy implementation")
#             self.logic = None

#     def plan(self, user_input):
#         return {"action": "run", **user_input}

#     def act(self, plan):
#         ticker = plan.get("ticker")
#         start = plan.get("start")
#         end = plan.get("end")

#         # Try fetching df for consistency
#         try:
#             df = self.call_tool("fetch_price", ticker, start, end)
#         except Exception as e:
#             logger.warning(f"fetch_price failed: {e}")
#             df = None

#         try:
#             if self.logic:
#                 out = self.logic.analyze(ticker, start_date=start, end_date=end)
#             else:
#                 # Fallback implementation
#                 out = {
#                     "status": "FALLBACK",
#                     "message": "Technical logic not available",
#                     "indicators": {},
#                     "signals": []
#                 }
#         except Exception as e:
#             logger.exception("Technical logic failed")
#             return {
#                 "status": "ERROR", 
#                 "agent": "technical", 
#                 "error": str(e),
#                 "ticker": ticker
#             }

#         return {
#             "ticker": ticker, 
#             "technical": out, 
#             "df": _extract_df(df),
#             "status": "SUCCESS"
#         }


# # ------------------------------------------------------------
# # RISK AGENT WRAPPER
# # ------------------------------------------------------------
# class RiskAgent(BaseAgent):
#     def __init__(self, name="risk", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         if RiskLogic:
#             self.logic = RiskLogic()
#         else:
#             logger.error("❌ RiskLogic not available - creating dummy implementation")
#             self.logic = None

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
#             if self.logic:
#                 out = self.logic.evaluate(
#                     ticker=ticker,
#                     df=dp,
#                     current_price=current_price,
#                     technical_confidence=plan.get("technical_confidence", 50),
#                     sentiment_confidence=plan.get("sentiment_confidence", 50)
#                 )
#             else:
#                 # Fallback implementation
#                 out = {
#                     "status": "FALLBACK",
#                     "risk_level": "MEDIUM",
#                     "message": "Risk logic not available"
#                 }
#         except Exception as e:
#             logger.exception("Risk logic failed")
#             return {"status": "ERROR", "agent": "risk", "error": str(e)}

#         return {
#             "ticker": ticker, 
#             "risk": out, 
#             "df": dp,
#             "status": "SUCCESS"
#         }


# # ------------------------------------------------------------
# # PORTFOLIO AGENT WRAPPER
# # ------------------------------------------------------------
# class PortfolioAgent(BaseAgent):
#     def __init__(self, name="portfolio", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         if PortfolioLogic:
#             self.logic = PortfolioLogic()
#         else:
#             logger.error("❌ PortfolioLogic not available - creating dummy implementation")
#             self.logic = None

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
#             if self.logic:
#                 action, quantity, meta = self.logic.decide(
#                     ticker=ticker,
#                     current_price=current_price,
#                     technical_signal=plan.get("technical_signal", {}),
#                     sentiment_signal=plan.get("sentiment_signal"),
#                     risk_metrics=plan.get("risk_metrics", {}),
#                     portfolio_state=plan.get("portfolio_state")
#                 )
#                 alloc = self.logic.get_allocation_metrics(action, quantity, current_price)
#             else:
#                 # Fallback implementation
#                 action = "HOLD"
#                 quantity = 0
#                 meta = {"message": "Portfolio logic not available"}
#                 alloc = {}
#         except Exception as e:
#             logger.exception("Portfolio logic failed")
#             return {"status": "ERROR", "agent": "portfolio", "error": str(e)}

#         return {
#             "ticker": ticker,
#             "action": action,
#             "quantity": int(quantity or 0),
#             "allocation": alloc,
#             "meta": meta if isinstance(meta, dict) else {"raw": str(meta)},
#             "df": df,
#             "status": "SUCCESS"
#         }


# # ------------------------------------------------------------
# # DEBATE AGENT WRAPPER
# # ------------------------------------------------------------
# class DebateAgent(BaseAgent):
#     def __init__(self, name="debate", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         if DebateLogic:
#             self.logic = DebateLogic(llm=llm)
#         else:
#             logger.error("❌ DebateLogic not available - creating dummy implementation")
#             self.logic = None

#     def plan(self, inp):
#         return {"action": "debate", **inp}

#     def act(self, plan):
#         try:
#             if self.logic:
#                 out = self.logic.debate(
#                     ticker=plan.get("ticker"),
#                     technical_result=plan.get("technical_result", {}),
#                     risk_metrics=plan.get("risk_metrics", {}),
#                     price_data=plan.get("df"),
#                     sentiment_score=plan.get("sentiment_score", 50)
#                 )
#             else:
#                 # Fallback implementation
#                 out = {
#                     "status": "FALLBACK",
#                     "consensus": "NEUTRAL",
#                     "message": "Debate logic not available",
#                     "arguments": []
#                 }
#         except Exception as e:
#             logger.exception("Debate logic failed")
#             return {"status": "ERROR", "agent": "debate", "error": str(e)}

#         return {
#             "ticker": plan.get("ticker"), 
#             "debate": out,
#             "status": "SUCCESS"
#         }


# # ------------------------------------------------------------
# # MASTER AGENT WRAPPER - UPDATED for direct Groq API usage
# # ------------------------------------------------------------
# class MasterAgent(BaseAgent):
    
#     def __init__(self, tools=None, llm=None, name="master"):
#         tools = tools or TOOLS if 'TOOLS' in globals() else {}
#         super().__init__(name=name, tools=tools, llm=llm)
        
#         print(f"🔧 MasterAgent wrapper init - MasterLogic available: {MasterLogic is not None}")
        
#         # Initialize MasterLogic with Groq API key from environment
#         if MasterLogic:
#             try:
#                 groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
#                 print(f"🔧 MasterAgent wrapper - Groq API key available: {bool(groq_api_key and groq_api_key != 'your-api-key-here')}")
                
#                 self.logic = MasterLogic(min_confidence=60.0, groq_api_key=groq_api_key)
#                 print("✅ MasterLogic initialized successfully in wrapper")
#             except Exception as e:
#                 logger.error(f"❌ Failed to initialize MasterLogic: {e}")
#                 print(f"❌ MasterLogic initialization failed: {e}")
#                 self.logic = None
#         else:
#             logger.error("❌ MasterLogic not available - creating dummy implementation")
#             print("❌ MasterLogic class not found/imported")
#             self.logic = None

#     def plan(self, inp):
#         return {"action": "synthesize", **inp}

#     def act(self, plan):
#          try:
#              ticker = plan.get("ticker")
#              if not ticker:
#                  return {
#                      "status": "ERROR", 
#                      "agent": "master", 
#                      "error": "Missing required 'ticker' in input"
#                  }

#              if self.logic:
#                  out = self.logic.synthesize(
#                      ticker=ticker,
#                      technical_result=plan.get("technical_result", {}),
#                      sentiment_result=plan.get("sentiment_result", {}),
#                      risk_metrics=plan.get("risk_metrics", {}),
#                      portfolio_metrics=plan.get("portfolio_metrics", {}),
#                      current_price=plan.get("current_price", 0.0),
#                  )
#              else:
#                  # Fallback implementation
#                  out = { ... }
     
#              return {
#                  "ticker": ticker,
#                  "master": out,  # This ensures the output is under "master" key
#                  "status": "SUCCESS"
#              }
#          except Exception as e:
#              logger.exception("Master agent execution failed")
#              return {
#                  "ticker": plan.get("ticker"), 
#                  "status": "ERROR", 
#                  "agent": "master", 
#                  "error": f"Master analysis failed: {str(e)}"
#              }



# # ------------------------------------------------------------
# # NEWS AGENT WRAPPER
# # ------------------------------------------------------------
# class NewsAgent(BaseAgent):
#     def __init__(self, name="news", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         if NewsLogic:
#             self.logic = NewsLogic()
#         else:
#             logger.error("❌ NewsLogic not available - creating dummy implementation")
#             self.logic = None

#     def plan(self, inp):
#         return {"action": "fetch", **inp}

#     def act(self, plan):
#         ticker = plan.get("ticker")
#         limit = plan.get("limit", 15)

#         try:
#             articles = self.call_tool("fetch_news", ticker, limit)
#         except Exception as e:
#             logger.warning(f"fetch_news failed: {e}")
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

#         return {
#             "ticker": ticker, 
#             "articles": articles, 
#             "summaries": summaries,
#             "status": "SUCCESS"
#         }


# # ------------------------------------------------------------
# # SENTIMENT AGENT WRAPPER
# # ------------------------------------------------------------
# class ProfessionalSentimentAgent(BaseAgent):
#     def __init__(self, name="sentiment", tools=None, llm=None):
#         super().__init__(name=name, tools=tools or TOOLS, llm=llm)
#         if SentimentLogic:
#             self.logic = SentimentLogic()
#         else:
#             logger.error("❌ SentimentLogic not available - creating dummy implementation")
#             self.logic = None

#     def plan(self, inp):
#         return {"action": "analyze", **inp}

#     def act(self, plan):
#         ticker = plan.get("ticker")
#         try:
#             if self.logic:
#                 # Call the sentiment analysis logic
#                 result = self.logic.analyze(ticker)
#             else:
#                 # Fallback implementation
#                 result = {
#                     "status": "FALLBACK",
#                     "sentiment": "NEUTRAL",
#                     "confidence": 0.5,
#                     "message": "Sentiment logic not available"
#                 }
#         except Exception as e:
#             logger.exception("Sentiment logic failed")
#             return {"status": "ERROR", "agent": "sentiment", "error": str(e)}

#         return {
#             "ticker": ticker,
#             "sentiment": result,
#             "status": "SUCCESS"
#         }


# # ------------------------------------------------------------
# # MAIN FACTORY
# # ------------------------------------------------------------
# def create_wrapped_agents(tools=None, llm=None) -> Dict[str, BaseAgent]:
#     """
#     Factory function that creates all wrapped agents.
#     """
#     tools = tools or TOOLS
#     logger.info("🏭 Creating wrapped agents...")
    
#     agents = {}
    
#     agent_classes = {
#         "technical": TechnicalAgent,
#         "risk": RiskAgent,
#         "portfolio": PortfolioAgent,
#         "debate": DebateAgent,
#         "master": MasterAgent,
#         "news": NewsAgent,
#         "sentiment": ProfessionalSentimentAgent,
#     }
    
#     for name, agent_class in agent_classes.items():
#         try:
#             agent = agent_class(tools=tools, llm=llm)
#             agents[name] = agent
#             logger.info(f"   ✅ Created agent: {name}")
#         except Exception as e:
#             logger.error(f"   ❌ Failed to create agent {name}: {e}")
    
#     logger.info(f"🎯 Successfully created {len(agents)} agents: {list(agents.keys())}")
#     return agents


# def create_agent_registry(tools=None, llm=None):
#     """Alternative factory function name for compatibility"""
#     return create_wrapped_agents(tools, llm)


# def register_agents_directly(runner, tools=None, llm=None):
#     """
#     Directly register agents to an AgentRunner instance.
#     """
#     agents = create_wrapped_agents(tools, llm)
#     for name, agent in agents.items():
#         try:
#             runner.register(name, agent)
#             logger.info(f"📋 Directly registered: {name}")
#         except Exception as e:
#             logger.error(f"Failed to directly register {name}: {e}")
    
#     return list(agents.keys())


"""
Wrapper layer converting pure-logic agents into BaseAgent-compatible agents.
"""

import logging
import os
from typing import Dict, Any
import json
import datetime

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
NewsLogic = safe_import('agents.news_agent', 'SentimentAnalysisAgent', 'trading_bot.agents.news_agent')
SentimentLogic = safe_import('agents.sentiment_agent', 'SentimentAnalysisAgent', 'trading_bot.agents.sentiment_agent')

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
# TECHNICAL AGENT WRAPPER
# ------------------------------------------------------------
class TechnicalAgent(BaseAgent):
    def __init__(self, name="technical", tools=None, llm=None):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm)
        print("technical agent init called from wrappers")
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
# RISK AGENT WRAPPER
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
# PORTFOLIO AGENT WRAPPER
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
# DEBATE AGENT WRAPPER
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
# MASTER AGENT WRAPPER - UPDATED for minimal input/output
# ------------------------------------------------------------
class MasterAgent(BaseAgent):
    
    def __init__(self, tools=None, llm=None, name="master"):
        tools = tools or TOOLS if 'TOOLS' in globals() else {}
        super().__init__(name=name, tools=tools, llm=llm)
        
        print(f"🔧 MasterAgent wrapper init - MasterLogic available: {MasterLogic is not None}")
        
        # Initialize MasterLogic with Groq API key
        if MasterLogic:
            try:
                # === YOUR API KEY HERE ===
                groq_api_key = "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"
                
                self.logic = MasterLogic(min_confidence=60.0, groq_api_key=groq_api_key)
                print("✅ MasterLogic initialized successfully in wrapper")
            except Exception as e:
                logger.error(f"❌ Failed to initialize MasterLogic: {e}")
                print(f"❌ MasterLogic initialization failed: {e}")
                self.logic = None
        else:
            logger.error("❌ MasterLogic not available - creating dummy implementation")
            print("❌ MasterLogic class not found/imported")
            self.logic = None

    def plan(self, inp):
        return {"action": "synthesize", **inp}

    def act(self, plan):
        try:
            ticker = plan.get("ticker")
            if not ticker:
                return {
                    "status": "ERROR", 
                    "agent": "master", 
                    "error": "Missing required 'ticker' in input"
                }

            print(f"🔧 MasterAgent.act() - logic available: {self.logic is not None}")

            if self.logic:
                print("🚀 Using MasterLogic for synthesis...")
                # Only pass the three essential inputs
                out = self.logic.synthesize(
                    ticker=ticker,
                    technical_result=plan.get("technical_result", {}),
                    risk_metrics=plan.get("risk_metrics", {}),
                    sentiment_result=plan.get("sentiment_result", {}),
                    current_price=plan.get("current_price", 0.0),
                )
                print(f"✅ MasterLogic synthesis completed")
                print(f"📊 Master output - Action: {out.get('action')}, Confidence: {out.get('confidence')}")
            else:
                print("🔄 Falling back to fallback logic...")
                # Fallback implementation with proper structure
                out = {
                    "ticker": ticker,
                    "current_price": plan.get("current_price", 0),
                    "action": "HOLD",
                    "confidence": 50,
                    "reasoning": "Master logic not available - using fallback",
                    "entry_price": plan.get("current_price", 0),
                    "stop_loss": None,
                    "take_profit": None,
                    "risk_reward_ratio": None,
                    "quantity": 0,
                    "position_size": 0.0,
                    "risk_level": "MEDIUM",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "ai_enhanced": False
                }

            # CRITICAL: Return the output directly (not nested under "master")
            # The UI expects master_payload to be the actual analysis result
            return out

        except Exception as e:
            logger.exception("Master agent execution failed")
            print(f"❌ Master agent execution failed: {e}")
            return {
                "status": "ERROR", 
                "agent": "master", 
                "error": f"Master analysis failed: {str(e)}"
            }


# ------------------------------------------------------------
# NEWS AGENT WRAPPER
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
# SENTIMENT AGENT WRAPPER
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
# MAIN FACTORY
# ------------------------------------------------------------
def create_wrapped_agents(tools=None, llm=None) -> Dict[str, BaseAgent]:
    """
    Factory function that creates all wrapped agents.
    """
    tools = tools or TOOLS
    logger.info("🏭 Creating wrapped agents...")
    
    agents = {}
    
    agent_classes = {
        "technical": TechnicalAgent,
        "risk": RiskAgent,
        "portfolio": PortfolioAgent,
        "debate": DebateAgent,
        "master": MasterAgent,
        "news": NewsAgent,
        "sentiment": ProfessionalSentimentAgent,
    }
    
    for name, agent_class in agent_classes.items():
        try:
            agent = agent_class(tools=tools, llm=llm)
            agents[name] = agent
            logger.info(f"   ✅ Created agent: {name}")
        except Exception as e:
            logger.error(f"   ❌ Failed to create agent {name}: {e}")
    
    logger.info(f"🎯 Successfully created {len(agents)} agents: {list(agents.keys())}")
    return agents


def create_agent_registry(tools=None, llm=None):
    """Alternative factory function name for compatibility"""
    return create_wrapped_agents(tools, llm)


def register_agents_directly(runner, tools=None, llm=None):
    """
    Directly register agents to an AgentRunner instance.
    """
    agents = create_wrapped_agents(tools, llm)
    for name, agent in agents.items():
        try:
            runner.register(name, agent)
            logger.info(f"📋 Directly registered: {name}")
        except Exception as e:
            logger.error(f"Failed to directly register {name}: {e}")
    
    return list(agents.keys())