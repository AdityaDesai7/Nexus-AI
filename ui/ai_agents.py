
# # trading_bot/ui/ai_agents.py
# import os
# from datetime import datetime, timedelta
# import logging
# import streamlit as st
# import pandas as pd

# from data.data_fetcher import fetch_data  # kept for potential future use

# # Runner + factories (robust imports)
# from agent_runner import AgentRunner
# try:
#     from agents.wrappers import create_wrapped_agents
# except Exception:
#     create_wrapped_agents = None

# try:
#     from agents.inst_wrappers import create_inst_wrappers
# except Exception:
#     create_inst_wrappers = None

# from tools.toolbox import TOOLS
# from llm.llm_wrapper import LLM

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


# # -------------------------
# # Helpers: normalize runner responses
# # -------------------------
# def normalize_response(resp):
#     """
#     Convert possible agent responses into a plain dict.
#     - dict -> returned as shallow copy
#     - pandas.DataFrame -> {'status':'OK','df': df}
#     - objects with .dict() -> use .dict()
#     - objects with __dict__ -> use vars()
#     - string/other -> {'status':'OK','text': str(resp)}
#     - None -> {'status':'ERROR', 'error': 'No response'}
#     """
#     try:
#         if resp is None:
#             return {"status": "ERROR", "error": "No response (None)"}

#         if isinstance(resp, dict):
#             return dict(resp)

#         if isinstance(resp, pd.DataFrame):
#             return {"status": "OK", "df": resp}

#         # pydantic-style
#         if hasattr(resp, "dict") and callable(getattr(resp, "dict")):
#             try:
#                 return {"status": "OK", **resp.dict()}
#             except Exception:
#                 pass

#         if hasattr(resp, "__dict__"):
#             try:
#                 return {"status": "OK", **vars(resp)}
#             except Exception:
#                 pass

#         return {"status": "OK", "text": str(resp)}
#     except Exception as e:
#         logger.exception("normalize_response failed")
#         return {"status": "ERROR", "error": f"normalize_response failed: {e}"}
    

# def extract_payload(norm: dict, preferred_keys=("result", "technical", "master", "risk", "portfolio")):
#     """
#     From a normalized dict, pick the nested payload under preferred_keys,
#     else return the normalized dict itself.
#     """
#     if not isinstance(norm, dict):
#         return {"status": "ERROR", "error": "normalize_response did not return dict"}

#     for k in preferred_keys:
#         if k in norm and isinstance(norm[k], dict):
#             return norm[k]
#     # sometimes responses are {'status':'OK', 'df': df}
#     if "df" in norm or "status" in norm:
#         return norm
#     return norm


# def display_json_friendly(label: str, payload, expand: bool = False):
#     """
#     Safely display payload in Streamlit:
#     - dict -> st.json (with dataframe shown separately)
#     - DataFrame -> st.dataframe
#     - list -> st.write
#     - string -> st.write
#     """
#     st.markdown(f"### {label}")
#     if payload is None:
#         st.info("No output")
#         return

#     if isinstance(payload, dict):
#         # show dataframe preview if present but don't mutate original dict
#         df = payload.get("df", None)
#         try:
#             st.json({k: v for k, v in payload.items() if k != "df"})
#         except Exception:
#             st.write(payload)
#         if df is not None:
#             if isinstance(df, pd.DataFrame):
#                 st.markdown("**Data (preview)**")
#                 #st.dataframe(df.head(10))
#             else:
#                 st.write(df)
#     elif isinstance(payload, pd.DataFrame):
#         st.dataframe(payload.head(10))
#     elif isinstance(payload, list):
#         st.write(payload)
#     else:
#         st.write(str(payload))


# def display_master_agent_analysis(master_payload: dict):
#     """
#     Clean display for minimal Master Agent output
#     """
#     if not isinstance(master_payload, dict):
#         st.error("No master analysis available")
#         return

#     st.markdown("---")
    
#     # Main Decision Card
#     col1, col2, col3 = st.columns(3)
    
#     action = master_payload.get("action", "HOLD")
#     confidence = master_payload.get("confidence", 50)
    
#     with col1:
#         if action == "BUY":
#             st.success(f"## 🟢 {action}")
#         elif action == "SELL":
#             st.error(f"## 🔴 {action}")  
#         else:
#             st.info(f"## ⚪ {action}")
            
#     with col2:
#         st.metric("Confidence", f"{confidence}%")
        
#     with col3:
#         risk_level = master_payload.get("risk_level", "MEDIUM")
#         if risk_level in ["LOW", "VERY_LOW"]:
#             st.success(f"Risk: {risk_level}")
#         elif risk_level in ["HIGH", "VERY_HIGH"]:
#             st.error(f"Risk: {risk_level}")
#         else:
#             st.warning(f"Risk: {risk_level}")
    
#     # Current Price
#     current_price = master_payload.get("current_price")
#     if current_price:
#         st.write(f"**Current Price:** ${current_price:.2f}")
    
#     # Reasoning
#     st.markdown("### 📋 AI Reasoning")
#     reasoning = master_payload.get("reasoning", "No reasoning provided")
#     st.info(reasoning)
    
#     # Price Targets (if available)
#     stop_loss = master_payload.get("stop_loss")
#     take_profit = master_payload.get("take_profit")
#     rr_ratio = master_payload.get("risk_reward_ratio")
    
#     if stop_loss or take_profit:
#         st.markdown("### 🎯 Price Targets")
#         price_col1, price_col2, price_col3 = st.columns(3)
        
#         with price_col1:
#             if stop_loss:
#                 st.metric("Stop Loss", f"${stop_loss:.2f}")
#             else:
#                 st.write("Stop Loss: N/A")
                
#         with price_col2:
#             if take_profit:
#                 st.metric("Take Profit", f"${take_profit:.2f}")
#             else:
#                 st.write("Take Profit: N/A")
                
#         with price_col3:
#             if rr_ratio:
#                 st.metric("R/R Ratio", f"{rr_ratio}:1")
#             else:
#                 st.write("R/R Ratio: N/A")
    
#     # Debug info (collapsed)
#     with st.expander("🔍 Raw Output"):
#         st.json(master_payload)

# # -------------------------
# # Runner initialization - UPDATED WITH FALLBACK
# # -------------------------
# def ensure_session_runner():
#     """
#     Initialize AgentRunner with guaranteed agent registration
#     """
#     if "agent_runner" not in st.session_state:
#         runner = AgentRunner()
        
#         # Force register core agents if empty (double safety)
#         if not runner.agents:
#             st.warning("🤖 No agents auto-registered, manually registering core agents...")
#             manual_register_core_agents(runner)
        
#         st.session_state.agent_runner = runner
#         st.session_state.wrapped_agents = list(runner.agents.keys())
#         logger.info(f"✅ AgentRunner initialized with agents: {st.session_state.wrapped_agents}")
    
#     return st.session_state.agent_runner


# def manual_register_core_agents(runner):
#     """Manually register core agents as backup"""
#     try:
#         from agents.wrappers import (
#             TechnicalAgent, RiskAgent, PortfolioAgent, 
#             DebateAgent, MasterAgent, NewsAgent,
#             ProfessionalSentimentAgent
#         )
        
#         core_agents = {
#             "technical": TechnicalAgent(tools=TOOLS, llm=LLM()),
#             "risk": RiskAgent(tools=TOOLS, llm=LLM()),
#             "portfolio": PortfolioAgent(tools=TOOLS, llm=LLM()),
#             "debate": DebateAgent(tools=TOOLS, llm=LLM()),
#             "master": MasterAgent(tools=TOOLS, llm=LLM()),
#             "news": NewsAgent(tools=TOOLS, llm=LLM()),
#             "sentiment": ProfessionalSentimentAgent(tools=TOOLS, llm=LLM()),
#         }
        
#         for name, agent in core_agents.items():
#             runner.register(name, agent)
            
#         logger.info(f"✅ Manually registered: {list(core_agents.keys())}")
        
#     except Exception as e:
#         logger.error(f"❌ Manual registration failed: {e}")
#         st.error(f"Failed to register agents: {e}")


# def safe_get_latest_close(payload):
#     """
#     Extract latest close price from a variety of payload shapes.
#     """
#     try:
#         if payload is None:
#             return 0.0

#         if isinstance(payload, dict):
#             for k in ("latest_close", "latest", "close", "price"):
#                 if k in payload:
#                     v = payload[k]
#                     if isinstance(v, (int, float)):
#                         return float(v)
#                     if isinstance(v, str) and v.replace('.', '', 1).isdigit():
#                         try:
#                             return float(v)
#                         except Exception:
#                             pass
#             if "df" in payload and isinstance(payload["df"], pd.DataFrame):
#                 df = payload["df"]
#                 if "Close" in df.columns and len(df) > 0:
#                     return float(df["Close"].iloc[-1])

#         if isinstance(payload, pd.DataFrame):
#             df = payload
#             if "Close" in df.columns and len(df) > 0:
#                 return float(df["Close"].iloc[-1])

#     except Exception:
#         pass
#     return 0.0


# def validate_agent_inputs(ticker: str, price_df: pd.DataFrame, latest_close: float) -> bool:
#     """Validate that we have fresh data for agent execution"""
#     if not ticker or ticker.strip() == "":
#         return False
#     if price_df is None or len(price_df) == 0:
#         return False
#     if latest_close <= 0:
#         return False
#     return True


# # -------------------------
# # Streamlit Page - FIXED PIPELINE
# # -------------------------
# def show_ai_agents_page():
#     st.set_page_config(layout="wide", page_title="SMART-MARKET — Agent Analysis")
#     st.title("🤖 SMART-MARKET — Agent Analysis")

#     runner = ensure_session_runner()

#     groq_key = os.getenv("GROQ_API_KEY", "").strip()
#     groq_ready = bool(groq_key)

#     with st.sidebar:
#         st.header("Run settings")
#         ticker = st.text_input("Ticker (e.g. RELIANCE.NS)", value=st.session_state.get("ticker", "RELIANCE.NS"))
#         start_date_input = st.date_input("Start date", value=datetime.now() - timedelta(days=180))
#         end_date_input = st.date_input("End date", value=datetime.now())
#         show_debate = st.checkbox("Show Debate", value=True)
#         run_btn = st.button("🚀 Analyze")
#         st.markdown("---")
#         st.markdown("Available Agents:")
#         st.write(", ".join(st.session_state.get("wrapped_agents", [])))
#         if not groq_ready:
#             st.warning("GROQ_API_KEY not configured. LLM features disabled.")
#         else:
#             st.success("Groq ready ✅")

#     st.session_state["ticker"] = ticker
#     start_date = datetime.combine(start_date_input, datetime.min.time())
#     end_date = datetime.combine(end_date_input, datetime.max.time())

#     if not run_btn:
#         st.info("Configure settings in the sidebar and click Analyze.")
#         return

#     # Run pipeline defensively
#     progress = st.progress(0)
#     status = st.empty()

#     try:
#         status.info("1/7 — Fetch canonical price data (via TOOLS)")
#         progress.progress(5)

#         price_df = None
#         latest_close = 0.0
#         try:
#             if "fetch_price" in TOOLS:
#                 price_res = TOOLS["fetch_price"](ticker, start_date, end_date)
#                 if isinstance(price_res, dict) and price_res.get("status") == "OK" and price_res.get("df") is not None:
#                     price_df = price_res.get("df")
#                 elif hasattr(price_res, "iloc"):
#                     price_df = price_res
#                 else:
#                     logger.warning("fetch_price returned unexpected shape: %s", type(price_res))
#             else:
#                 logger.warning("fetch_price not available in TOOLS")
#         except Exception as e:
#             logger.exception("TOOLS.fetch_price failed: %s", e)
#             status.error(f"Price fetch failed: {e}")

#         if isinstance(price_df, pd.DataFrame) and len(price_df) > 0:
#             latest_close = float(price_df["Close"].iloc[-1]) if "Close" in price_df.columns else 0.0

#         # Validate inputs before proceeding
#         if not validate_agent_inputs(ticker, price_df, latest_close):
#             st.error("❌ Invalid input data. Please check ticker symbol and date range.")
#             return

#         # TECHNICAL
#         status.info("2/7 — Running Technical Analysis")
#         progress.progress(15)
#         tech_raw = runner.run("technical", {"ticker": ticker, "start": start_date, "end": end_date})
#         tech_norm = normalize_response(tech_raw)
#         tech_payload = extract_payload(tech_norm, preferred_keys=("technical", "result"))
#         if isinstance(tech_payload, dict) and "df" not in tech_payload and isinstance(price_df, pd.DataFrame):
#             tech_payload["df"] = price_df

#         latest_close = latest_close or safe_get_latest_close(tech_payload)

#         # SENTIMENT - ADDED MISSING AGENT
#         status.info("3/7 — Running Sentiment Analysis")
#         progress.progress(30)
#         sentiment_raw = runner.run("sentiment", {"ticker": ticker})
#         sentiment_norm = normalize_response(sentiment_raw)
#         sentiment_payload = extract_payload(sentiment_norm)

#         # NEWS
#         status.info("4/7 — Running News Analysis")
#         progress.progress(45)
#         news_raw = runner.run("news", {"ticker": ticker, "limit": 10})
#         news_norm = normalize_response(news_raw)
#         news_payload = extract_payload(news_norm)

#         # RISK
#         status.info("5/7 — Running Risk Analysis")
#         progress.progress(60)
#         risk_input = {
#             "ticker": ticker,
#             "start": start_date,
#             "end": end_date,
#             "technical_confidence": tech_payload.get("confidence", 50) if isinstance(tech_payload, dict) else 50,
#             "sentiment_confidence": sentiment_payload.get("confidence", 50) if isinstance(sentiment_payload, dict) else 50,
#             "df": price_df,  # Use canonical price_df
#             "current_price": latest_close
#         }
#         risk_raw = runner.run("risk", risk_input)
#         risk_norm = normalize_response(risk_raw)
#         risk_payload = extract_payload(risk_norm)

#         # PORTFOLIO
#         status.info("6/7 — Running Portfolio Analysis")
#         progress.progress(75)
#         port_input = {
#             "ticker": ticker,
#             "current_price": latest_close,
#             "technical_signal": tech_payload if isinstance(tech_payload, dict) else {},
#             "sentiment_signal": sentiment_payload if isinstance(sentiment_payload, dict) else {},
#             "risk_metrics": risk_payload if isinstance(risk_payload, dict) else {},
#             "portfolio_state": {},  # Add empty portfolio state
#             "df": price_df
#         }
#         port_raw = runner.run("portfolio", port_input)
#         port_norm = normalize_response(port_raw)
#         port_payload = extract_payload(port_norm)

#         # DEBATE (optional)
#         debate_payload = None
#         if show_debate:
#             status.info("7/7 — Running Debate Agent")
#             progress.progress(85)
#             debate_raw = runner.run("debate", {
#                 "ticker": ticker,
#                 "technical_result": tech_payload,
#                 "risk_metrics": risk_payload,
#                 "sentiment_score": sentiment_payload.get("confidence", 50) if isinstance(sentiment_payload, dict) else 50,
#                 "price_data": price_df
#             })
#             debate_norm = normalize_response(debate_raw)
#             debate_payload = extract_payload(debate_norm)

#         progress.progress(90)

#         # MASTER - WITH CORRECT INPUTS
#         status.info("Finalizing — Running Master Agent")
#         master_input = {
#             "ticker": ticker,
#             "technical_result": tech_payload,
#             "sentiment_result": sentiment_payload,  # Use dedicated sentiment analysis
#             "risk_metrics": risk_payload,
#             "portfolio_metrics": port_payload,
#             "current_price": latest_close
#         }

#         master_raw = runner.run("master", master_input)
#         master_norm = normalize_response(master_raw)
#         master_payload = master_norm.get("master") if isinstance(master_norm, dict) and "master" in master_norm else master_norm

#         progress.progress(100)
#         status.empty()
#         progress.empty()
#         st.success("✅ Analysis Complete")

#         # ---------- DISPLAY ----------
#         tab_all, tab_master, tab_sentiment, tab_debate, tab_news = st.tabs(["All INFO", "Master Agent", "Sentiment Agent", "Debate Agent", "News "])

#         with tab_all:
#             col1, col2 = st.columns(2)
#             with col1:
#                 display_json_friendly("Technical Analysis", tech_payload)
#                 st.markdown("---")
#                 display_json_friendly("Risk Analysis", risk_payload)
#             with col2:
#                 display_json_friendly("Portfolio Analysis", port_payload)
#                 st.markdown("---")
#                 display_json_friendly("Sentiment Analysis", sentiment_payload)

#         with tab_master:
#     # Debug: Show what we're actually getting
#              st.write("🔍 Debug - Master Payload Structure:")
#              st.json(master_payload)  # This will show you the actual structure
    
#              if isinstance(master_payload, dict):
#                  if master_payload.get("status") != "ERROR":
#                      try:
#                          display_master_agent_analysis(master_payload)
#                      except Exception as e:
#                          st.error(f"Error displaying master analysis: {e}")
#                          st.json(master_payload)  # Show raw data for debugging
#                  else:
#                      st.error("Master analysis returned error status")
#                      st.json(master_payload)
#              else:
#                  st.error(f"Master analysis returned unexpected type: {type(master_payload)}")
#                  st.write("Raw master payload:", master_payload)

#         with tab_sentiment:
#             display_json_friendly("Sentiment Analysis", sentiment_payload)

#         with tab_debate:
#             if debate_payload:
#                 display_json_friendly("Debate Output", debate_payload)
#             else:
#                 st.info("Debate agent was not run")

#         with tab_news:
#             if isinstance(news_payload, dict):
#                 summaries = news_payload.get("summaries") or news_payload.get("articles") or []
#                 if summaries:
#                     for s in summaries[:10]:
#                         if isinstance(s, dict):
#                             title = s.get("title", s.get("headline", "No title"))
#                             src = s.get("source", "unknown")
#                             st.write(f"- {title} — {src}")
#                         else:
#                             st.write(f"- {s}")
#                 else:
#                     st.info("No news articles found")
#             else:
#                 st.write(news_payload)

#         # Quick execution panel (simulated)
#         st.markdown("---")
#         if isinstance(master_payload, dict) and (master_payload.get("action") or master_payload.get("recommendation")):
#             master_action = master_payload.get("action", master_payload.get("recommendation", "HOLD"))
#             if master_action != "HOLD":
#                 qty_default = int(port_payload.get("suggested_quantity", port_payload.get("quantity", 1)) if isinstance(port_payload, dict) else 1)
#                 qty = st.number_input("Quantity", min_value=1, value=qty_default)
#                 if st.button(f"Simulate {master_action}"):
#                     st.success(f"Simulated {master_action} {qty} @ {latest_close:.2f}")
#             else:
#                 st.info("Master recommends HOLD")

#     except Exception as e:
#         logger.exception("Pipeline failure")
#         st.error(f"Pipeline error: {e}")
#         with st.expander("Trace"):
#             import traceback
#             st.code(traceback.format_exc())


# # -------------------------
# # Agent Runner - FIXED REGISTRATION
# # -------------------------
# import logging
# from typing import Dict, Any
# import time

# # FIXED IMPORTS (correct for your folder structure)
# from agents.base_agent import BaseAgent
# from tools.toolbox import TOOLS
# from llm.llm_wrapper import LLM   # your Groq/Dummy unified wrapper

# # Try to import institutional wrapper factory if present
# try:
#     from agents.inst_wrappers import create_inst_wrappers
# except Exception:
#     create_inst_wrappers = None

# # Try to import regular wrapper factory if present
# try:
#     from agents.wrappers import create_wrapped_agents
# except Exception:
#     create_wrapped_agents = None

# logger = logging.getLogger(__name__)


# class AgentRunner:
#     """
#     Central manager for all agents.
#     - Registers agents by name
#     - Injects tools + LLM into every agent
#     - Runs agents safely
#     - Supports both: regular 5 agents + 9 institutional agents
#     """

#     def __init__(self):
#         self.agents: Dict[str, BaseAgent] = {}

#         # shared tools + shared LLM wrapper (Groq or Dummy)
#         self.tools = TOOLS
#         self.llm = LLM()

#         logger.info("🤖 AgentRunner initialized with unified tools + LLM")

#         # Use single registration method
#         self.registerAllAgents()
        
#         logger.info(f"🎯 Final registered agents: {list(self.agents.keys())}")

#     def registerAllAgents(self):
#         """Register all core agents - single source of truth"""
#         try:
#             from agents.wrappers import (
#                 TechnicalAgent, RiskAgent, PortfolioAgent, 
#                 DebateAgent, MasterAgent, NewsAgent,
#                 ProfessionalSentimentAgent
#             )
            
#             core_agents = {
#                 "technical": TechnicalAgent(tools=self.tools, llm=self.llm),
#                 "risk": RiskAgent(tools=self.tools, llm=self.llm),
#                 "portfolio": PortfolioAgent(tools=self.tools, llm=self.llm),
#                 "debate": DebateAgent(tools=self.tools, llm=self.llm),
#                 "master": MasterAgent(tools=self.tools, llm=self.llm),
#                 "news": NewsAgent(tools=self.tools, llm=self.llm),
#                 "sentiment": ProfessionalSentimentAgent(tools=self.tools, llm=self.llm),
#             }
            
#             for name, agent in core_agents.items():
#                 self.register(name, agent)
                
#             logger.info(f"✅ Successfully registered: {list(core_agents.keys())}")
            
#         except Exception as e:
#             logger.error(f"❌ Failed to register agents: {e}")
#             # Don't raise, allow graceful degradation

#     # --------------------------------------------------------------
#     # REGISTER ANY AGENT
#     # --------------------------------------------------------------
#     def register(self, name: str, agent: BaseAgent):
#         if not isinstance(name, str):
#             raise ValueError("Agent name must be a string")

#         if not isinstance(agent, BaseAgent):
#             raise TypeError(f"Agent '{name}' must inherit BaseAgent")

#         self.agents[name] = agent
#         logger.info(f"📋 Registered agent: {name}")

#     # --------------------------------------------------------------
#     # RUN ONE AGENT
#     # --------------------------------------------------------------
#     def run(self, name: str, user_input: Dict[str, Any]):
#         """
#         Run one agent and return its output.
#         Inject tools + LLM automatically.
#         """
#         agent = self.agents.get(name)
#         if not agent:
            
#             logger.error(f"Unknown agent: {name}. Available: {list(self.agents.keys())}")
#             raise RuntimeError(f"Unknown agent: {name}")
            
#         logger.info(f"🚀 Running agent: {name}")

#         # Inject dependencies if missing
#         agent.tools = getattr(agent, "tools", None) or self.tools
#         agent.llm = getattr(agent, "llm", None) or self.llm

#         # Execute safely with timing
#         start_time = time.time()
#         try:
#             result = agent.run(user_input)
#             elapsed = time.time() - start_time
#             logger.info(f"✅ Agent '{name}' completed in {elapsed:.2f}s")
#             return result

#         except Exception as e:
#             elapsed = time.time() - start_time
#             logger.error(f"❌ Agent '{name}' failed after {elapsed:.2f}s: {e}", exc_info=True)
#             return {
#                 "status": "ERROR",
#                 "agent": name,
#                 "error": str(e)
#             }

#     # --------------------------------------------------------------
#     # RUN ALL AGENTS (optional but recommended)
#     # --------------------------------------------------------------
#     def run_all(self, user_input: Dict[str, Any]):
#         """
#         Runs EVERY registered agent.
#         Returns a dict: { agent_name: result }
#         """
#         outputs = {}
#         logger.info(f"Running all agents: {list(self.agents.keys())}")

#         for name, agent in self.agents.items():
#             try:
#                 outputs[name] = self.run(name, user_input)
#             except Exception as e:
#                 logger.error(f"Agent '{name}' crashed inside run_all: {e}")
#                 outputs[name] = {
#                     "status": "ERROR",
#                     "agent": name,
#                     "error": str(e)
#                 }
    
#         return outputs


# # trading_bot/ui/ai_agents.py
# import os
# from datetime import datetime, timedelta
# import logging
# import streamlit as st
# import pandas as pd

# from data.data_fetcher import fetch_data  # kept for potential future use

# # Runner + factories (robust imports)
# from agent_runner import AgentRunner
# try:
#     from agents.wrappers import create_wrapped_agents
# except Exception:
#     create_wrapped_agents = None

# try:
#     from agents.inst_wrappers import create_inst_wrappers
# except Exception:
#     create_inst_wrappers = None

# from tools.toolbox import TOOLS
# from llm.llm_wrapper import LLM

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


# def extract_master_output(master_response):
#     """
#     Specialized extraction for Master Agent that only returns the clean analysis output
#     Handles the nested structure: {"result": {"master": {actual_output}}}
#     """
#     try:
#         if master_response is None:
#             return {"status": "ERROR", "error": "No master response"}
            
#         # If it's already the clean output we want, return it directly
#         if isinstance(master_response, dict) and "action" in master_response and "confidence" in master_response:
#             return master_response
            
#         # Handle the nested structure from agent execution
#         if isinstance(master_response, dict):
#             # Look for result -> master nesting
#             if "result" in master_response and isinstance(master_response["result"], dict):
#                 result = master_response["result"]
#                 if "master" in result and isinstance(result["master"], dict):
#                     return result["master"]
            
#             # Look for direct master key
#             if "master" in master_response and isinstance(master_response["master"], dict):
#                 return master_response["master"]
                
#             # If it's an error response, return as is
#             if "status" in master_response and master_response.get("status") == "ERROR":
#                 return master_response
                
#         # Fallback: return whatever we got
#         return master_response
        
#     except Exception as e:
#         logger.error(f"Error extracting master output: {e}")
#         return {"status": "ERROR", "error": f"Extraction failed: {e}"}

# # -------------------------
# # Helpers: normalize runner responses
# # -------------------------
# def normalize_response(resp):
#     """
#     Convert possible agent responses into a plain dict.
#     - dict -> returned as shallow copy
#     - pandas.DataFrame -> {'status':'OK','df': df}
#     - objects with .dict() -> use .dict()
#     - objects with __dict__ -> use vars()
#     - string/other -> {'status':'OK','text': str(resp)}
#     - None -> {'status':'ERROR', 'error': 'No response'}
#     """
#     try:
#         if resp is None:
#             return {"status": "ERROR", "error": "No response (None)"}

#         if isinstance(resp, dict):
#             return dict(resp)

#         if isinstance(resp, pd.DataFrame):
#             return {"status": "OK", "df": resp}

#         # pydantic-style
#         if hasattr(resp, "dict") and callable(getattr(resp, "dict")):
#             try:
#                 return {"status": "OK", **resp.dict()}
#             except Exception:
#                 pass

#         if hasattr(resp, "__dict__"):
#             try:
#                 return {"status": "OK", **vars(resp)}
#             except Exception:
#                 pass

#         return {"status": "OK", "text": str(resp)}
#     except Exception as e:
#         logger.exception("normalize_response failed")
#         return {"status": "ERROR", "error": f"normalize_response failed: {e}"}
    

# def extract_payload(norm: dict, preferred_keys=("result", "technical", "master", "risk", "portfolio")):
#     """
#     From a normalized dict, pick the nested payload under preferred_keys,
#     else return the normalized dict itself.
#     """
#     if not isinstance(norm, dict):
#         return {"status": "ERROR", "error": "normalize_response did not return dict"}

#     for k in preferred_keys:
#         if k in norm and isinstance(norm[k], dict):
#             return norm[k]
#     # sometimes responses are {'status':'OK', 'df': df}
#     if "df" in norm or "status" in norm:
#         return norm
#     return norm


# def display_json_friendly(label: str, payload, expand: bool = False):
#     """
#     Safely display payload in Streamlit:
#     - dict -> st.json (with dataframe shown separately)
#     - DataFrame -> st.dataframe
#     - list -> st.write
#     - string -> st.write
#     """
#     st.markdown(f"### {label}")
#     if payload is None:
#         st.info("No output")
#         return

#     if isinstance(payload, dict):
#         # show dataframe preview if present but don't mutate original dict
#         df = payload.get("df", None)
#         try:
#             st.json({k: v for k, v in payload.items() if k != "df"})
#         except Exception:
#             st.write(payload)
#         if df is not None:
#             if isinstance(df, pd.DataFrame):
#                 st.markdown("**Data (preview)**")
#                 #st.dataframe(df.head(10))
#             else:
#                 st.write(df)
#     elif isinstance(payload, pd.DataFrame):
#         st.dataframe(payload.head(10))
#     elif isinstance(payload, list):
#         st.write(payload)
#     else:
#         st.write(str(payload))


# def display_master_agent_analysis(master_payload: dict):
#     """
#     Clean display for minimal Master Agent output
#     """
#     if not isinstance(master_payload, dict):
#         st.error("No master analysis available")
#         return

#     st.markdown("---")
    
#     # Main Decision Card
#     col1, col2, col3 = st.columns(3)
    
#     action = master_payload.get("action", "HOLD")
#     confidence = master_payload.get("confidence", 50)
    
#     with col1:
#         if action == "BUY":
#             st.success(f"## 🟢 {action}")
#         elif action == "SELL":
#             st.error(f"## 🔴 {action}")  
#         else:
#             st.info(f"## ⚪ {action}")
            
#     with col2:
#         st.metric("Confidence", f"{confidence}%")
        
#     with col3:
#         risk_level = master_payload.get("risk_level", "MEDIUM")
#         if risk_level in ["LOW", "VERY_LOW"]:
#             st.success(f"Risk: {risk_level}")
#         elif risk_level in ["HIGH", "VERY_HIGH"]:
#             st.error(f"Risk: {risk_level}")
#         else:
#             st.warning(f"Risk: {risk_level}")
    
#     # Current Price
#     current_price = master_payload.get("current_price")
#     if current_price:
#         st.write(f"**Current Price:** ${current_price:.2f}")
    
#     # Reasoning
#     st.markdown("### 📋 AI Reasoning")
#     reasoning = master_payload.get("reasoning", "No reasoning provided")
#     st.info(reasoning)
    
#     # Price Targets (if available)
#     stop_loss = master_payload.get("stop_loss")
#     take_profit = master_payload.get("take_profit")
#     rr_ratio = master_payload.get("risk_reward_ratio")
    
#     if stop_loss or take_profit:
#         st.markdown("### 🎯 Price Targets")
#         price_col1, price_col2, price_col3 = st.columns(3)
        
#         with price_col1:
#             if stop_loss:
#                 st.metric("Stop Loss", f"${stop_loss:.2f}")
#             else:
#                 st.write("Stop Loss: N/A")
                
#         with price_col2:
#             if take_profit:
#                 st.metric("Take Profit", f"${take_profit:.2f}")
#             else:
#                 st.write("Take Profit: N/A")
                
#         with price_col3:
#             if rr_ratio:
#                 st.metric("R/R Ratio", f"{rr_ratio}:1")
#             else:
#                 st.write("R/R Ratio: N/A")
    
#     # Debug info (collapsed)
#     with st.expander("🔍 Raw Output"):
#         st.json(master_payload)


# # -------------------------
# # Runner initialization - UPDATED WITH FALLBACK
# # -------------------------
# def ensure_session_runner():
#     """
#     Initialize AgentRunner with guaranteed agent registration
#     """
#     if "agent_runner" not in st.session_state:
#         runner = AgentRunner()
        
#         # Force register core agents if empty (double safety)
#         if not runner.agents:
#             st.warning("🤖 No agents auto-registered, manually registering core agents...")
#             manual_register_core_agents(runner)
        
#         st.session_state.agent_runner = runner
#         st.session_state.wrapped_agents = list(runner.agents.keys())
#         logger.info(f"✅ AgentRunner initialized with agents: {st.session_state.wrapped_agents}")
    
#     return st.session_state.agent_runner


# def manual_register_core_agents(runner):
#     """Manually register core agents as backup"""
#     try:
#         from agents.wrappers import (
#             TechnicalAgent, RiskAgent, PortfolioAgent, 
#             DebateAgent, MasterAgent, NewsAgent,
#             ProfessionalSentimentAgent
#         )
        
#         core_agents = {
#             "technical": TechnicalAgent(tools=TOOLS, llm=LLM()),
#             "risk": RiskAgent(tools=TOOLS, llm=LLM()),
#             "portfolio": PortfolioAgent(tools=TOOLS, llm=LLM()),
#             "debate": DebateAgent(tools=TOOLS, llm=LLM()),
#             "master": MasterAgent(tools=TOOLS, llm=LLM()),
#             "news": NewsAgent(tools=TOOLS, llm=LLM()),
#             "sentiment": ProfessionalSentimentAgent(tools=TOOLS, llm=LLM()),
#         }
        
#         for name, agent in core_agents.items():
#             runner.register(name, agent)
            
#         logger.info(f"✅ Manually registered: {list(core_agents.keys())}")
        
#     except Exception as e:
#         logger.error(f"❌ Manual registration failed: {e}")
#         st.error(f"Failed to register agents: {e}")


# def safe_get_latest_close(payload):
#     """
#     Extract latest close price from a variety of payload shapes.
#     """
#     try:
#         if payload is None:
#             return 0.0

#         if isinstance(payload, dict):
#             for k in ("latest_close", "latest", "close", "price"):
#                 if k in payload:
#                     v = payload[k]
#                     if isinstance(v, (int, float)):
#                         return float(v)
#                     if isinstance(v, str) and v.replace('.', '', 1).isdigit():
#                         try:
#                             return float(v)
#                         except Exception:
#                             pass
#             if "df" in payload and isinstance(payload["df"], pd.DataFrame):
#                 df = payload["df"]
#                 if "Close" in df.columns and len(df) > 0:
#                     return float(df["Close"].iloc[-1])

#         if isinstance(payload, pd.DataFrame):
#             df = payload
#             if "Close" in df.columns and len(df) > 0:
#                 return float(df["Close"].iloc[-1])

#     except Exception:
#         pass
#     return 0.0


# def validate_agent_inputs(ticker: str, price_df: pd.DataFrame, latest_close: float) -> bool:
#     """Validate that we have fresh data for agent execution"""
#     if not ticker or ticker.strip() == "":
#         return False
#     if price_df is None or len(price_df) == 0:
#         return False
#     if latest_close <= 0:
#         return False
#     return True


# # -------------------------
# # Streamlit Page - FIXED PIPELINE
# # -------------------------
# def show_ai_agents_page():
#     st.set_page_config(layout="wide", page_title="SMART-MARKET — Agent Analysis")
#     st.title("🤖 SMART-MARKET — Agent Analysis")

#     runner = ensure_session_runner()

#     groq_key = os.getenv("GROQ_API_KEY", "").strip()
#     groq_ready = bool(groq_key)

#     with st.sidebar:
#         st.header("Run settings")
#         ticker = st.text_input("Ticker (e.g. RELIANCE.NS)", value=st.session_state.get("ticker", "RELIANCE.NS"))
#         start_date_input = st.date_input("Start date", value=datetime.now() - timedelta(days=180))
#         end_date_input = st.date_input("End date", value=datetime.now())
#         show_debate = st.checkbox("Show Debate", value=True)
#         run_btn = st.button("🚀 Analyze")
#         st.markdown("---")
#         st.markdown("Available Agents:")
#         st.write(", ".join(st.session_state.get("wrapped_agents", [])))
#         if not groq_ready:
#             st.warning("GROQ_API_KEY not configured. LLM features disabled.")
#         else:
#             st.success("Groq ready ✅")

#     st.session_state["ticker"] = ticker
#     start_date = datetime.combine(start_date_input, datetime.min.time())
#     end_date = datetime.combine(end_date_input, datetime.max.time())

#     if not run_btn:
#         st.info("Configure settings in the sidebar and click Analyze.")
#         return

#     # Run pipeline defensively
#     progress = st.progress(0)
#     status = st.empty()

#     try:
#         status.info("1/7 — Fetch canonical price data (via TOOLS)")
#         progress.progress(5)

#         price_df = None
#         latest_close = 0.0
#         try:
#             if "fetch_price" in TOOLS:
#                 price_res = TOOLS["fetch_price"](ticker, start_date, end_date)
#                 if isinstance(price_res, dict) and price_res.get("status") == "OK" and price_res.get("df") is not None:
#                     price_df = price_res.get("df")
#                 elif hasattr(price_res, "iloc"):
#                     price_df = price_res
#                 else:
#                     logger.warning("fetch_price returned unexpected shape: %s", type(price_res))
#             else:
#                 logger.warning("fetch_price not available in TOOLS")
#         except Exception as e:
#             logger.exception("TOOLS.fetch_price failed: %s", e)
#             status.error(f"Price fetch failed: {e}")

#         if isinstance(price_df, pd.DataFrame) and len(price_df) > 0:
#             latest_close = float(price_df["Close"].iloc[-1]) if "Close" in price_df.columns else 0.0

#         # Validate inputs before proceeding
#         if not validate_agent_inputs(ticker, price_df, latest_close):
#             st.error("❌ Invalid input data. Please check ticker symbol and date range.")
#             return

#         # TECHNICAL
#         status.info("2/7 — Running Technical Analysis")
#         progress.progress(15)
#         tech_raw = runner.run("technical", {"ticker": ticker, "start": start_date, "end": end_date})
#         tech_norm = normalize_response(tech_raw)
#         tech_payload = extract_payload(tech_norm, preferred_keys=("technical", "result"))
#         if isinstance(tech_payload, dict) and "df" not in tech_payload and isinstance(price_df, pd.DataFrame):
#             tech_payload["df"] = price_df

#         latest_close = latest_close or safe_get_latest_close(tech_payload)

#         # SENTIMENT - ADDED MISSING AGENT
#         status.info("3/7 — Running Sentiment Analysis")
#         progress.progress(30)
#         sentiment_raw = runner.run("sentiment", {"ticker": ticker})
#         sentiment_norm = normalize_response(sentiment_raw)
#         sentiment_payload = extract_payload(sentiment_norm)

#         # NEWS
#         status.info("4/7 — Running News Analysis")
#         progress.progress(45)
#         news_raw = runner.run("news", {"ticker": ticker, "limit": 10})
#         news_norm = normalize_response(news_raw)
#         news_payload = extract_payload(news_norm)

#         # RISK
#         status.info("5/7 — Running Risk Analysis")
#         progress.progress(60)
#         risk_input = {
#             "ticker": ticker,
#             "start": start_date,
#             "end": end_date,
#             "technical_confidence": tech_payload.get("confidence", 50) if isinstance(tech_payload, dict) else 50,
#             "sentiment_confidence": sentiment_payload.get("confidence", 50) if isinstance(sentiment_payload, dict) else 50,
#             "df": price_df,  # Use canonical price_df
#             "current_price": latest_close
#         }
#         risk_raw = runner.run("risk", risk_input)
#         risk_norm = normalize_response(risk_raw)
#         risk_payload = extract_payload(risk_norm)

#         # PORTFOLIO
#         status.info("6/7 — Running Portfolio Analysis")
#         progress.progress(75)
#         port_input = {
#             "ticker": ticker,
#             "current_price": latest_close,
#             "technical_signal": tech_payload if isinstance(tech_payload, dict) else {},
#             "sentiment_signal": sentiment_payload if isinstance(sentiment_payload, dict) else {},
#             "risk_metrics": risk_payload if isinstance(risk_payload, dict) else {},
#             "portfolio_state": {},  # Add empty portfolio state
#             "df": price_df
#         }
#         port_raw = runner.run("portfolio", port_input)
#         port_norm = normalize_response(port_raw)
#         port_payload = extract_payload(port_norm)

#         # DEBATE (optional)
#         debate_payload = None
#         if show_debate:
#             status.info("7/7 — Running Debate Agent")
#             progress.progress(85)
#             debate_raw = runner.run("debate", {
#                 "ticker": ticker,
#                 "technical_result": tech_payload,
#                 "risk_metrics": risk_payload,
#                 "sentiment_score": sentiment_payload.get("confidence", 50) if isinstance(sentiment_payload, dict) else 50,
#                 "price_data": price_df
#             })
#             debate_norm = normalize_response(debate_raw)
#             debate_payload = extract_payload(debate_norm)

#         progress.progress(90)

#         # MASTER - WITH CORRECT INPUTS AND OUTPUT EXTRACTION
#         status.info("Finalizing — Running Master Agent")
#         master_input = {
#             "ticker": ticker,
#             "technical_result": tech_payload,
#             "risk_metrics": risk_payload,
#             "sentiment_result": sentiment_payload,
#             "current_price": latest_close
#         }

#         master_raw = runner.run("master", master_input)
#         master_norm = normalize_response(master_raw)
        
#         # DEBUG: Show what we're getting
#         st.sidebar.write("🔍 Master Raw:", master_raw)
#         st.sidebar.write("🔍 Master Norm:", master_norm)
        
#         # SIMPLIFIED EXTRACTION - Master agent now returns the analysis directly
#         master_payload = master_norm

#         progress.progress(100)
#         status.empty()
#         progress.empty()
        
#         if master_payload and isinstance(master_payload, dict) and master_payload.get("status") != "ERROR":
#             st.success("✅ Analysis Complete")
#         else:
#             st.error("❌ Master analysis failed")

#         master_raw = runner.run("master", master_input)
#         master_norm = normalize_response(master_raw)
        
#         # DEBUG: Show what we're getting
#         st.sidebar.write("🔍 Master Raw:", master_raw)
#         st.sidebar.write("🔍 Master Norm:", master_norm)
        
#         # CORRECT EXTRACTION - Master agent returns {"master": {output}}
#         if isinstance(master_norm, dict) and "master" in master_norm:
#             master_payload = master_norm["master"]
#         else:
#             master_payload = master_norm

#         progress.progress(100)
#         status.empty()
#         progress.empty()
        
#         if master_payload and isinstance(master_payload, dict) and master_payload.get("status") != "ERROR":
#             st.success("✅ Analysis Complete")
#         else:
#             st.error("❌ Master analysis failed")

#         # ---------- DISPLAY ----------
#         tab_all, tab_master, tab_sentiment, tab_debate, tab_news = st.tabs(["All INFO", "Master Agent", "Sentiment Agent", "Debate Agent", "News "])

#         with tab_all:
#             col1, col2 = st.columns(2)
#             with col1:
#                 display_json_friendly("Technical Analysis", tech_payload)
#                 st.markdown("---")
#                 display_json_friendly("Risk Analysis", risk_payload)
#             with col2:
#                 display_json_friendly("Portfolio Analysis", port_payload)
#                 st.markdown("---")
#                 display_json_friendly("Sentiment Analysis", sentiment_payload)

        
#         with tab_master:
#             st.markdown("### 🧠 Master Agent Analysis")
            
#             # Show debug info in sidebar
#             with st.sidebar.expander("🔍 Master Debug Info"):
#                 st.write("Master Raw:", master_raw)
#                 st.write("Master Payload:", master_payload)
            
#             if master_payload is None:
#                 st.error("❌ Master payload is None")
#             elif not isinstance(master_payload, dict):
#                 st.error(f"❌ Master payload is not a dict: {type(master_payload)}")
#                 st.write("Raw output:", master_payload)
#             elif master_payload.get("status") == "ERROR":
#                 st.error("❌ Master analysis returned error")
#                 st.json(master_payload)
#             elif "action" not in master_payload:
#                 st.warning("⚠️ Master output missing 'action' key - showing raw output")
#                 st.json(master_payload)
#             else:
#                 # SUCCESS! We have the clean Master Agent output
#                 display_master_agent_analysis(master_payload)
            

#         with tab_sentiment:
#             display_json_friendly("Sentiment Analysis", sentiment_payload)

#         with tab_debate:
#             if debate_payload:
#                 display_json_friendly("Debate Output", debate_payload)
#             else:
#                 st.info("Debate agent was not run")

#         with tab_news:
#             if isinstance(news_payload, dict):
#                 summaries = news_payload.get("summaries") or news_payload.get("articles") or []
#                 if summaries:
#                     for s in summaries[:10]:
#                         if isinstance(s, dict):
#                             title = s.get("title", s.get("headline", "No title"))
#                             src = s.get("source", "unknown")
#                             st.write(f"- {title} — {src}")
#                         else:
#                             st.write(f"- {s}")
#                 else:
#                     st.info("No news articles found")
#             else:
#                 st.write(news_payload)

#         # Quick execution panel (simulated)
#         st.markdown("---")
#         if isinstance(master_payload, dict) and (master_payload.get("action") or master_payload.get("recommendation")):
#             master_action = master_payload.get("action", master_payload.get("recommendation", "HOLD"))
#             if master_action != "HOLD":
#                 qty_default = int(port_payload.get("suggested_quantity", port_payload.get("quantity", 1)) if isinstance(port_payload, dict) else 1)
#                 qty = st.number_input("Quantity", min_value=1, value=qty_default)
#                 if st.button(f"Simulate {master_action}"):
#                     st.success(f"Simulated {master_action} {qty} @ {latest_close:.2f}")
#             else:
#                 st.info("Master recommends HOLD")

#     except Exception as e:
#         logger.exception("Pipeline failure")
#         st.error(f"Pipeline error: {e}")
#         with st.expander("Trace"):
#             import traceback
#             st.code(traceback.format_exc())


# # -------------------------
# # Agent Runner - FIXED REGISTRATION
# # -------------------------
# import logging
# from typing import Dict, Any
# import time

# # FIXED IMPORTS (correct for your folder structure)
# from agents.base_agent import BaseAgent
# from tools.toolbox import TOOLS
# from llm.llm_wrapper import LLM   # your Groq/Dummy unified wrapper

# # Try to import institutional wrapper factory if present
# try:
#     from agents.inst_wrappers import create_inst_wrappers
# except Exception:
#     create_inst_wrappers = None

# # Try to import regular wrapper factory if present
# try:
#     from agents.wrappers import create_wrapped_agents
# except Exception:
#     create_wrapped_agents = None

# logger = logging.getLogger(__name__)


# class AgentRunner:
#     """
#     Central manager for all agents.
#     - Registers agents by name
#     - Injects tools + LLM into every agent
#     - Runs agents safely
#     - Supports both: regular 5 agents + 9 institutional agents
#     """

#     def __init__(self):
#         self.agents: Dict[str, BaseAgent] = {}

#         # shared tools + shared LLM wrapper (Groq or Dummy)
#         self.tools = TOOLS
#         self.llm = LLM()

#         logger.info("🤖 AgentRunner initialized with unified tools + LLM")

#         # Use single registration method
#         self.registerAllAgents()
        
#         logger.info(f"🎯 Final registered agents: {list(self.agents.keys())}")

#     def registerAllAgents(self):
#         """Register all core agents - single source of truth"""
#         try:
#             from agents.wrappers import (
#                 TechnicalAgent, RiskAgent, PortfolioAgent, 
#                 DebateAgent, MasterAgent, NewsAgent,
#                 ProfessionalSentimentAgent
#             )
            
#             core_agents = {
#                 "technical": TechnicalAgent(tools=self.tools, llm=self.llm),
#                 "risk": RiskAgent(tools=self.tools, llm=self.llm),
#                 "portfolio": PortfolioAgent(tools=self.tools, llm=self.llm),
#                 "debate": DebateAgent(tools=self.tools, llm=self.llm),
#                 "master": MasterAgent(tools=self.tools, llm=self.llm),
#                 "news": NewsAgent(tools=self.tools, llm=self.llm),
#                 "sentiment": ProfessionalSentimentAgent(tools=self.tools, llm=self.llm),
#             }
            
#             for name, agent in core_agents.items():
#                 self.register(name, agent)
                
#             logger.info(f"✅ Successfully registered: {list(core_agents.keys())}")
            
#         except Exception as e:
#             logger.error(f"❌ Failed to register agents: {e}")
#             # Don't raise, allow graceful degradation

#     # --------------------------------------------------------------
#     # REGISTER ANY AGENT
#     # --------------------------------------------------------------
#     def register(self, name: str, agent: BaseAgent):
#         if not isinstance(name, str):
#             raise ValueError("Agent name must be a string")

#         if not isinstance(agent, BaseAgent):
#             raise TypeError(f"Agent '{name}' must inherit BaseAgent")

#         self.agents[name] = agent
#         logger.info(f"📋 Registered agent: {name}")

#     # --------------------------------------------------------------
#     # RUN ONE AGENT
#     # --------------------------------------------------------------
#     def run(self, name: str, user_input: Dict[str, Any]):
#         """
#         Run one agent and return its output.
#         Inject tools + LLM automatically.
#         """
#         agent = self.agents.get(name)
#         if not agent:
            
#             logger.error(f"Unknown agent: {name}. Available: {list(self.agents.keys())}")
#             raise RuntimeError(f"Unknown agent: {name}")
            
#         logger.info(f"🚀 Running agent: {name}")

#         # Inject dependencies if missing
#         agent.tools = getattr(agent, "tools", None) or self.tools
#         agent.llm = getattr(agent, "llm", None) or self.llm

#         # Execute safely with timing
#         start_time = time.time()
#         try:
#             result = agent.run(user_input)
#             elapsed = time.time() - start_time
#             logger.info(f"✅ Agent '{name}' completed in {elapsed:.2f}s")
#             return result

#         except Exception as e:
#             elapsed = time.time() - start_time
#             logger.error(f"❌ Agent '{name}' failed after {elapsed:.2f}s: {e}", exc_info=True)
#             return {
#                 "status": "ERROR",
#                 "agent": name,
#                 "error": str(e)
#             }

#     # --------------------------------------------------------------
#     # RUN ALL AGENTS (optional but recommended)
#     # --------------------------------------------------------------
#     def run_all(self, user_input: Dict[str, Any]):
#         """
#         Runs EVERY registered agent.
#         Returns a dict: { agent_name: result }
#         """
#         outputs = {}
#         logger.info(f"Running all agents: {list(self.agents.keys())}")

#         for name, agent in self.agents.items():
#             try:
#                 outputs[name] = self.run(name, user_input)
#             except Exception as e:
#                 logger.error(f"Agent '{name}' crashed inside run_all: {e}")
#                 outputs[name] = {
#                     "status": "ERROR",
#                     "agent": name,
#                     "error": str(e)
#                 }
    
#         return outputs

# trading_bot/ui/ai_agents.py
import os
from datetime import datetime, timedelta
import logging
import streamlit as st
import pandas as pd
import json
import re

from data.data_fetcher import fetch_data
from agent_runner import AgentRunner
try:
    from agents.wrappers import create_wrapped_agents
except Exception:
    create_wrapped_agents = None

try:
    from agents.inst_wrappers import create_inst_wrappers
except Exception:
    create_inst_wrappers = None

from tools.toolbox import TOOLS
from llm.llm_wrapper import LLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_master_output(master_response):
    """
    Specialized extraction for Master Agent that handles multiple output structures
    """
    try:
        if master_response is None:
            return {"status": "ERROR", "error": "No master response"}
        
        logger.info(f"Master response type: {type(master_response)}")
        
        # Case 1: Already the clean output we want
        if (isinstance(master_response, dict) and 
            any(key in master_response for key in ["action", "recommendation", "decision", "analysis"])):
            logger.info("Found clean master output (direct)")
            return master_response
        
        # Case 2: Nested in result -> master
        if (isinstance(master_response, dict) and 
            "result" in master_response):
            result = master_response["result"]
            if isinstance(result, dict) and "master" in result:
                logger.info("Found master output in result->master")
                return result["master"]
            elif isinstance(result, dict) and any(key in result for key in ["action", "recommendation", "decision"]):
                logger.info("Found action keys in result")
                return result
            elif isinstance(result, str):
                # Try to parse string result as JSON
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict) and "master" in parsed_result:
                        return parsed_result["master"]
                    elif isinstance(parsed_result, dict):
                        return parsed_result
                except:
                    pass
        
        # Case 3: Direct master key
        if (isinstance(master_response, dict) and 
            "master" in master_response):
            master_data = master_response["master"]
            if isinstance(master_data, dict):
                logger.info("Found master output in direct master key")
                return master_data
            elif isinstance(master_data, str):
                try:
                    return json.loads(master_data)
                except:
                    return {"analysis": master_data}
        
        # Case 4: Text response - try to extract structure
        if isinstance(master_response, str):
            logger.info("Master response is string, attempting to parse")
            # Try JSON parsing first
            try:
                parsed = json.loads(master_response)
                if isinstance(parsed, dict):
                    return extract_master_output(parsed)
            except:
                pass
            
            # Look for structured patterns in text
            lines = master_response.split('\n')
            result = {}
            current_key = None
            current_value = []
            
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith(' '):
                    # Save previous key-value pair
                    if current_key and current_value:
                        result[current_key.lower()] = ' '.join(current_value).strip()
                    
                    # Start new key
                    parts = line.split(':', 1)
                    current_key = parts[0].strip()
                    current_value = [parts[1].strip()] if len(parts) > 1 else []
                elif current_key and line:
                    current_value.append(line)
                elif line and not current_key:
                    # If no structure found, treat as analysis
                    current_key = "analysis"
                    current_value.append(line)
            
            # Add the last key-value pair
            if current_key and current_value:
                result[current_key.lower()] = ' '.join(current_value).strip()
            
            if result:
                logger.info("Extracted structure from text response")
                return result
        
        # Case 5: Try to find ANY nested dict that looks like master output
        if isinstance(master_response, dict):
            for key, value in master_response.items():
                if (isinstance(value, dict) and 
                    any(k in value for k in ["action", "recommendation", "decision", "analysis"])):
                    logger.info(f"Found master output in key: {key}")
                    return value
        
        # Final fallback: return as analysis text
        logger.warning("Using fallback - treating entire response as analysis")
        if isinstance(master_response, dict):
            return {"analysis": str(master_response)}
        else:
            return {"analysis": str(master_response)}
        
    except Exception as e:
        logger.error(f"Error extracting master output: {e}")
        return {"status": "ERROR", "error": f"Extraction failed: {e}", "analysis": str(master_response)}


def calculate_missing_metrics(master_payload: dict) -> dict:
    """
    Calculate missing metrics like risk/reward ratio, percentage changes, etc.
    """
    if not isinstance(master_payload, dict):
        return master_payload
    
    enhanced = master_payload.copy()
    current_price = enhanced.get("current_price")
    stop_loss = enhanced.get("stop_loss")
    take_profit = enhanced.get("take_profit")
    action = enhanced.get("action", "HOLD")
    
    # Calculate Risk/Reward Ratio if missing
    if (enhanced.get("risk_reward_ratio") in [None, "NULL", "N/A", 0] and 
        current_price and stop_loss and take_profit and stop_loss != take_profit):
        try:
            if str(action).upper() in ["BUY", "LONG"]:
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
            else:  # SELL/SHORT
                risk = abs(stop_loss - current_price)
                reward = abs(current_price - take_profit)
                
            if risk > 0:
                rr_ratio = round(reward / risk, 2)
                enhanced["risk_reward_ratio"] = rr_ratio
                enhanced["risk_percent"] = round((risk / current_price) * 100, 1)
                enhanced["reward_percent"] = round((reward / current_price) * 100, 1)
        except (TypeError, ZeroDivisionError):
            enhanced["risk_reward_ratio"] = "N/A"
    
    # Calculate position sizing if quantity is 0
    if enhanced.get("quantity") in [0, None, "NULL"] and current_price:
        # Simple position sizing based on risk
        enhanced["suggested_quantity"] = 100  # Default minimum
        enhanced["position_size"] = round(100 * current_price, 2)
    
    # Ensure confidence is properly formatted
    if "confidence" in enhanced:
        try:
            enhanced["confidence"] = int(float(enhanced["confidence"]))
        except (TypeError, ValueError):
            enhanced["confidence"] = 50
    
    # Format prices to 2 decimal places
    price_fields = ["current_price", "entry_price", "stop_loss", "take_profit"]
    for field in price_fields:
        if field in enhanced and enhanced[field] is not None:
            try:
                enhanced[field] = round(float(enhanced[field]), 2)
            except (TypeError, ValueError):
                pass
    
    return enhanced


def generate_comprehensive_llm_reasoning(master_payload: dict, technical_payload: dict, risk_payload: dict, sentiment_payload: dict, ticker: str) -> str:
    """
    Generate comprehensive LLM reasoning by analyzing all agent outputs
    This creates rich, detailed reasoning even if Master Agent provides minimal output
    """
    try:
        # Extract key metrics from all agents
        current_price = master_payload.get("current_price")
        action = master_payload.get("action", "HOLD")
        confidence = master_payload.get("confidence", 50)
        risk_level = master_payload.get("risk_level", "MEDIUM")
        
        # Parse original reasoning to extract actual values
        original_reasoning = master_payload.get("reasoning", "")
        tech_signal_actual = "NEUTRAL"
        tech_confidence_actual = 50
        sentiment_actual = "neutral"
        sentiment_confidence_actual = 50
        
        # Parse technical from original reasoning
        tech_match = re.search(r'Technical:\s*(\w+)\s*\(([\d.]+)%\)', original_reasoning)
        if tech_match:
            tech_signal_actual = tech_match.group(1)
            tech_confidence_actual = float(tech_match.group(2))
        
        # Parse sentiment from original reasoning
        sentiment_match = re.search(r'Sentiment:\s*(\w+)\s*\(([\d.]+)%\)', original_reasoning)
        if sentiment_match:
            sentiment_actual = sentiment_match.group(1)
            sentiment_confidence_actual = float(sentiment_match.group(2))
        
        # Technical Analysis Insights
        tech_insights = []
        tech_insights.append(f"Technical signal: {tech_signal_actual} ({tech_confidence_actual}% confidence)")
        
        # Add technical indicators if available
        if isinstance(technical_payload, dict):
            indicators = technical_payload.get("indicators", {})
            if isinstance(indicators, dict):
                for indicator, value in indicators.items():
                    if isinstance(value, (int, float)) and indicator in ["rsi", "macd", "stochastic"]:
                        tech_insights.append(f"{indicator.upper()}: {value}")
        
        # Risk Analysis Insights
        risk_insights = []
        if isinstance(risk_payload, dict):
            volatility = risk_payload.get("volatility", "MEDIUM")
            max_drawdown = risk_payload.get("max_drawdown")
            var = risk_payload.get("var")
            
            risk_insights.append(f"Volatility: {volatility}")
            if max_drawdown:
                risk_insights.append(f"Max Drawdown: {max_drawdown}%")
            if var:
                risk_insights.append(f"VaR: {var}%")
        
        # Sentiment Analysis Insights - Use actual parsed values
        sentiment_insights = []
        sentiment_insights.append(f"Overall sentiment: {sentiment_actual.upper()} (confidence: {sentiment_confidence_actual}%)")
        
        # Add sentiment details if available
        if isinstance(sentiment_payload, dict):
            article_count = sentiment_payload.get("article_count")
            positive_articles = sentiment_payload.get("positive_articles")
            negative_articles = sentiment_payload.get("negative_articles")
            neutral_articles = sentiment_payload.get("neutral_articles")
            
            if article_count:
                sentiment_insights.append(f"Based on {article_count} articles: {positive_articles or 0} positive, {negative_articles or 0} negative, {neutral_articles or 0} neutral")
        
        # Trade Setup Analysis
        trade_analysis = []
        stop_loss = master_payload.get("stop_loss")
        take_profit = master_payload.get("take_profit")
        rr_ratio = master_payload.get("risk_reward_ratio")
        
        if stop_loss and take_profit and current_price:
            if str(action).upper() in ["BUY", "LONG"]:
                risk_amount = current_price - stop_loss
                reward_amount = take_profit - current_price
            else:
                risk_amount = stop_loss - current_price
                reward_amount = current_price - take_profit
            
            if risk_amount > 0:
                risk_percent = master_payload.get("risk_percent")
                risk_display = f"₹{risk_amount:.2f}"
                if risk_percent:
                    risk_display += f" ({risk_percent}%)"
                trade_analysis.append(f"Risk per share: {risk_display}")
            
            if reward_amount > 0:
                reward_percent = master_payload.get("reward_percent")
                reward_display = f"₹{reward_amount:.2f}"
                if reward_percent:
                    reward_display += f" ({reward_percent}%)"
                trade_analysis.append(f"Reward per share: {reward_display}")
            
            if rr_ratio and rr_ratio != "N/A":
                trade_analysis.append(f"Risk/Reward ratio: {rr_ratio}:1")
        
        # Build comprehensive reasoning
        reasoning_parts = []
        
        # Decision Summary
        reasoning_parts.append(f"## 📊 Comprehensive Analysis for {ticker}")
        reasoning_parts.append("")
        
        action_display = str(action).upper() if action else "HOLD"
        reasoning_parts.append(f"**Final Decision: {action_display}** with {confidence}% confidence")
        reasoning_parts.append(f"**Risk Assessment:** {risk_level} risk level")
        reasoning_parts.append("")
        
        # Market Overview
        reasoning_parts.append("### 📈 Market Overview")
        
        # Determine market alignment
        if tech_signal_actual.upper() in ["BUY", "BULLISH"] and sentiment_actual.upper() in ["POSITIVE", "BULLISH"]:
            reasoning_parts.append("- 🟢 **Strong bullish alignment** between technical and sentiment analysis")
        elif tech_signal_actual.upper() in ["SELL", "BEARISH"] and sentiment_actual.upper() in ["NEGATIVE", "BEARISH"]:
            reasoning_parts.append("- 🔴 **Strong bearish alignment** between technical and sentiment analysis")
        elif tech_signal_actual.upper() in ["BUY", "BULLISH"] and sentiment_actual.upper() in ["NEGATIVE", "BEARISH"]:
            reasoning_parts.append("- 🟡 **Mixed signals** - Bullish technicals vs Bearish sentiment")
        elif tech_signal_actual.upper() in ["SELL", "BEARISH"] and sentiment_actual.upper() in ["POSITIVE", "BULLISH"]:
            reasoning_parts.append("- 🟡 **Mixed signals** - Bearish technicals vs Bullish sentiment")
        else:
            reasoning_parts.append("- ⚪ **Neutral market conditions** - Balanced signals across indicators")
        
        reasoning_parts.append("")
        
        # Technical Analysis Section
        reasoning_parts.append("### 🔧 Technical Analysis")
        reasoning_parts.extend([f"- {insight}" for insight in tech_insights])
        
        # Add technical interpretation
        if tech_signal_actual.upper() in ["BUY", "BULLISH"]:
            reasoning_parts.append("- 📈 **Technical Outlook:** Bullish momentum with positive indicators")
        elif tech_signal_actual.upper() in ["SELL", "BEARISH"]:
            reasoning_parts.append("- 📉 **Technical Outlook:** Bearish pressure with negative indicators")
        else:
            reasoning_parts.append("- ⚖️ **Technical Outlook:** Neutral with mixed or consolidating signals")
        
        reasoning_parts.append("")
        
        # Sentiment Analysis Section
        reasoning_parts.append("### 😊 Market Sentiment")
        reasoning_parts.extend([f"- {insight}" for insight in sentiment_insights])
        
        # Add sentiment interpretation
        if sentiment_actual.upper() in ["POSITIVE", "BULLISH"]:
            reasoning_parts.append("- 👍 **Sentiment Outlook:** Positive market sentiment with favorable news flow")
        elif sentiment_actual.upper() in ["NEGATIVE", "BEARISH"]:
            reasoning_parts.append("- 👎 **Sentiment Outlook:** Negative sentiment with concerning developments")
        else:
            reasoning_parts.append("- 🤝 **Sentiment Outlook:** Neutral sentiment with balanced news coverage")
        
        reasoning_parts.append("")
        
        # Risk Assessment Section
        if risk_insights:
            reasoning_parts.append("### ⚠️ Risk Assessment")
            reasoning_parts.extend([f"- {insight}" for insight in risk_insights])
            
            # Add risk interpretation
            risk_level_str = str(risk_level).upper() if risk_level else "MEDIUM"
            if risk_level_str in ["HIGH", "VERY_HIGH"]:
                reasoning_parts.append("- 🔴 **High Risk Environment:** Elevated volatility and potential for large moves")
            elif risk_level_str in ["MEDIUM"]:
                reasoning_parts.append("- 🟡 **Moderate Risk:** Standard market conditions with typical volatility")
            else:
                reasoning_parts.append("- 🟢 **Low Risk:** Stable conditions with limited downside risk")
            
            reasoning_parts.append("")
        
        # Trade Setup Section
        if trade_analysis:
            reasoning_parts.append("### 💼 Trade Setup")
            reasoning_parts.extend([f"- {analysis}" for analysis in trade_analysis])
            
            # Add R/R interpretation
            if rr_ratio and rr_ratio != "N/A":
                if rr_ratio >= 2:
                    reasoning_parts.append("- 🟢 **Excellent Risk/Reward:** Favorable ratio for position sizing")
                elif rr_ratio >= 1:
                    reasoning_parts.append("- 🟡 **Good Risk/Reward:** Acceptable ratio for trading")
                else:
                    reasoning_parts.append("- 🔴 **Poor Risk/Reward:** Unfavorable ratio - consider adjusting targets")
            
            reasoning_parts.append("")
        
        # Strategic Recommendation
        reasoning_parts.append("### 🎯 Strategic Recommendation")
        
        if action_display in ["BUY", "LONG"]:
            if confidence >= 70:
                reasoning_parts.append("- 🟢 **Strong Buy Conviction:** High-confidence bullish setup")
                reasoning_parts.append("- 💡 **Strategy:** Consider aggressive position sizing with tight risk management")
                reasoning_parts.append("- 📊 **Target:** Primary objective at take-profit level")
            elif confidence >= 50:
                reasoning_parts.append("- 🟡 **Moderate Buy Opportunity:** Reasonable bullish case")
                reasoning_parts.append("- 💡 **Strategy:** Standard position sizing with defined exit points")
                reasoning_parts.append("- 📊 **Target:** Conservative approach to take-profit")
            else:
                reasoning_parts.append("- 🟠 **Cautious Buy Consideration:** Weak bullish signals")
                reasoning_parts.append("- 💡 **Strategy:** Small position size or wait for confirmation")
                reasoning_parts.append("- 📊 **Target:** Consider scaling into position")
                
        elif action_display in ["SELL", "SHORT"]:
            if confidence >= 70:
                reasoning_parts.append("- 🔴 **Strong Sell Conviction:** High-confidence bearish setup")
                reasoning_parts.append("- 💡 **Strategy:** Consider short positions with strict risk limits")
                reasoning_parts.append("- 📊 **Target:** Primary objective at take-profit level")
            elif confidence >= 50:
                reasoning_parts.append("- 🟡 **Moderate Sell Opportunity:** Reasonable bearish case")
                reasoning_parts.append("- 💡 **Strategy:** Hedging or reduced long exposure")
                reasoning_parts.append("- 📊 **Target:** Conservative approach to targets")
            else:
                reasoning_parts.append("- 🟠 **Cautious Sell Consideration:** Weak bearish signals")
                reasoning_parts.append("- 💡 **Strategy:** Wait for confirmation or consider options strategies")
                reasoning_parts.append("- 📊 **Target:** Limited position size")
        else:  # HOLD
            reasoning_parts.append("- ⚪ **Neutral Stance Recommended:** Insufficient edge for directional trade")
            reasoning_parts.append("- 💡 **Strategy:** Maintain current positions or stay in cash")
            reasoning_parts.append("- 📊 **Action:** Wait for clearer market direction before committing capital")
            reasoning_parts.append("- 🔍 **Monitor:** Key levels at stop-loss and take-profit for breakout signals")
        
        reasoning_parts.append("")
        
        # Risk Management
        reasoning_parts.append("### 🛡️ Risk Management")
        
        risk_level_str = str(risk_level).upper() if risk_level else "MEDIUM"
        if risk_level_str in ["HIGH", "VERY_HIGH"]:
            reasoning_parts.append("- 🔴 **High Risk Protocol:**")
            reasoning_parts.append("  - Reduce position size by 50-70% from normal allocation")
            reasoning_parts.append("  - Implement tight stop-losses and daily monitoring")
            reasoning_parts.append("  - Consider hedging strategies for portfolio protection")
            reasoning_parts.append("  - Prepare for increased volatility and larger price swings")
        elif risk_level_str in ["MEDIUM"]:
            reasoning_parts.append("- 🟡 **Standard Risk Protocol:**")
            reasoning_parts.append("  - Use normal position sizing according to your risk tolerance")
            reasoning_parts.append("  - Regular monitoring with weekly position reviews")
            reasoning_parts.append("  - Balanced approach between growth and capital preservation")
        else:  # LOW
            reasoning_parts.append("- 🟢 **Low Risk Protocol:**")
            reasoning_parts.append("  - Consider aggressive position sizing for high-conviction ideas")
            reasoning_parts.append("  - Favorable conditions for trend-following strategies")
            reasoning_parts.append("  - Opportunity to add to winning positions on pullbacks")
        
        reasoning_parts.append("")
        
        # Market Context & Next Steps
        reasoning_parts.append("### 🌍 Market Context & Next Steps")
        
        # Time horizon based on confidence and risk
        if confidence >= 70:
            time_horizon = "Short to medium term (1-4 weeks)"
        elif confidence >= 50:
            time_horizon = "Short term (1-2 weeks)"
        else:
            time_horizon = "Very short term (intraday to 1 week)"
        
        reasoning_parts.append(f"- ⏰ **Time Horizon:** {time_horizon}")
        reasoning_parts.append("- 📈 **Key Levels to Watch:**")
        
        if stop_loss:
            reasoning_parts.append(f"  - **Support/Stop-Loss:** ₹{stop_loss:.2f}")
        if take_profit:
            reasoning_parts.append(f"  - **Resistance/Take-Profit:** ₹{take_profit:.2f}")
        if current_price:
            reasoning_parts.append(f"  - **Current Level:** ₹{current_price:.2f}")
        
        reasoning_parts.append("- 🔄 **Next Catalyst:** Monitor for breakout above/below key technical levels")
        reasoning_parts.append("- 📰 **News Monitor:** Watch for earnings, sector news, or market-moving events")
        
        return "\n".join(reasoning_parts)
        
    except Exception as e:
        logger.error(f"Error generating comprehensive reasoning: {e}")
        return f"## 📊 Comprehensive Analysis for {ticker}\n\n**Analysis Generation Issue**\n\nWe're experiencing technical difficulties generating the full analysis. Please refer to the individual agent tabs below for complete details.\n\n*Error details: {str(e)}*"


def enhance_with_ai_reasoning(master_payload: dict, technical_payload: dict, risk_payload: dict, sentiment_payload: dict, ticker: str) -> dict:
    """
    Force-enhance the payload with comprehensive AI reasoning
    """
    if not isinstance(master_payload, dict):
        return master_payload
    
    enhanced = master_payload.copy()
    
    # Generate comprehensive LLM reasoning
    llm_reasoning = generate_comprehensive_llm_reasoning(
        master_payload, technical_payload, risk_payload, sentiment_payload, ticker
    )
    
    # Always set AI enhanced to true since we're generating reasoning
    enhanced["ai_enhanced"] = True
    enhanced["llm_reasoning"] = llm_reasoning
    
    # Preserve original reasoning if it exists
    if "reasoning" in enhanced and enhanced["reasoning"]:
        enhanced["original_reasoning"] = enhanced["reasoning"]
    
    # Add metadata about the enhancement
    enhanced["analysis_timestamp"] = datetime.now().isoformat()
    enhanced["comprehensive_analysis"] = True
    
    return enhanced


def normalize_response(resp):
    """
    Convert possible agent responses into a plain dict.
    """
    try:
        if resp is None:
            return {"status": "ERROR", "error": "No response (None)"}

        if isinstance(resp, dict):
            return dict(resp)

        if isinstance(resp, pd.DataFrame):
            return {"status": "OK", "df": resp}

        # pydantic-style
        if hasattr(resp, "dict") and callable(getattr(resp, "dict")):
            try:
                return {"status": "OK", **resp.dict()}
            except Exception:
                pass

        if hasattr(resp, "__dict__"):
            try:
                return {"status": "OK", **vars(resp)}
            except Exception:
                pass

        return {"status": "OK", "text": str(resp)}
    except Exception as e:
        logger.exception("normalize_response failed")
        return {"status": "ERROR", "error": f"normalize_response failed: {e}"}
    

def extract_payload(norm: dict, preferred_keys=("result", "technical", "master", "risk", "portfolio")):
    """
    From a normalized dict, pick the nested payload under preferred_keys,
    else return the normalized dict itself.
    """
    if not isinstance(norm, dict):
        return {"status": "ERROR", "error": "normalize_response did not return dict"}

    for k in preferred_keys:
        if k in norm and isinstance(norm[k], dict):
            return norm[k]
    # sometimes responses are {'status':'OK', 'df': df}
    if "df" in norm or "status" in norm:
        return norm
    return norm


def display_json_friendly(label: str, payload, expand: bool = False):
    """
    Safely display payload in Streamlit
    """
    st.markdown(f"### {label}")
    if payload is None:
        st.info("No output")
        return

    if isinstance(payload, dict):
        df = payload.get("df", None)
        try:
            st.json({k: v for k, v in payload.items() if k != "df"})
        except Exception:
            st.write(payload)
        if df is not None:
            if isinstance(df, pd.DataFrame):
                st.markdown("**Data (preview)**")
                st.dataframe(df.head(10))
    elif isinstance(payload, pd.DataFrame):
        st.dataframe(payload.head(10))
    elif isinstance(payload, list):
        st.write(payload)
    else:
        st.write(str(payload))


def display_master_agent_analysis(master_payload: dict):
    """
    Enhanced display for Master Agent output with rich LLM reasoning
    """
    if not isinstance(master_payload, dict):
        st.error("No master analysis available")
        return

    st.markdown("---")
    
    # Extract and validate action
    action_keys = ["action", "recommendation", "decision"]
    action = "HOLD"
    for key in action_keys:
        if key in master_payload and master_payload[key]:
            action = master_payload[key]
            break
    
    # Extract and validate confidence
    confidence = master_payload.get("confidence")
    if confidence in [None, "NULL", "N/A"]:
        confidence = 50  # Default fallback
    else:
        try:
            confidence = int(float(confidence))
        except (TypeError, ValueError):
            confidence = 50
    
    # Main Decision Card
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        action_display = str(action).upper() if action else "HOLD"
        if any(buy_word in action_display for buy_word in ["BUY", "LONG", "BULLISH"]):
            st.success(f"## 🟢 {action_display}")
        elif any(sell_word in action_display for sell_word in ["SELL", "SHORT", "BEARISH"]):
            st.error(f"## 🔴 {action_display}")  
        else:
            st.info(f"## ⚪ {action_display}")
            
    with col2:
        confidence_color = "🟡"
        if confidence >= 70:
            confidence_color = "🟢"
        elif confidence <= 30:
            confidence_color = "🔴"
        st.metric(f"{confidence_color} Confidence", f"{confidence}%")
        
    with col3:
        risk_level = master_payload.get("risk_level", "MEDIUM")
        risk_display = str(risk_level).upper() if risk_level else "MEDIUM"
        if risk_display in ["LOW", "VERY_LOW"]:
            st.success(f"🛡️ Risk: {risk_display}")
        elif risk_display in ["HIGH", "VERY_HIGH"]:
            st.error(f"⚡ Risk: {risk_display}")
        else:
            st.warning(f"⚠️ Risk: {risk_display}")
    
    with col4:
        current_price = master_payload.get("current_price")
        if current_price:
            st.metric("💰 Current Price", f"₹{current_price:.2f}")
        else:
            st.write("💰 Price: N/A")
    
    # Enhanced LLM Reasoning Section
    st.markdown("### 🧠 AI Comprehensive Analysis")
    
    # Get LLM reasoning - prioritize our enhanced reasoning
    llm_reasoning = master_payload.get("llm_reasoning")
    original_reasoning = master_payload.get("original_reasoning")
    
    if llm_reasoning and "Analysis Generation Issue" not in llm_reasoning:
        # Display our comprehensive LLM reasoning with proper markdown
        st.markdown(llm_reasoning)
        
        # Show original reasoning if it exists and is different
        if original_reasoning and original_reasoning != llm_reasoning and len(original_reasoning) > 10:
            with st.expander("📋 Original Agent Reasoning"):
                st.info(original_reasoning)
    elif original_reasoning:
        # Fallback to original reasoning
        st.info(original_reasoning)
        st.warning("⚠️ Basic reasoning only - comprehensive analysis not available")
    else:
        st.warning("No reasoning analysis available")
    
    # Enhanced Price Targets Section
    stop_loss = master_payload.get("stop_loss")
    take_profit = master_payload.get("take_profit")
    rr_ratio = master_payload.get("risk_reward_ratio")
    risk_percent = master_payload.get("risk_percent")
    reward_percent = master_payload.get("reward_percent")
    
    if stop_loss or take_profit:
        st.markdown("### 🎯 Trade Setup")
        price_col1, price_col2, price_col3, price_col4 = st.columns(4)
        
        with price_col1:
            if stop_loss:
                stop_loss_pct = ""
                if risk_percent:
                    stop_loss_pct = f"({risk_percent}%)"
                st.metric("🛑 Stop Loss", f"₹{stop_loss:.2f}", delta=stop_loss_pct, delta_color="inverse")
            else:
                st.write("🛑 Stop Loss: N/A")
                
        with price_col2:
            if take_profit:
                take_profit_pct = ""
                if reward_percent:
                    take_profit_pct = f"({reward_percent}%)"
                st.metric("🎯 Take Profit", f"₹{take_profit:.2f}", delta=take_profit_pct)
            else:
                st.write("🎯 Take Profit: N/A")
                
        with price_col3:
            if rr_ratio and rr_ratio != "N/A":
                rr_quality = ""
                if rr_ratio >= 2:
                    rr_quality = "🟢"
                elif rr_ratio >= 1:
                    rr_quality = "🟡"
                else:
                    rr_quality = "🔴"
                st.metric(f"{rr_quality} R/R Ratio", f"{rr_ratio}:1")
            else:
                st.write("⚖️ R/R Ratio: N/A")
                
        with price_col4:
            if current_price and stop_loss:
                if action_display in ["BUY", "LONG"]:
                    distance_to_stop = current_price - stop_loss
                    distance_to_take = take_profit - current_price if take_profit else 0
                else:
                    distance_to_stop = stop_loss - current_price
                    distance_to_take = current_price - take_profit if take_profit else 0
                
                if distance_to_stop > 0:
                    st.metric("📏 Stop Distance", f"₹{distance_to_stop:.2f}")
    
    # Position Information
    st.markdown("### 💼 Position Details")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    
    with info_col1:
        quantity = master_payload.get("quantity") or master_payload.get("suggested_quantity")
        if quantity and quantity > 0:
            st.metric("📦 Quantity", f"{quantity}")
        else:
            st.write("📦 Quantity: N/A")
    
    with info_col2:
        entry_price = master_payload.get("entry_price")
        if entry_price:
            st.metric("🎫 Entry Price", f"₹{entry_price:.2f}")
        else:
            st.write("🎫 Entry Price: N/A")
    
    with info_col3:
        ai_enhanced = master_payload.get("ai_enhanced", False)
        if ai_enhanced:
            st.success("🤖 AI Enhanced: ✅ Yes")
        else:
            st.warning("🤖 AI Enhanced: ❌ No")
    
    with info_col4:
        timestamp = master_payload.get("timestamp") or master_payload.get("analysis_timestamp")
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                st.write(f"🕒 Analysis Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except:
                st.write(f"🕒 Timestamp: {timestamp}")
    
    # Strategy Summary
    if action_display != "HOLD":
        st.markdown("### 📈 Strategy Summary")
        
        if action_display in ["BUY", "LONG"]:
            if stop_loss and current_price:
                risk_amount = current_price - stop_loss
                st.write(f"**Strategy:** Buy at ₹{current_price:.2f} with stop loss at ₹{stop_loss:.2f}")
                st.write(f"**Risk per share:** ₹{risk_amount:.2f} ({risk_percent or 'N/A'}%)")
                if take_profit:
                    reward_amount = take_profit - current_price
                    st.write(f"**Reward per share:** ₹{reward_amount:.2f} ({reward_percent or 'N/A'}%)")
                    if rr_ratio and rr_ratio != "N/A":
                        quality = "Excellent" if rr_ratio >= 2 else "Good" if rr_ratio >= 1 else "Poor"
                        st.write(f"**Risk/Reward:** {rr_ratio}:1 - {quality}")
        
        elif action_display in ["SELL", "SHORT"]:
            if stop_loss and current_price:
                risk_amount = stop_loss - current_price
                st.write(f"**Strategy:** Sell at ₹{current_price:.2f} with stop loss at ₹{stop_loss:.2f}")
                st.write(f"**Risk per share:** ₹{risk_amount:.2f} ({risk_percent or 'N/A'}%)")
                if take_profit:
                    reward_amount = current_price - take_profit
                    st.write(f"**Reward per share:** ₹{reward_amount:.2f} ({reward_percent or 'N/A'}%)")
    
    # Debug info (collapsed)
    with st.expander("🔍 Raw Output"):
        st.json(master_payload)


# ... [Keep all other functions exactly the same as before - ensure_session_runner, manual_register_core_agents, safe_get_latest_close, validate_agent_inputs, show_ai_agents_page, AgentRunner]

def ensure_session_runner():
    """
    Initialize AgentRunner with guaranteed agent registration
    """
    if "agent_runner" not in st.session_state:
        runner = AgentRunner()
        
        if not runner.agents:
            st.warning("🤖 No agents auto-registered, manually registering core agents...")
            manual_register_core_agents(runner)
        
        st.session_state.agent_runner = runner
        st.session_state.wrapped_agents = list(runner.agents.keys())
        logger.info(f"✅ AgentRunner initialized with agents: {st.session_state.wrapped_agents}")
    
    return st.session_state.agent_runner


def manual_register_core_agents(runner):
    """Manually register core agents as backup"""
    try:
        from agents.wrappers import (
            TechnicalAgent, RiskAgent, PortfolioAgent, 
            DebateAgent, MasterAgent, NewsAgent,
            ProfessionalSentimentAgent
        )
        
        core_agents = {
            "technical": TechnicalAgent(tools=TOOLS, llm=LLM()),
            "risk": RiskAgent(tools=TOOLS, llm=LLM()),
            "portfolio": PortfolioAgent(tools=TOOLS, llm=LLM()),
            "debate": DebateAgent(tools=TOOLS, llm=LLM()),
            "master": MasterAgent(tools=TOOLS, llm=LLM()),
            "news": NewsAgent(tools=TOOLS, llm=LLM()),
            "sentiment": ProfessionalSentimentAgent(tools=TOOLS, llm=LLM()),
        }
        
        for name, agent in core_agents.items():
            runner.register(name, agent)
            
        logger.info(f"✅ Manually registered: {list(core_agents.keys())}")
        
    except Exception as e:
        logger.error(f"❌ Manual registration failed: {e}")
        st.error(f"Failed to register agents: {e}")


def safe_get_latest_close(payload):
    """
    Extract latest close price from a variety of payload shapes.
    """
    try:
        if payload is None:
            return 0.0

        if isinstance(payload, dict):
            for k in ("latest_close", "latest", "close", "price", "current_price"):
                if k in payload:
                    v = payload[k]
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, str) and v.replace('.', '', 1).isdigit():
                        try:
                            return float(v)
                        except Exception:
                            pass
            if "df" in payload and isinstance(payload["df"], pd.DataFrame):
                df = payload["df"]
                if "Close" in df.columns and len(df) > 0:
                    return float(df["Close"].iloc[-1])

        if isinstance(payload, pd.DataFrame):
            df = payload
            if "Close" in df.columns and len(df) > 0:
                return float(df["Close"].iloc[-1])

    except Exception:
        pass
    return 0.0


def validate_agent_inputs(ticker: str, price_df: pd.DataFrame, latest_close: float) -> bool:
    """Validate that we have fresh data for agent execution"""
    if not ticker or ticker.strip() == "":
        return False
    if price_df is None or len(price_df) == 0:
        return False
    if latest_close <= 0:
        return False
    return True


def show_ai_agents_page():
    st.set_page_config(layout="wide", page_title="SMART-MARKET — Agent Analysis")
    st.title("🤖 SMART-MARKET — Agent Analysis")

    runner = ensure_session_runner()

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_ready = bool(groq_key)

    with st.sidebar:
        st.header("Run settings")
        ticker = st.text_input("Ticker (e.g. RELIANCE.NS)", value=st.session_state.get("ticker", "RELIANCE.NS"))
        start_date_input = st.date_input("Start date", value=datetime.now() - timedelta(days=180))
        end_date_input = st.date_input("End date", value=datetime.now())
        show_debate = st.checkbox("Show Debate", value=True)
        run_btn = st.button("🚀 Analyze")
        st.markdown("---")
        st.markdown("Available Agents:")
        st.write(", ".join(st.session_state.get("wrapped_agents", [])))
        if not groq_ready:
            st.warning("GROQ_API_KEY not configured. LLM features disabled.")
        else:
            st.success("Groq ready ✅")

    st.session_state["ticker"] = ticker
    start_date = datetime.combine(start_date_input, datetime.min.time())
    end_date = datetime.combine(end_date_input, datetime.max.time())

    if not run_btn:
        st.info("Configure settings in the sidebar and click Analyze.")
        return

    # Run pipeline defensively
    progress = st.progress(0)
    status = st.empty()

    try:
        status.info("1/7 — Fetch canonical price data (via TOOLS)")
        progress.progress(5)

        price_df = None
        latest_close = 0.0
        try:
            if "fetch_price" in TOOLS:
                price_res = TOOLS["fetch_price"](ticker, start_date, end_date)
                if isinstance(price_res, dict) and price_res.get("status") == "OK" and price_res.get("df") is not None:
                    price_df = price_res.get("df")
                elif hasattr(price_res, "iloc"):
                    price_df = price_res
                else:
                    logger.warning("fetch_price returned unexpected shape: %s", type(price_res))
            else:
                logger.warning("fetch_price not available in TOOLS")
        except Exception as e:
            logger.exception("TOOLS.fetch_price failed: %s", e)
            status.error(f"Price fetch failed: {e}")

        if isinstance(price_df, pd.DataFrame) and len(price_df) > 0:
            latest_close = float(price_df["Close"].iloc[-1]) if "Close" in price_df.columns else 0.0

        # Validate inputs before proceeding
        if not validate_agent_inputs(ticker, price_df, latest_close):
            st.error("❌ Invalid input data. Please check ticker symbol and date range.")
            return

        # TECHNICAL
        status.info("2/7 — Running Technical Analysis")
        progress.progress(15)
        tech_raw = runner.run("technical", {"ticker": ticker, "start": start_date, "end": end_date})
        tech_norm = normalize_response(tech_raw)
        tech_payload = extract_payload(tech_norm, preferred_keys=("technical", "result"))
        if isinstance(tech_payload, dict) and "df" not in tech_payload and isinstance(price_df, pd.DataFrame):
            tech_payload["df"] = price_df

        latest_close = latest_close or safe_get_latest_close(tech_payload)

        # SENTIMENT
        status.info("3/7 — Running Sentiment Analysis")
        progress.progress(30)
        sentiment_raw = runner.run("sentiment", {"ticker": ticker})
        sentiment_norm = normalize_response(sentiment_raw)
        sentiment_payload = extract_payload(sentiment_norm)

        # NEWS
        status.info("4/7 — Running News Analysis")
        progress.progress(45)
        news_raw = runner.run("news", {"ticker": ticker, "limit": 10})
        news_norm = normalize_response(news_raw)
        news_payload = extract_payload(news_norm)

        # RISK
        status.info("5/7 — Running Risk Analysis")
        progress.progress(60)
        risk_input = {
            "ticker": ticker,
            "start": start_date,
            "end": end_date,
            "technical_confidence": tech_payload.get("confidence", 50) if isinstance(tech_payload, dict) else 50,
            "sentiment_confidence": sentiment_payload.get("confidence", 50) if isinstance(sentiment_payload, dict) else 50,
            "df": price_df,
            "current_price": latest_close
        }
        risk_raw = runner.run("risk", risk_input)
        risk_norm = normalize_response(risk_raw)
        risk_payload = extract_payload(risk_norm)

        # PORTFOLIO
        status.info("6/7 — Running Portfolio Analysis")
        progress.progress(75)
        port_input = {
            "ticker": ticker,
            "current_price": latest_close,
            "technical_signal": tech_payload if isinstance(tech_payload, dict) else {},
            "sentiment_signal": sentiment_payload if isinstance(sentiment_payload, dict) else {},
            "risk_metrics": risk_payload if isinstance(risk_payload, dict) else {},
            "portfolio_state": {},
            "df": price_df
        }
        port_raw = runner.run("portfolio", port_input)
        port_norm = normalize_response(port_raw)
        port_payload = extract_payload(port_norm)

        # DEBATE (optional)
        debate_payload = None
        if show_debate:
            status.info("7/7 — Running Debate Agent")
            progress.progress(85)
            debate_raw = runner.run("debate", {
                "ticker": ticker,
                "technical_result": tech_payload,
                "risk_metrics": risk_payload,
                "sentiment_score": sentiment_payload.get("confidence", 50) if isinstance(sentiment_payload, dict) else 50,
                "price_data": price_df
            })
            debate_norm = normalize_response(debate_raw)
            debate_payload = extract_payload(debate_norm)

        progress.progress(90)

        # MASTER - WITH COMPREHENSIVE AI REASONING
        status.info("Finalizing — Running Master Agent")
        master_input = {
            "ticker": ticker,
            "technical_result": tech_payload,
            "risk_metrics": risk_payload,
            "sentiment_result": sentiment_payload,
            "current_price": latest_close
        }

        master_raw = runner.run("master", master_input)

        # USE OUR SPECIALIZED EXTRACTOR
        master_payload = extract_master_output(master_raw)

        # FORCE-ENHANCE WITH COMPREHENSIVE AI REASONING
        master_payload = calculate_missing_metrics(master_payload)
        master_payload = enhance_with_ai_reasoning(master_payload, tech_payload, risk_payload, sentiment_payload, ticker)

        progress.progress(100)
        status.empty()
        progress.empty()

        # Check if we got valid master output
        action_keys = ["action", "recommendation", "decision", "verdict"]
        has_action = any(key in master_payload for key in action_keys) if isinstance(master_payload, dict) else False
        has_llm_reasoning = "llm_reasoning" in master_payload if isinstance(master_payload, dict) else False

        if (master_payload and isinstance(master_payload, dict) and 
            master_payload.get("status") != "ERROR" and
            (has_action or has_llm_reasoning)):
            st.success("✅ Analysis Complete with AI Enhancement")
        else:
            st.error("❌ Master analysis failed or returned invalid output")
            with st.expander("Debug Info"):
                st.write("Master Raw:", master_raw)
                st.write("Master Payload:", master_payload)

        # ---------- DISPLAY ----------
        tab_all, tab_master, tab_sentiment, tab_debate, tab_news = st.tabs(["All INFO", "Master Agent", "Sentiment Agent", "Debate Agent", "News "])

        with tab_all:
            col1, col2 = st.columns(2)
            with col1:
                display_json_friendly("Technical Analysis", tech_payload)
                st.markdown("---")
                display_json_friendly("Risk Analysis", risk_payload)
            with col2:
                display_json_friendly("Portfolio Analysis", port_payload)
                st.markdown("---")
                display_json_friendly("Sentiment Analysis", sentiment_payload)

        with tab_master:
            st.markdown("### 🧠 Master Agent Analysis")
            
            if master_payload is None:
                st.error("❌ Master payload is None")
            elif not isinstance(master_payload, dict):
                st.error(f"❌ Master payload is not a dict: {type(master_payload)}")
                st.write("Raw output:", master_payload)
            elif master_payload.get("status") == "ERROR":
                st.error("❌ Master analysis returned error")
                st.json(master_payload)
            else:
                # SUCCESS! We have a valid Master Agent output
                display_master_agent_analysis(master_payload)

        with tab_sentiment:
            display_json_friendly("Sentiment Analysis", sentiment_payload)

        with tab_debate:
            if debate_payload:
                display_json_friendly("Debate Output", debate_payload)
            else:
                st.info("Debate agent was not run")

        with tab_news:
            if isinstance(news_payload, dict):
                summaries = news_payload.get("summaries") or news_payload.get("articles") or []
                if summaries:
                    for s in summaries[:10]:
                        if isinstance(s, dict):
                            title = s.get("title", s.get("headline", "No title"))
                            src = s.get("source", "unknown")
                            st.write(f"- {title} — {src}")
                        else:
                            st.write(f"- {s}")
                else:
                    st.info("No news articles found")
            else:
                st.write(news_payload)

        # Quick execution panel (simulated)
        st.markdown("---")
        if isinstance(master_payload, dict):
            # Support multiple action key names
            action_keys = ["action", "recommendation", "decision", "verdict"]
            master_action = "HOLD"
            for key in action_keys:
                if key in master_payload:
                    master_action = master_payload[key]
                    break
            
            if master_action != "HOLD":
                # Use suggested quantity if available, otherwise calculate based on position size
                qty_default = master_payload.get("suggested_quantity", 100)
                position_size = master_payload.get("position_size")
                current_price = master_payload.get("current_price")
                
                if position_size and current_price and current_price > 0:
                    qty_default = max(100, int(position_size / current_price))
                
                st.markdown("#### 🎮 Quick Execution (Simulated)")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    qty = st.number_input("Quantity", min_value=1, value=qty_default, key="execution_qty")
                
                with col2:
                    if st.button(f"📤 Simulate {master_action} Order", type="primary", use_container_width=True):
                        order_value = qty * latest_close
                        st.success(f"✅ Simulated {master_action} order for {qty} shares @ ₹{latest_close:.2f}")
                        st.info(f"💵 Order Value: ₹{order_value:,.2f}")
            else:
                st.info("🎯 Master recommends HOLD - No trade execution suggested")

    except Exception as e:
        logger.exception("Pipeline failure")
        st.error(f"Pipeline error: {e}")
        with st.expander("Trace"):
            import traceback
            st.code(traceback.format_exc())


# Agent Runner class (same as before)
import logging
from typing import Dict, Any
import time

from agents.base_agent import BaseAgent
from tools.toolbox import TOOLS
from llm.llm_wrapper import LLM

logger = logging.getLogger(__name__)


class AgentRunner:
    """
    Central manager for all agents.
    - Registers agents by name
    - Injects tools + LLM into every agent
    - Runs agents safely
    - Supports both: regular 5 agents + 9 institutional agents
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}

        # shared tools + shared LLM wrapper (Groq or Dummy)
        self.tools = TOOLS
        self.llm = LLM()

        logger.info("🤖 AgentRunner initialized with unified tools + LLM")

        # Use single registration method
        self.registerAllAgents()
        
        logger.info(f"🎯 Final registered agents: {list(self.agents.keys())}")

    def registerAllAgents(self):
        """Register all core agents - single source of truth"""
        try:
            from agents.wrappers import (
                TechnicalAgent, RiskAgent, PortfolioAgent, 
                DebateAgent, MasterAgent, NewsAgent,
                ProfessionalSentimentAgent
            )
            
            core_agents = {
                "technical": TechnicalAgent(tools=self.tools, llm=self.llm),
                "risk": RiskAgent(tools=self.tools, llm=self.llm),
                "portfolio": PortfolioAgent(tools=self.tools, llm=self.llm),
                "debate": DebateAgent(tools=self.tools, llm=self.llm),
                "master": MasterAgent(tools=self.tools, llm=self.llm),
                "news": NewsAgent(tools=self.tools, llm=self.llm),
                "sentiment": ProfessionalSentimentAgent(tools=self.tools, llm=self.llm),
            }
            
            for name, agent in core_agents.items():
                self.register(name, agent)
                
            logger.info(f"✅ Successfully registered: {list(core_agents.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register agents: {e}")
            # Don't raise, allow graceful degradation

    # --------------------------------------------------------------
    # REGISTER ANY AGENT
    # --------------------------------------------------------------
    def register(self, name: str, agent: BaseAgent):
        if not isinstance(name, str):
            raise ValueError("Agent name must be a string")

        if not isinstance(agent, BaseAgent):
            raise TypeError(f"Agent '{name}' must inherit BaseAgent")

        self.agents[name] = agent
        logger.info(f"📋 Registered agent: {name}")

    # --------------------------------------------------------------
    # RUN ONE AGENT
    # --------------------------------------------------------------
    def run(self, name: str, user_input: Dict[str, Any]):
        """
        Run one agent and return its output.
        Inject tools + LLM automatically.
        """
        agent = self.agents.get(name)
        if not agent:
            
            logger.error(f"Unknown agent: {name}. Available: {list(self.agents.keys())}")
            raise RuntimeError(f"Unknown agent: {name}")
            
        logger.info(f"🚀 Running agent: {name}")

        # Inject dependencies if missing
        agent.tools = getattr(agent, "tools", None) or self.tools
        agent.llm = getattr(agent, "llm", None) or self.llm

        # Execute safely with timing
        start_time = time.time()
        try:
            result = agent.run(user_input)
            elapsed = time.time() - start_time
            logger.info(f"✅ Agent '{name}' completed in {elapsed:.2f}s")
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Agent '{name}' failed after {elapsed:.2f}s: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "agent": name,
                "error": str(e)
            }

    # --------------------------------------------------------------
    # RUN ALL AGENTS (optional but recommended)
    # --------------------------------------------------------------
    def run_all(self, user_input: Dict[str, Any]):
        """
        Runs EVERY registered agent.
        Returns a dict: { agent_name: result }
        """
        outputs = {}
        logger.info(f"Running all agents: {list(self.agents.keys())}")

        for name, agent in self.agents.items():
            try:
                outputs[name] = self.run(name, user_input)
            except Exception as e:
                logger.error(f"Agent '{name}' crashed inside run_all: {e}")
                outputs[name] = {
                    "status": "ERROR",
                    "agent": name,
                    "error": str(e)
                }
    
        return outputs