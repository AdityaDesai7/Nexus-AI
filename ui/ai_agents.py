
# import os
# import logging
# import time
# from datetime import datetime, timedelta
# import streamlit as st

# from main import get_trading_system, fetch_and_enhance
# from agents.technical_agent import TechnicalAnalysisAgent
# from agents.sentiment_agent import SentimentAnalysisAgent
# from agents.risk_agent import RiskManagementAgent
# from agents.portfolio_agent import PortfolioManagerAgent
# from agents.master_agent import MasterAgent
# from agents.debate_agent import DebateAgent
# from agents.data_collection_agent import DataCollectionAgent

# # Legacy fetcher (keep for compatibility); provide a safe fallback if missing
# try:
#     from data.news_fetcher import NewsFetcher
# except Exception:
#     class NewsFetcher:
#         def get_stock_news(self, ticker, company_name, max_results=10):
#             return []

# # New News imports (make sure these files exist)
# from agents.new_news_agent import NewNews
# from data.new_news_fetcher import NewNewsFetcher

# logger = logging.getLogger(__name__)


# def show_ai_agents_page():
#     """Display AI agents analysis page"""
#     if not st.session_state.get('logged_in'):
#         st.error("⚠️ Please login first")
#         if st.button("Go to Login"):
#             st.session_state.page = 'login'
#             st.rerun()
#         return
    
#     ticker = st.session_state.get('ticker', 'RELIANCE.NS')
    
#     # Header
#     col1, col2, col3 = st.columns([1, 6, 1])
#     with col1:
#         if st.button("← Back", use_container_width=True):
#             st.session_state.page = 'analysis'
#             st.rerun()
#     with col2:
#         st.markdown("<h1 style='text-align:center; color:#00b894;'>🤖 Multi-Agent Analysis</h1>", unsafe_allow_html=True)
#         st.markdown(f"<p style='text-align:center;'>{ticker}</p>", unsafe_allow_html=True)
    
#     # Initialize agents
#     if 'agents_initialized' not in st.session_state:
#         st.session_state.tech_agent = TechnicalAnalysisAgent()
#         st.session_state.sentiment_agent = SentimentAnalysisAgent()
#         st.session_state.risk_agent = RiskManagementAgent()
#         st.session_state.portfolio_agent = PortfolioManagerAgent()
#         st.session_state.master_agent = MasterAgent()
#         st.session_state.debate_agent = DebateAgent()
#         st.session_state.data_collector = DataCollectionAgent()
#         st.session_state.news_fetcher = NewsFetcher()
#         # New News components
#         st.session_state.new_news_fetcher = NewNewsFetcher()
#         st.session_state.new_news_agent = NewNews(groq_api_key=os.getenv("GROQ_API_KEY"))
#         st.session_state.agents_initialized = True
    
#     # Date range
#     st.markdown("---")
#     col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])
#     with col1:
#         start_date_input = st.date_input("Start Date", value=datetime.now() - timedelta(days=180), key="start_date_key")
#         start_date = datetime.combine(start_date_input, datetime.min.time())
#     with col2:
#         end_date_input = st.date_input("End Date", value=datetime.now(), key="end_date_key")
#         end_date = datetime.combine(end_date_input, datetime.max.time())
#     with col3:
#         show_debate = st.checkbox("Show Debate", value=True)
#     with col4:
#         run_analysis = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
#     # Main analysis
#     if run_analysis:
#         progress_bar = st.progress(0)
#         status_container = st.empty()
        
#         try:
#             # Step 1: Collect Data
#             status_container.markdown("""
#             <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #00b894;'>
#                 <strong style='color:#00b894;'>📊 Collecting Data...</strong>
#             </div>
#             """, unsafe_allow_html=True)
#             progress_bar.progress(10)
            
#             data_collector = st.session_state.data_collector
#             collection = data_collector.collect(
#                 ticker=ticker,
#                 start_date=start_date,
#                 end_date=end_date,
#                 include_news=True,
#                 include_fundamentals=True
#             )
            
#             tech_data = collection.get('technical_data', {})
#             news_data = collection.get('news_data', [])
            
#             if tech_data.get('status') != 'SUCCESS':
#                 st.error(f"❌ Failed to fetch data: {tech_data.get('error', 'Unknown error')}")
#                 return
            
#             price_df = tech_data.get('dataframe')
#             latest_close = tech_data.get('latest_close', 0)

#             # Step 1.1: Fetch New News articles (Tavily + NewsAPI)
#             status_container.markdown("""
#             <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #0984e3;'>
#                 <strong style='color:#74b9ff;'>🗞️ Fetching New News...</strong>
#             </div>
#             """, unsafe_allow_html=True)
#             progress_bar.progress(20)

#             new_news_fetcher = st.session_state.new_news_fetcher
#             new_news_articles = new_news_fetcher.get_news(
#                 company_name=ticker,  # replace with actual company name if available
#                 ticker=ticker,
#                 max_results=16,
#                 days=7,
#                 dedupe=True
#             )
#             st.session_state['new_news_articles'] = new_news_articles  # persist

#             # Step 2: Technical Analysis
#             status_container.markdown("""
#             <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #00b894;'>
#                 <strong style='color:#00b894;'>📈 Technical Agent...</strong>
#             </div>
#             """, unsafe_allow_html=True)
#             progress_bar.progress(30)
#             time.sleep(0.3)
            
#             tech_agent = st.session_state.tech_agent
#             tech_result = tech_agent.analyze(ticker, start_date, end_date)
#             tech_action = "BUY" if "BUY" in tech_result.recommendation else ("SELL" if "SELL" in tech_result.recommendation else "HOLD")
#             tech_confidence = tech_result.overall_confidence if hasattr(tech_result, 'overall_confidence') else 50
            
#             # Step 3: Sentiment Analysis (legacy)
#             status_container.markdown("""
#             <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #6c5ce7;'>
#                 <strong style='color:#6c5ce7;'>📰 Sentiment Agent...</strong>
#             </div>
#             """, unsafe_allow_html=True)
#             progress_bar.progress(45)
#             time.sleep(0.3)
            
#             sentiment_agent = st.session_state.sentiment_agent
#             if not news_data:
#                 news_fetcher = st.session_state.news_fetcher
#                 news_data = news_fetcher.get_stock_news(ticker, ticker, max_results=10)
#             sentiment_result = sentiment_agent.analyze(ticker=ticker, news_data=news_data if news_data else [])
#             sent_action = sentiment_result.get('overall_sentiment', 'neutral')
#             sent_confidence = sentiment_result.get('overall_confidence', 50)

#             # Step 3.1: New News Agent (deterministic + Groq-ready)
#             status_container.markdown("""
#             <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #00cec9;'>
#                 <strong style='color:#00cec9;'>🤖 New News Agent...</strong>
#             </div>
#             """, unsafe_allow_html=True)
#             progress_bar.progress(55)
#             time.sleep(0.2)

#             new_news_agent = st.session_state.new_news_agent
#             new_news_agent.set_context(ticker=ticker, company_name=ticker, news_data=new_news_articles or [])
#             new_news_result = new_news_agent.analyze(ticker=ticker, news_data=new_news_articles or [])
#             st.session_state['new_news_result'] = new_news_result

#             # Step 4: Risk Assessment
#             status_container.markdown("""
#             <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #e17055;'>
#                 <strong style='color:#e17055;'>⚠️ Risk Agent...</strong>
#             </div>
#             """, unsafe_allow_html=True)
#             progress_bar.progress(60)
#             time.sleep(0.3)
            
#             risk_agent = st.session_state.risk_agent
#             risk_metrics = risk_agent.evaluate(
#                 ticker=ticker,
#                 df=price_df,
#                 current_price=latest_close,
#                 technical_confidence=tech_confidence,
#                 sentiment_confidence=sent_confidence
#             )
#             risk_dict = risk_metrics.dict() if hasattr(risk_metrics, 'dict') else dict(risk_metrics) if not isinstance(risk_metrics, dict) else risk_metrics
#             risk_level = risk_dict.get('risk_level', 'MEDIUM')
            
#             # Step 5: Portfolio Allocation
#             status_container.markdown("""
#             <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #00cec9;'>
#                 <strong style='color:#00cec9;'>💼 Portfolio Agent...</strong>
#             </div>
#             """, unsafe_allow_html=True)
#             progress_bar.progress(75)
#             time.sleep(0.3)
            
#             portfolio_agent = st.session_state.portfolio_agent
#             portfolio_action, portfolio_qty, portfolio_meta = portfolio_agent.decide(
#                 ticker=ticker,
#                 current_price=latest_close,
#                 technical_signal={'recommendation': tech_result.recommendation, 'action': tech_action},
#                 sentiment_signal=sentiment_result,
#                 risk_metrics=risk_dict
#             )
            
#             # Step 6: Debate Analysis
#             debate_result = None
#             if show_debate:
#                 status_container.markdown("""
#                 <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #fd79a8;'>
#                     <strong style='color:#fd79a8;'>🐂🐻 Debate Agent...</strong>
#                 </div>
#                 """, unsafe_allow_html=True)
#                 progress_bar.progress(85)
#                 time.sleep(0.3)
                
#                 debate_agent = st.session_state.debate_agent
#                 debate_result = debate_agent.debate(
#                     ticker=ticker,
#                     technical_result=dict(tech_result) if hasattr(tech_result, '__dict__') else tech_result,
#                     risk_metrics=risk_dict,
#                     price_data=price_df,
#                     sentiment_score=sentiment_result.get('overall_score', 50)
#                 )
            
#             # Step 7: Master Decision
#             status_container.markdown("""
#             <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd93d;'>
#                 <strong style='color:#ffd93d;'>🎯 Master Agent...</strong>
#             </div>
#             """, unsafe_allow_html=True)
#             progress_bar.progress(95)
#             time.sleep(0.3)
            
#             master_agent = st.session_state.master_agent
#             master_result = master_agent.synthesize(
#                 ticker=ticker,
#                 technical_result=dict(tech_result) if hasattr(tech_result, '__dict__') else tech_result,
#                 sentiment_result=sentiment_result,
#                 risk_metrics=risk_dict,
#                 portfolio_metrics={'quantity': portfolio_qty},
#                 current_price=latest_close
#             )
#             master_dict = master_result.dict() if hasattr(master_result, 'dict') else dict(master_result) if not isinstance(master_result, dict) else master_result
#             master_action = master_dict.get('action', 'HOLD')
#             master_conf = master_dict.get('confidence', 50)
            
#             progress_bar.progress(100)
#             time.sleep(0.2)
#             progress_bar.empty()
#             status_container.empty()
            
#             st.success("✅ Analysis Complete!")
            
#             # Display results
#             tab_all, tab_master, tab_debate, tab_details, tab_newnews = st.tabs(
#                 ["🤖 All Agents", "🎯 Master Decision", "🐂🐻 Debate", "📊 Details", "🗞️ New News"]
#             )
            
#             # TAB 1: ALL AGENTS
#             with tab_all:
#                 st.markdown("## Individual Agent Recommendations")
#                 col1, col2 = st.columns(2)
                
#                 # Technical
#                 with col1:
#                     action_color = "#00b894" if "BUY" in tech_action else "#ff7675" if "SELL" in tech_action else "#fdcb6e"
#                     st.markdown(f"""
#                     <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {action_color};'>
#                         <h3 style='color:{action_color}; margin-top:0;'>📊 Technical Agent</h3>
#                         <p><strong>Signal:</strong> {tech_action}</p>
#                         <p><strong>Confidence:</strong> {tech_confidence:.1f}%</p>
#                         <hr style='margin: 10px 0; opacity: 0.3;'>
#                         <p><strong>Key Indicators:</strong></p>
#                         <ul style='margin: 10px 0; padding-left: 20px;'>
#                             <li>MACD: {tech_result.macd:.4f}</li>
#                             <li>RSI: {tech_result.rsi:.2f}</li>
#                             <li>BB Upper: {tech_result.bollinger_upper:.2f}</li>
#                         </ul>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 # Sentiment (legacy)
#                 with col2:
#                     sent_color = "#00b894" if sent_action == "positive" else "#ff7675" if sent_action == "negative" else "#fdcb6e"
#                     st.markdown(f"""
#                     <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {sent_color};'>
#                         <h3 style='color:{sent_color}; margin-top:0;'>📰 Sentiment Agent</h3>
#                         <p><strong>Sentiment:</strong> {sent_action.upper()}</p>
#                         <p><strong>Confidence:</strong> {sent_confidence:.1f}%</p>
#                         <p><strong>Summary:</strong></p>
#                         <p style='font-size: 0.9em; color: #b2bec3;'>{sentiment_result.get('summary', 'No summary')}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 col3, col4 = st.columns(2)
                
#                 # Risk
#                 with col3:
#                     risk_color = "#00b894" if risk_level == "LOW" else "#fdcb6e" if risk_level == "MEDIUM" else "#ff7675"
#                     st.markdown(f"""
#                     <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {risk_color};'>
#                         <h3 style='color:{risk_color}; margin-top:0;'>⚠️ Risk Agent</h3>
#                         <p><strong>Risk Level:</strong> {risk_level}</p>
#                         <p><strong>Volatility:</strong> {risk_dict.get('volatility', 0):.2%}</p>
#                         <p><strong>Position Size:</strong> {risk_dict.get('position_size', 0):.2%}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 # Portfolio
#                 with col4:
#                     port_color = "#00b894" if portfolio_action == "BUY" else "#ff7675" if portfolio_action == "SELL" else "#fdcb6e"
#                     st.markdown(f"""
#                     <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {port_color};'>
#                         <h3 style='color:{port_color}; margin-top:0;'>💼 Portfolio Agent</h3>
#                         <p><strong>Action:</strong> {portfolio_action}</p>
#                         <p><strong>Quantity:</strong> {portfolio_qty} shares</p>
#                         <p><strong>Allocation:</strong> ₹{portfolio_qty * latest_close:,.0f}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             # TAB 2: MASTER DECISION
#             with tab_master:
#                 st.markdown("## 🎯 Master Agent Final Decision")
#                 master_color = "#00b894" if master_action == "BUY" else "#ff7675" if master_action == "SELL" else "#fdcb6e"
#                 col1, col2, col3 = st.columns([2, 1, 1])
#                 with col1:
#                     st.markdown(f"""
#                     <div style='background: linear-gradient(135deg, {master_color}22, {master_color}11); padding: 40px; border-radius: 16px; border: 3px solid {master_color};'>
#                         <h1 style='color:{master_color}; margin:0; font-size:3rem; text-align:center;'>{master_action}</h1>
#                         <p style='color:#b2bec3; text-align:center; margin:10px 0 0 0; font-size:1.2rem;'>Master Recommendation</p>
#                     </div>
#                     """, unsafe_allow_html=True)
#                 with col2:
#                     st.metric("Confidence", f"{master_conf:.1f}%")
#                     st.metric("Quantity", f"{portfolio_qty}")
#                 with col3:
#                     st.metric("Risk Level", risk_level)
#                     st.metric("Entry Price", f"₹{latest_close:.2f}")
#                 st.markdown("---")
#                 st.markdown("### 📝 Master Agent Reasoning")
#                 st.info(master_dict.get('reasoning', 'No reasoning'))
            
#             # TAB 3: DEBATE
#             if show_debate and debate_result:
#                 with tab_debate:
#                     st.markdown("## 🐂 vs 🐻 Bull vs Bear Debate")
#                     debate_dict = debate_result.dict() if hasattr(debate_result, 'dict') else dict(debate_result) if not isinstance(debate_result, dict) else debate_result
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         bull = debate_dict.get('bull_case', {})
#                         st.markdown(f"""
#                         <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid #00b894;'>
#                             <h3 style='color:#00b894; margin-top:0;'>🐂 BULL CASE</h3>
#                             <p><strong>Strength:</strong> {bull.get('strength', 0):.0f}/100</p>
#                             <p><strong>Arguments:</strong></p>
#                             <ul style='margin: 10px 0; padding-left: 20px;'>
#                         """
#                         + "".join([f"<li>{arg}</li>" for arg in bull.get('arguments', [])[:5]])
#                         + """
#                             </ul>
#                         </div>
#                         """, unsafe_allow_html=True)
#                     with col2:
#                         bear = debate_dict.get('bear_case', {})
#                         st.markdown(f"""
#                         <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid #ff7675;'>
#                             <h3 style='color:#ff7675; margin-top:0;'>🐻 BEAR CASE</h3>
#                             <p><strong>Strength:</strong> {bear.get('strength', 0):.0f}/100</p>
#                             <p><strong>Arguments:</strong></p>
#                             <ul style='margin: 10px 0; padding-left: 20px;'>
#                         """
#                         + "".join([f"<li>{arg}</li>" for arg in bear.get('arguments', [])[:5]])
#                         + """
#                             </ul>
#                         </div>
#                         """, unsafe_allow_html=True)
            
#             # TAB 4: DETAILS
#             with tab_details:
#                 st.markdown("## 📊 Detailed Data")
#                 subtab1, subtab2, subtab3 = st.tabs(["Technical", "Sentiment", "Risk"])
#                 with subtab1:
#                     st.json(dict(tech_result) if hasattr(tech_result, '__dict__') else tech_result)
#                 with subtab2:
#                     st.json(sentiment_result)
#                 with subtab3:
#                     st.json(risk_dict)

#             # TAB 5: NEW NEWS (Groq + Chat)
#             with tab_newnews:
#                 st.markdown("## 🗞️ New News (Tavily + NewsAPI + Groq)")
                
#                 new_news_result = st.session_state.get('new_news_result', {})
#                 new_news_articles = st.session_state.get('new_news_articles', [])

#                 # Cards: Deterministic sentiment
#                 nn_sent = new_news_result.get('overall_sentiment', 'neutral')
#                 nn_conf = new_news_result.get('overall_confidence', 0)
#                 nn_score = new_news_result.get('overall_score', 0.0)
#                 nn_color = "#00b894" if nn_sent == "positive" else "#ff7675" if nn_sent == "negative" else "#fdcb6e"

#                 colA, colB = st.columns([2, 3])
#                 with colA:
#                     st.markdown(f"""
#                     <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {nn_color};'>
#                         <h3 style='color:{nn_color}; margin-top:0;'>🤖 New News Sentiment</h3>
#                         <p><strong>Sentiment:</strong> {nn_sent.upper()}</p>
#                         <p><strong>Score:</strong> {nn_score:.3f}</p>
#                         <p><strong>Confidence:</strong> {nn_conf:.1f}%</p>
#                         <p><strong>Summary:</strong></p>
#                         <p style='font-size: 0.9em; color: #b2bec3;'>{new_news_result.get('summary', 'No summary')}</p>
#                     </div>
#                     """, unsafe_allow_html=True)

#                 with colB:
#                     st.markdown("""
#                     <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid #0984e3;'>
#                         <h3 style='color:#74b9ff; margin-top:0;'>📰 Latest Articles</h3>
#                     </div>
#                     """, unsafe_allow_html=True)
#                     if new_news_articles:
#                         for i, a in enumerate(new_news_articles[:10], start=1):
#                             title = a.get("title", "N/A")
#                             src = a.get("source", "Unknown")
#                             date = a.get("publishedAt", "Unknown")
#                             url = a.get("url", "")
#                             st.markdown(f"- [{title}]({url}) — {src} • {date}")
#                     else:
#                         st.warning("No articles found for New News.")

#                 st.markdown("---")

#                 # Groq Executive Summary
#                 groq_ready = getattr(st.session_state.new_news_agent, "client", None) is not None
#                 col_groq1, col_groq2 = st.columns([1, 3])
#                 with col_groq1:
#                     gen_summary = st.button("⚡ Generate Groq Executive Summary", use_container_width=True)
#                 with col_groq2:
#                     if not groq_ready:
#                         st.info("Set GROQ_API_KEY to enable AI summary and chat.")
#                 if gen_summary and groq_ready:
#                     with st.spinner("Analyzing with Groq..."):
#                         summary = st.session_state.new_news_agent.groq_summary(top_k=10)
#                         st.session_state['new_news_groq_summary'] = summary or "No AI summary available."
#                 summary_text = st.session_state.get('new_news_groq_summary')
#                 if summary_text:
#                     st.markdown("### 🧠 AI Executive Summary")
#                     st.markdown(summary_text)

#                 st.markdown("---")

#                 # Grounded Chatbot
#                 st.markdown("### 💬 New News Chatbot (Grounded with Citations)")
#                 chat_q = st.text_input("Ask about catalysts, risks, revenue impact, etc.", key=f"nn_chat_{ticker}")
#                 colAsk1, colAsk2 = st.columns([1, 3])
#                 with colAsk1:
#                     ask = st.button("Ask", key=f"nn_chat_btn_{ticker}", use_container_width=True)
#                 with colAsk2:
#                     if not groq_ready:
#                         st.info("Groq not configured.")
#                 if ask and chat_q and groq_ready:
#                     with st.spinner("Thinking..."):
#                         resp = st.session_state.new_news_agent.chat(chat_q, max_articles=12, temperature=0.2, return_citations=True)
#                         st.session_state['nn_last_answer'] = resp

#                 resp = st.session_state.get('nn_last_answer')
#                 if resp:
#                     st.markdown("#### Answer")
#                     st.markdown(resp.get("answer", ""))
#                     cites = resp.get("citations", [])
#                     if cites:
#                         st.markdown("#### Citations")
#                         for c in cites:
#                             st.markdown(f"- [{c['index']}] [{c['title']}]({c['url']}) — {c.get('source','Unknown')} • {c.get('publishedAt','Unknown')}")
            
#             # Trade execution
#             st.markdown("---")
#             st.markdown("## 💼 Trade Execution")
#             if master_action != "HOLD":
#                 col1, col2 = st.columns([2, 1])
#                 with col1:
#                     trade_qty = st.number_input("Quantity", min_value=1, value=portfolio_qty, step=1, key=f"qty_{ticker}")
#                     st.info(f"💰 Value: ₹{trade_qty * latest_close:,.2f}")
#                 with col2:
#                     st.markdown("<br>", unsafe_allow_html=True)
#                     if st.button(f"✅ Execute {master_action}", type="primary", use_container_width=True, key=f"exe_{ticker}"):
#                         if not st.session_state.get('user_id'):
#                             st.error("❌ Not logged in!")
#                             st.stop()
#                         try:
#                             from models.database import get_database
#                             db = get_database()
#                             agent_rec = f"{master_action} - Tech: {tech_action} | Sentiment: {sent_action} | Risk: {risk_level}"
#                             with st.spinner("Executing..."):
#                                 success, message = db.execute_trade(
#                                     user_id=st.session_state.user_id,
#                                     ticker=ticker,
#                                     action=master_action,
#                                     quantity=trade_qty,
#                                     price=latest_close,
#                                     agent_rec=agent_rec
#                                 )
#                             if success:
#                                 st.balloons()
#                                 st.success(f"✅ {message}")
#                                 time.sleep(2)
#                                 st.session_state.page = 'portfolio'
#                                 st.rerun()
#                             else:
#                                 st.error(f"❌ {message}")
#                         except Exception as e:
#                             st.error(f"❌ Error: {str(e)}")
#                             logger.error(f"Trade execution error: {e}")
#             else:
#                 st.warning("⏸️ Master recommends HOLD")
        
#         except Exception as e:
#             st.error(f"❌ Error: {str(e)}")
#             logger.error(f"Analysis error: {e}")
#             import traceback
#             with st.expander("🔍 Debug Info"):
#                 st.code(traceback.format_exc())
#     else:
#         st.info("👆 Select dates and click Analyze")


# trading_bot/ui/ai_agents.py
"""
Streamlit page to run your SMART-MARKET multi-agent pipeline.

Requirements (already assumed present in repo):
 - trading_bot.agent_runner.AgentRunner
 - trading_bot.agents.wrappers.create_wrapped_agents
 - trading_bot.tools.toolbox.TOOLS (with fetch_price, fetch_news)
 - trading_bot.llm.llm_wrapper.LLM (optional; DummyLLM fallback allowed)
 - Your original logic agents in trading_bot.agents.*

Notes:
 - All numeric/market data is fetched via tools (fetch_price) for auditability.
 - LLM is used only for text summaries; numbers from LLM are never trusted.
"""

# import os
# import logging
# import time
# from datetime import datetime, timedelta
# from typing import Any, Dict

# import streamlit as st

# # ---------- imports from your agent engine ----------
# from trading_bot.agent_runner import AgentRunner
# from trading_bot.agents.wrappers import create_wrapped_agents
# from trading_bot.tools.toolbox import TOOLS
# from trading_bot.llm.llm_wrapper import LLM

# logger = logging.getLogger("ai_agents")
# logger.setLevel(logging.INFO)


# # -------------------------
# # Helper utilities
# # -------------------------
# def ensure_session_runner():
#     """
#     Initialize AgentRunner + wrapped agents in session_state.
#     Keeps a persistent runner across Streamlit reruns.
#     """
#     if "agent_runner" not in st.session_state:
#         runner = AgentRunner()
#         # instantiate wrapped agents (they will receive tools and llm on runner.run)
#         wrapped = create_wrapped_agents(tools=TOOLS, llm=LLM)
#         for name, agent in wrapped.items():
#             runner.register(name, agent)
#         st.session_state.agent_runner = runner
#         st.session_state.wrapped_agents = list(wrapped.keys())
#         logger.info("AgentRunner initialized with agents: %s", st.session_state.wrapped_agents)
#     return st.session_state.agent_runner


# def safe_get_latest_close_from_technical(tech_out: Dict[str, Any]) -> float:
#     """Best-effort to extract latest price / close from technical output or fallback to 0."""
#     try:
#         tech = tech_out.get("technical", {})
#         # Common keys used in your technical outputs
#         for k in ("latest_close", "close", "price", "support"):
#             if k in tech:
#                 return float(tech.get(k) or 0.0)
#         # fallback: check for nested object attributes
#         if isinstance(tech, dict):
#             for v in tech.values():
#                 try:
#                     return float(v)
#                 except Exception:
#                     continue
#     except Exception:
#         pass
#     return 0.0


# # -------------------------
# # Streamlit UI
# # -------------------------
# def show_ai_agents_page():
#     st.set_page_config(layout="wide", page_title="SMART-MARKET — Multi-Agent Analysis")
#     st.title("🤖 SMART-MARKET — Multi-Agent Analysis")

#     # Quick checks
#     runner = ensure_session_runner()

#     groq_key = os.getenv("GROQ_API_KEY", "").strip()
#     groq_ready = bool(groq_key) and not isinstance(LLM, type(None)) and not hasattr(LLM, "ask") == False

#     # Sidebar - controls
#     with st.sidebar:
#         st.header("Run settings")
#         ticker = st.text_input("Ticker (e.g. TCS.NS or AAPL)", value=st.session_state.get("ticker", "RELIANCE.NS"))
#         start_date_input = st.date_input("Start date", value=datetime.now() - timedelta(days=180))
#         end_date_input = st.date_input("End date", value=datetime.now())
#         show_debate = st.checkbox("Show Debate", value=True)
#         run_btn = st.button("🚀 Analyze")
#         st.markdown("---")
#         st.markdown("Agent status:")
#         st.write(", ".join(st.session_state.wrapped_agents))
#         st.markdown("---")
#         if not groq_ready:
#             st.warning("Groq LLM not configured. Set GROQ_API_KEY to enable Groq features.")
#         else:
#             st.success("Groq ready ✅")

#     # persist ticker & dates
#     st.session_state["ticker"] = ticker
#     start_date = datetime.combine(start_date_input, datetime.min.time())
#     end_date = datetime.combine(end_date_input, datetime.max.time())

#     # Display area
#     if run_btn:
#         progress = st.progress(0)
#         status = st.empty()

#         try:
#             status.info("1/7 — Collecting data (tools)")
#             progress.progress(5)
#             # Use the Data Collection tool via the agents/tools - prefer calling news via tools
#             # Step A: Technical Agent run
#             status.info("2/7 — Running Technical Agent")
#             tech_resp = runner.run("technical", {"ticker": ticker, "start": start_date, "end": end_date})
#             progress.progress(25)
#             tech_result = tech_resp.get("result", {})
#             st.session_state.last_tech = tech_result

#             # Extract a safe latest_close (best-effort)
#             latest_close = safe_get_latest_close_from_technical(tech_result) or 0.0

#             status.info("3/7 — Running News Agent")
#             news_resp = runner.run("news", {"ticker": ticker})
#             progress.progress(40)
#             news_result = news_resp.get("result", {})

#             status.info("4/7 — Running Risk Agent")
#             # Provide technical confidence if available
#             tech_confidence = 50
#             try:
#                 tech_rec = tech_result.get("technical", {})
#                 # many technical outputs include 'confidence' or 'overall_confidence'
#                 tech_confidence = float(tech_rec.get("confidence", tech_rec.get("overall_confidence", 50)))
#             except Exception:
#                 tech_confidence = 50
#             risk_resp = runner.run("risk", {
#                 "ticker": ticker,
#                 "start": start_date,
#                 "end": end_date,
#                 "technical_confidence": tech_confidence
#             })
#             progress.progress(60)
#             risk_result = risk_resp.get("result", {}).get("risk", risk_resp.get("result", {}))

#             status.info("5/7 — Running Portfolio Agent")
#             port_resp = runner.run("portfolio", {
#                 "ticker": ticker,
#                 "start": start_date,
#                 "end": end_date,
#                 "technical_signal": {"recommendation": tech_result.get("technical")},
#                 "risk_metrics": risk_result
#             })
#             progress.progress(75)
#             port_result = port_resp.get("result", {})

#             # Debate (optional)
#             debate_result = None
#             if show_debate:
#                 status.info("6/7 — Running Debate Agent")
#                 debate_resp = runner.run("debate", {
#                     "ticker": ticker,
#                     "technical_result": tech_result.get("technical") if isinstance(tech_result, dict) else {},
#                     "risk_metrics": risk_result,
#                     "start": start_date,
#                     "end": end_date
#                 })
#                 progress.progress(88)
#                 debate_result = debate_resp.get("result", {})

#             # Master Agent
#             status.info("7/7 — Running Master Agent")
#             master_resp = runner.run("master", {
#                 "ticker": ticker,
#                 "technical_result": tech_result.get("technical") if isinstance(tech_result, dict) else {},
#                 "sentiment_result": None,
#                 "risk_metrics": risk_result,
#                 "portfolio_metrics": port_result.get("meta", {}),
#                 "current_price": latest_close,
#             })
#             progress.progress(100)
#             master_result = master_resp.get("result", {}).get("master", master_resp.get("result", {}))

#             # clear status / progress
#             time.sleep(0.25)
#             status.empty()
#             progress.empty()

#             st.success("✅ Analysis Complete")

#             # ---------- UI: Tabs to display structured outputs ----------
#             tab_all, tab_master, tab_debate, tab_details, tab_news = st.tabs(
#                 ["All Agents", "Master Decision", "Debate", "Details", "News"]
#             )

#             with tab_all:
#                 st.header("Individual Agent Outputs")
#                 c1, c2 = st.columns(2)

#                 with c1:
#                     st.subheader("Technical Agent")
#                     st.json(tech_result)
#                 with c2:
#                     st.subheader("Portfolio Agent")
#                     st.json(port_result)

#                 c3, c4 = st.columns(2)
#                 with c3:
#                     st.subheader("Risk Agent")
#                     st.json(risk_result)
#                 with c4:
#                     st.subheader("News Agent (top summaries)")
#                     # show short summaries
#                     summaries = news_result.get("summaries") if isinstance(news_result, dict) else None
#                     if summaries:
#                         for s in summaries[:6]:
#                             st.markdown(f"- **{s.get('title','-')}** — {s.get('source','')}")
#                     else:
#                         st.info("No news summaries available")

#             with tab_master:
#                 st.header("Master Agent Decision")
#                 st.metric("Action", master_result.get("action", "HOLD"))
#                 st.metric("Confidence", f"{master_result.get('confidence', 50)}%")
#                 st.write("Reasoning:")
#                 st.info(master_result.get("reasoning", "No reasoning provided"))
#                 st.markdown("---")
#                 st.write(master_result)

#             with tab_debate:
#                 st.header("Debate Agent")
#                 if debate_result:
#                     st.json(debate_result)
#                 else:
#                     st.info("Debate was disabled or returned no content.")

#             with tab_details:
#                 st.header("Traces & Tool Calls (Audit)")
#                 st.markdown("### Technical agent trace")
#                 tech_trace = tech_resp.get("trace", [])
#                 st.write(tech_trace)
#                 st.markdown("### Risk agent trace")
#                 st.write(risk_resp.get("trace", []))
#                 st.markdown("### Portfolio agent trace")
#                 st.write(port_resp.get("trace", []))
#                 st.markdown("### Master agent trace")
#                 st.write(master_resp.get("trace", []))

#             with tab_news:
#                 st.header("News — Full articles & AI summary")
#                 # raw articles
#                 articles = news_result.get("articles", [])
#                 if articles:
#                     for a in articles[:10]:
#                         title = a.get("title", "No title")
#                         src = a.get("source", "Unknown")
#                         url = a.get("url", "")
#                         st.markdown(f"- [{title}]({url}) — {src}")
#                 else:
#                     st.info("No news articles fetched.")

#                 # Groq summary if present
#                 groq_summary = news_result.get("groq")
#                 if groq_summary:
#                     st.markdown("### Groq AI Summary (grounded)")
#                     st.write(groq_summary)

#             # Offer quick trade execution panel (disabled if master is HOLD)
#             st.markdown("---")
#             st.header("Trade Execution (manual)")
#             master_action = master_result.get("action", "HOLD")
#             if master_action != "HOLD":
#                 qty = st.number_input("Quantity", min_value=1, value=int(port_result.get("quantity", 1)))
#                 st.write(f"Estimated value: ₹{qty * latest_close:,.2f}")
#                 if st.button(f"Execute {master_action} (simulate)"):
#                     st.success(f"Simulated {master_action} of {qty} shares at ₹{latest_close:.2f}")
#             else:
#                 st.info("Master recommends HOLD — no trade suggested.")

#         except Exception as e:
#             logger.exception("Analysis pipeline failed")
#             st.error(f"Analysis failed: {str(e)}")
#             # show debug info
#             with st.expander("Debug info"):
#                 import traceback
#                 st.code(traceback.format_exc())
#     else:
#         st.info("Configure ticker & dates in the sidebar and click 'Analyze' to run the multi-agent pipeline.")


# # Run the page directly (if user runs this module)
# if __name__ == "__main__":
#     # allow direct testing outside Streamlit (minimal)
#     print("This module is a Streamlit page. Run with `streamlit run trading_bot/ui/ai_agents.py`")

# # trading_bot/ui/ai_agents.py
# import os
# import time
# from datetime import datetime, timedelta
# import logging
# import streamlit as st
# import pandas as pd
# from data.data_fetcher import fetch_data


# from agent_runner import AgentRunner
# from agents.wrappers import create_wrapped_agents
# from tools.toolbox import TOOLS
# from llm.llm_wrapper import LLM
# from agent_runner import AgentRunner



# # try both factories; prefer standard wrappers, fallback to inst_wrappers
# try:
#     from agents.wrappers import create_wrapped_agents
# except Exception:
#     create_wrapped_agents = None

# try:
#     from agents.inst_wrappers import create_inst_wrappers
# except Exception:
#     create_inst_wrappers = None

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


# # -------------------------
# # Helpers: normalize runner responses
# # -------------------------
# def normalize_response(resp):
#     """
#     Convert various possible agent responses into a plain dict.
#     Supports:
#      - dict (returned as-is)
#      - objects with .dict() (pydantic)
#      - objects with __dict__
#      - pandas DataFrame -> {'status':'OK','df': df}
#      - string -> {'status':'OK','text': str}
#      - None -> {'status':'ERROR', 'error': 'No response'}
#     """
#     try:
#         if resp is None:
#             return {"status": "ERROR", "error": "No response (None)"}

#         # If runner returns a wrapper dict like {"result": {...}} or {"technical": {...}}
#         if isinstance(resp, dict):
#             # make a shallow copy to avoid mutating original
#             return dict(resp)

#         # pandas DataFrame
#         if isinstance(resp, pd.DataFrame):
#             return {"status": "OK", "df": resp}

#         # pydantic or similar
#         if hasattr(resp, "dict") and callable(getattr(resp, "dict")):
#             try:
#                 return {"status": "OK", **resp.dict()}
#             except Exception:
#                 return {"status": "OK", **(vars(resp) if hasattr(resp, "__dict__") else {"value": str(resp)})}

#         # plain object with __dict__
#         if hasattr(resp, "__dict__"):
#             return {"status": "OK", **vars(resp)}

#         # fallback to string
#         return {"status": "OK", "text": str(resp)}

#     except Exception as e:
#         logger.exception("normalize_response failed")
#         return {"status": "ERROR", "error": f"normalize_response failed: {e}"}


# def extract_payload(norm: dict, preferred_keys=("result", "technical", "master", "risk", "portfolio")):
#     """
#     From a normalized dict, pick the nested payload under one of preferred_keys,
#     else return the normalized dict itself.
#     """
#     if not isinstance(norm, dict):
#         return {"status": "ERROR", "error": "normalize_response did not return dict"}

#     for k in preferred_keys:
#         if k in norm and isinstance(norm[k], dict):
#             return norm[k]
#     # sometimes agents return {'status':'OK', 'df': df}
#     if "df" in norm or "status" in norm:
#         return norm
#     return norm


# def display_json_friendly(label: str, payload, expand: bool = False):
#     """
#     Safely display payload in Streamlit:
#     - dict -> st.json
#     - DataFrame -> st.dataframe
#     - list -> st.write
#     - string -> st.write
#     """
#     st.markdown(f"### {label}")
#     if payload is None:
#         st.info("No output")
#         return

#     if isinstance(payload, dict):
#         # remove DataFrame before st.json
#         df = payload.pop("df", None)
#         try:
#             st.json(payload)
#         except Exception:
#             # fallback: pretty print
#             st.write(payload)
#         if df is not None:
#             if isinstance(df, pd.DataFrame):
#                 st.markdown("**Data (preview)**")
#                 st.dataframe(df.head(10))
#             else:
#                 st.write(df)
#     elif isinstance(payload, pd.DataFrame):
#         st.dataframe(payload.head(10))
#     elif isinstance(payload, list):
#         st.write(payload)
#     else:
#         st.write(str(payload))


# # -------------------------
# # Runner initialization
# # -------------------------
# def ensure_session_runner():
#     """
#     Initialize AgentRunner + wrapped agents in session_state.
#     Robustly supports either create_wrapped_agents or create_inst_wrappers,
#     and factories that accept (tools, llm) or (llm) signature.
#     """
#     if "agent_runner" not in st.session_state:
#         runner = AgentRunner()

#         # Build wrapped agent dict using available factory
#         wrapped = {}
#         try:
#             # prefer normal wrapper factory first
#             if create_wrapped_agents is not None:
#                 try:
#                     wrapped = create_wrapped_agents(tools=TOOLS, llm=LLM())
#                 except TypeError:
#                     wrapped = create_wrapped_agents(LLM())
#             elif create_inst_wrappers is not None:
#                 try:
#                     wrapped = create_inst_wrappers(tools=TOOLS, llm=LLM())
#                 except TypeError:
#                     wrapped = create_inst_wrappers(LLM())
#             else:
#                 logger.warning("No agent factory function available; starting empty AgentRunner.")
#                 wrapped = {}

#             # register returned agents
#             for name, agent in (wrapped or {}).items():
#                 try:
#                     runner.register(name, agent)
#                 except Exception as e:
#                     logger.exception("Failed to register agent %s: %s", name, e)

#             st.session_state.agent_runner = runner
#             st.session_state.wrapped_agents = list((wrapped or {}).keys())
#             logger.info("AgentRunner initialized with agents: %s", st.session_state.wrapped_agents)

#         except Exception as e:
#             logger.exception("Failed to initialize wrapped agents")
#             st.session_state.agent_runner = runner
#             st.session_state.wrapped_agents = []
#     return st.session_state.agent_runner


# def safe_get_latest_close(payload):
#     """
#     Try multiple common shapes to extract latest close.
#     """
#     try:
#         if payload is None:
#             return 0.0
#         # payload could be {'latest_close': x}
#         if isinstance(payload, dict):
#             for k in ("latest_close", "latest", "close", "price"):
#                 if k in payload and (isinstance(payload[k], (int, float)) or (isinstance(payload[k], str) and payload[k].replace('.', '', 1).isdigit())):
#                     try:
#                         return float(payload[k])
#                     except Exception:
#                         pass
#             # payload may include df
#             if "df" in payload and isinstance(payload["df"], pd.DataFrame):
#                 df = payload["df"]
#                 if "Close" in df.columns and len(df) > 0:
#                     return float(df["Close"].iloc[-1])
#         # if DataFrame
#         if isinstance(payload, pd.DataFrame):
#             df = payload
#             if "Close" in df.columns and len(df) > 0:
#                 return float(df["Close"].iloc[-1])
#     except Exception:
#         pass
#     return 0.0


# # -------------------------
# # Streamlit Page
# # -------------------------
# def show_ai_agents_page():
#     st.set_page_config(layout="wide", page_title="SMART-MARKET — Multi-Agent Analysis")
#     st.title("🤖 SMART-MARKET — Multi-Agent Analysis")

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
#         st.markdown("Agents:")
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
#         status.info("1/6 — Fetch canonical price data (via TOOLS)")
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
#                     # log and continue; some tools return errors
#                     logger.warning("fetch_price returned unexpected shape: %s", type(price_res))
#             else:
#                 logger.warning("fetch_price not available in TOOLS")
#         except Exception as e:
#             logger.exception("TOOLS.fetch_price failed: %s", e)
#             status.error(f"Price fetch failed: {e}")

#         if isinstance(price_df, pd.DataFrame) and len(price_df) > 0:
#             latest_close = float(price_df["Close"].iloc[-1]) if "Close" in price_df.columns else 0.0

#         # TECHNICAL
#         status.info("2/6 — Running Technical Agent")
#         progress.progress(20)
#         tech_raw = runner.run("technical", {"ticker": ticker, "start": start_date, "end": end_date})
#         tech_norm = normalize_response(tech_raw)
#         tech_payload = extract_payload(tech_norm, preferred_keys=("technical", "result"))
#         # If agent returned df but not computed indicators, attach price_df
#         if "df" not in tech_payload and isinstance(price_df, pd.DataFrame):
#             tech_payload["df"] = price_df

#         # ensure we have latest_close from technical if present
#         latest_close = latest_close or safe_get_latest_close(tech_payload)

#         # NEWS
#         status.info("3/6 — Running News Agent")
#         progress.progress(40)
#         news_raw = runner.run("news", {"ticker": ticker, "start": start_date, "end": end_date})
#         news_norm = normalize_response(news_raw)
#         news_payload = extract_payload(news_norm)

#         # RISK
#         status.info("4/6 — Running Risk Agent")
#         progress.progress(55)
#         risk_input = {
#             "ticker": ticker,
#             "start": start_date,
#             "end": end_date,
#             "technical_confidence": tech_payload.get("confidence", 50) if isinstance(tech_payload, dict) else 50,
#             "df": tech_payload.get("df", price_df),
#             "current_price": latest_close
#         }
#         risk_raw = runner.run("risk", risk_input)
#         risk_norm = normalize_response(risk_raw)
#         risk_payload = extract_payload(risk_norm)

#         # PORTFOLIO
#         status.info("5/6 — Running Portfolio Agent")
#         progress.progress(75)
#         port_input = {
#             "ticker": ticker,
#             "current_price": latest_close,
#             "technical_signal": tech_payload if isinstance(tech_payload, dict) else {},
#             "risk_metrics": risk_payload if isinstance(risk_payload, dict) else {},
#             "df": tech_payload.get("df", price_df)
#         }
#         port_raw = runner.run("portfolio", port_input)
#         port_norm = normalize_response(port_raw)
#         port_payload = extract_payload(port_norm)

#         # DEBATE (optional)
#         debate_payload = None
#         if show_debate:
#             status.info("6/6 — Running Debate Agent")
#             progress.progress(85)
#             debate_raw = runner.run("debate", {
#                 "ticker": ticker,
#                 "technical_result": tech_payload,
#                 "risk_metrics": risk_payload,
#                 "df": port_payload.get("df", tech_payload.get("df", price_df))
#             })
#             debate_norm = normalize_response(debate_raw)
#             debate_payload = extract_payload(debate_norm)
#             progress.progress(90)

#         # MASTER
#         status.info("Finalizing — Running Master Agent")
#         master_raw = runner.run("master", {
#             "ticker": ticker,
#             "technical_result": tech_payload,
#             "sentiment_result": news_payload,
#             "risk_metrics": risk_payload,
#             "portfolio_metrics": port_payload,
#             "current_price": latest_close
#         })
#         master_norm = normalize_response(master_raw)
#         master_payload = extract_payload(master_norm, preferred_keys=("master", "result"))

#         progress.progress(100)
#         status.empty()
#         progress.empty()
#         st.success("✅ Analysis Complete")

#         # ---------- DISPLAY ----------
#         tab_all, tab_master, tab_debate, tab_news = st.tabs(["All Agents", "Master", "Debate", "News"])

#         with tab_all:
#             col1, col2 = st.columns(2)
#             with col1:
#                 display_json_friendly("Technical Agent", tech_payload)
#                 st.markdown("---")
#                 display_json_friendly("Risk Agent", risk_payload)
#             with col2:
#                 display_json_friendly("Portfolio Agent", port_payload)
#                 st.markdown("---")
#                 if debate_payload:
#                     display_json_friendly("Debate Agent", debate_payload)

#         with tab_master:
#             if isinstance(master_payload, dict):
#                 action = master_payload.get("action", "HOLD")
#                 confidence = master_payload.get("confidence", 50)
        
#                 st.metric("Action", action)
#                 st.metric("Confidence", f"{confidence}%")
        
#         # Show rule-based reasoning
#                 rb = master_payload.get("reasoning", None)
#                 if rb:
#                     st.markdown("### Rule-based Reasoning")
#                     st.write(rb)
        
#                 # Show LLM reasoning if available
#                 if "llm_reasoning" in master_payload and master_payload["llm_reasoning"]:
#                     st.markdown("### LLM Explanation")
#                     st.write(master_payload["llm_reasoning"])
        
#                 # Show signals (clean)
#                 if "signals" in master_payload:
#                     st.markdown("### Signal Breakdown")
#                     st.json(master_payload["signals"])
#             else:
#                 st.info("No master decision available")
#                 st.write(master_payload)


#         with tab_debate:
#             if debate_payload:
#                 st.json(debate_payload)
#             else:
#                 st.info("No debate output")

#         with tab_news:
#             if isinstance(news_payload, dict):
#                 summaries = news_payload.get("summaries") or news_payload.get("articles") or []
#                 if summaries:
#                     for s in summaries[:10]:
#                         # pretty print if dict-like
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
#                 st.dataframe(df.head(10))
#             else:
#                 st.write(df)
#     elif isinstance(payload, pd.DataFrame):
#         st.dataframe(payload.head(10))
#     elif isinstance(payload, list):
#         st.write(payload)
#     else:
#         st.write(str(payload))


# # -------------------------
# # Runner initialization
# # -------------------------
# def ensure_session_runner():
#     """
#     Initialize AgentRunner + wrapped agents in session_state.
#     Robustly supports either create_wrapped_agents or create_inst_wrappers,
#     and factories that accept (tools, llm) or (llm) signature.
#     """
#     if "agent_runner" not in st.session_state:
#         runner = AgentRunner()
#         wrapped = {}
#         try:
#             # Prefer normal wrapper factory first
#             if create_wrapped_agents is not None:
#                 try:
#                     wrapped = create_wrapped_agents(tools=TOOLS, llm=LLM())
#                 except TypeError:
#                     wrapped = create_wrapped_agents(LLM())
#             elif create_inst_wrappers is not None:
#                 try:
#                     wrapped = create_inst_wrappers(tools=TOOLS, llm=LLM())
#                 except TypeError:
#                     wrapped = create_inst_wrappers(LLM())
#             else:
#                 logger.warning("No agent factory function available; starting empty AgentRunner.")
#                 wrapped = {}

#             # Register returned agents safely
#             for name, agent in (wrapped or {}).items():
#                 try:
#                     runner.register(name, agent)
#                 except Exception as e:
#                     logger.exception("Failed to register agent %s: %s", name, e)

#             st.session_state.agent_runner = runner
#             st.session_state.wrapped_agents = list((wrapped or {}).keys())
#             logger.info("AgentRunner initialized with agents: %s", st.session_state.wrapped_agents)

#         except Exception:
#             logger.exception("Failed to initialize wrapped agents")
#             st.session_state.agent_runner = runner
#             st.session_state.wrapped_agents = []
#     return st.session_state.agent_runner


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


# # -------------------------
# # Streamlit Page
# # -------------------------
# def show_ai_agents_page():
#     st.set_page_config(layout="wide", page_title="SMART-MARKET — Multi-Agent Analysis")
#     st.title("🤖 SMART-MARKET — Multi-Agent Analysis")

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
#         st.markdown("Agents:")
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
#         status.info("1/6 — Fetch canonical price data (via TOOLS)")
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

#         # TECHNICAL
#         status.info("2/6 — Running Technical Agent")
#         progress.progress(20)
#         tech_raw = runner.run("technical", {"ticker": ticker, "start": start_date, "end": end_date})
#         tech_norm = normalize_response(tech_raw)
#         tech_payload = extract_payload(tech_norm, preferred_keys=("technical", "result"))
#         if isinstance(tech_payload, dict) and "df" not in tech_payload and isinstance(price_df, pd.DataFrame):
#             tech_payload["df"] = price_df

#         latest_close = latest_close or safe_get_latest_close(tech_payload)

#         # NEWS
#         status.info("3/6 — Running News Agent")
#         progress.progress(40)
#         news_raw = runner.run("news", {"ticker": ticker, "start": start_date, "end": end_date})
#         news_norm = normalize_response(news_raw)
#         news_payload = extract_payload(news_norm)

#         # RISK
#         status.info("4/6 — Running Risk Agent")
#         progress.progress(55)
#         risk_input = {
#             "ticker": ticker,
#             "start": start_date,
#             "end": end_date,
#             "technical_confidence": tech_payload.get("confidence", 50) if isinstance(tech_payload, dict) else 50,
#             "df": tech_payload.get("df", price_df),
#             "current_price": latest_close
#         }
#         risk_raw = runner.run("risk", risk_input)
#         risk_norm = normalize_response(risk_raw)
#         risk_payload = extract_payload(risk_norm)

#         # PORTFOLIO
#         status.info("5/6 — Running Portfolio Agent")
#         progress.progress(75)
#         port_input = {
#             "ticker": ticker,
#             "current_price": latest_close,
#             "technical_signal": tech_payload if isinstance(tech_payload, dict) else {},
#             "risk_metrics": risk_payload if isinstance(risk_payload, dict) else {},
#             "df": tech_payload.get("df", price_df)
#         }
#         port_raw = runner.run("portfolio", port_input)
#         port_norm = normalize_response(port_raw)
#         port_payload = extract_payload(port_norm)

#         # DEBATE (optional)
#         debate_payload = None
#         if show_debate:
#             status.info("6/6 — Running Debate Agent")
#             progress.progress(85)
#             debate_raw = runner.run("debate", {
#                 "ticker": ticker,
#                 "technical_result": tech_payload,
#                 "risk_metrics": risk_payload,
#                 "df": port_payload.get("df", tech_payload.get("df", price_df))
#             })
#             debate_norm = normalize_response(debate_raw)
#             debate_payload = extract_payload(debate_norm)
#             progress.progress(90)

#         # MASTER
#         status.info("Finalizing — Running Master Agent")
#         master_raw = runner.run("master", {
#             "ticker": ticker,
#             "technical_result": tech_payload,
#             "sentiment_result": debate_payload or news_payload,
#             "risk_metrics": risk_payload,
#             "portfolio_metrics": port_payload,
#             "current_price": latest_close
#         })
#         master_norm = normalize_response(master_raw)
#         # master can be nested under 'master' key or returned flat
#         master_payload = master_norm.get("master") if isinstance(master_norm, dict) and "master" in master_norm else extract_payload(master_norm, preferred_keys=("master", "result"))

#         progress.progress(100)
#         status.empty()
#         progress.empty()
#         st.success("✅ Analysis Complete")

#         # ---------- DISPLAY ----------
#         tab_all, tab_master, tab_debate, tab_news = st.tabs(["All Agents", "Master", "Debate", "News"])

#         with tab_all:
#             col1, col2 = st.columns(2)
#             with col1:
#                 display_json_friendly("Technical Agent", tech_payload)
#                 st.markdown("---")
#                 display_json_friendly("Risk Agent", risk_payload)
#             with col2:
#                 display_json_friendly("Portfolio Agent", port_payload)
#                 st.markdown("---")
#                 if debate_payload:
#                     display_json_friendly("Debate Agent", debate_payload)

#         with tab_master:
#             # Debug raw payload (toggleable) — uncomment to aid troubleshooting
#             # st.write("DEBUG MASTER_RAW:", master_raw)
#             # st.write("DEBUG MASTER_NORM:", master_norm)
#             # st.write("DEBUG MASTER_PAYLOAD:", master_payload)

#             if isinstance(master_payload, dict):
#                 # Multiple fallback locations handled
#                 action = master_payload.get("action") or master_payload.get("recommendation") or master_payload.get("final_action") or "HOLD"
#                 confidence = master_payload.get("confidence") or master_payload.get("final_confidence") or master_payload.get("confidence_pct") or 50

#                 st.metric("Action", action)
#                 st.metric("Confidence", f"{confidence}%")

#                 # Rule-based reasoning
#                 rb = master_payload.get("reasoning") or master_payload.get("explanation") or master_payload.get("reason")
#                 if rb:
#                     st.markdown("### Rule-based Reasoning")
#                     st.write(rb)

#                 # LLM reasoning (many agents call it llm_reasoning or llm_action/llm_confidence)
#                 llm_text = master_payload.get("llm_reasoning") or master_payload.get("llm_explanation") or master_payload.get("explain_llm")
#                 if llm_text:
#                     st.markdown("### LLM Explanation")
#                     st.write(llm_text)

#                 # Clean signal breakdown
#                 signals = master_payload.get("signals") or master_payload.get("signal_breakdown") or {}
#                 if signals:
#                     st.markdown("### Signal Breakdown")
#                     st.json(signals)

#             else:
#                 st.info("No master decision available")
#                 st.write(master_payload)

#         with tab_debate:
#             if debate_payload:
#                 display_json_friendly("Debate Output", debate_payload)
#             else:
#                 st.info("No debate output")

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
#                 st.dataframe(df.head(10))
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
#     Clean, professional display for the new Master Agent
#     """
#     if not isinstance(master_payload, dict):
#         st.error("No master analysis available")
#         return

#     # Extract display data or use main payload
#     display_data = master_payload.get("display_data", {})
#     if not display_data:
#         display_data = {
#             "final_decision": master_payload.get("action", "HOLD"),
#             "confidence": master_payload.get("confidence", 50),
#             "reasoning": master_payload.get("reasoning", "No reasoning provided"),
#             "narrative": master_payload.get("narrative", "No analysis available"),
#             "quantity": master_payload.get("quantity", 0),
#             "stop_loss": master_payload.get("risk_management", {}).get("stop_loss"),
#             "take_profit": master_payload.get("risk_management", {}).get("take_profit"),
#             "current_price": master_payload.get("current_price", 0)
#         }

#     st.markdown("---")
    
#     # Main Decision Card
#     col1, col2, col3 = st.columns(3)
    
#     decision = display_data.get("final_decision", "HOLD")
#     confidence = display_data.get("confidence", 50)
    
#     with col1:
#         if decision == "BUY":
#             st.success(f"## 🟢 {decision}")
#         elif decision == "SELL":
#             st.error(f"## 🔴 {decision}")  
#         else:
#             st.info(f"## ⚪ {decision}")
            
#     with col2:
#         st.metric("Confidence", f"{confidence}%")
        
#     with col3:
#         st.metric("Position Size", display_data.get("quantity", 0))
    
#     # Current Price
#     current_price = display_data.get("current_price")
#     if current_price:
#         st.write(f"**Current Price:** ${current_price:.2f}")
    
#     # Narrative (Main story)
#     st.markdown("### 📋 AI Analysis Summary")
#     narrative = display_data.get("narrative", "No analysis available")
#     st.info(narrative)
    
#     # Risk Management
#     st.markdown("### 🛡️ Risk Management")
#     risk_col1, risk_col2 = st.columns(2)
    
#     with risk_col1:
#         stop_loss = display_data.get("stop_loss")
#         if stop_loss:
#             st.metric("Stop Loss", f"${stop_loss:.2f}")
#         else:
#             st.write("Stop Loss: N/A")
            
#     with risk_col2:
#         take_profit = display_data.get("take_profit") 
#         if take_profit:
#             st.metric("Take Profit", f"${take_profit:.2f}")
#         else:
#             st.write("Take Profit: N/A")
    
#     # Signal Breakdown (expandable)
#     with st.expander("🔍 Signal Analysis Details"):
#         reasoning = display_data.get("reasoning", "No detailed reasoning")
#         st.write(f"**Decision Logic:** {reasoning}")
        
#         # Show signal scores if available
#         signals = master_payload.get("signals", {})
#         if signals:
#             col1, col2 = st.columns(2)
#             with col1:
#                 buy_score = signals.get("buy_score", 0)
#                 st.metric("Buy Score", f"{buy_score:.1f}")
#             with col2:
#                 sell_score = signals.get("sell_score", 0)
#                 st.metric("Sell Score", f"{sell_score:.1f}")
            
#             # Show arguments
#             buy_args = signals.get("buy_arguments", [])
#             sell_args = signals.get("sell_arguments", [])
            
#             if buy_args:
#                 st.write("**Bullish Factors:**")
#                 for arg in buy_args:
#                     st.write(f"• {arg}")
            
#             if sell_args:
#                 st.write("**Bearish Factors:**")
#                 for arg in sell_args:
#                     st.write(f"• {arg}")


# # -------------------------
# # Runner initialization
# # -------------------------
# def ensure_session_runner():
#     """
#     Initialize AgentRunner + wrapped agents in session_state.
#     Robustly supports either create_wrapped_agents or create_inst_wrappers,
#     and factories that accept (tools, llm) or (llm) signature.
#     """
#     if "agent_runner" not in st.session_state:
#         runner = AgentRunner()
#         wrapped = {}
#         try:
#             # Prefer normal wrapper factory first
#             if create_wrapped_agents is not None:
#                 try:
#                     wrapped = create_wrapped_agents(tools=TOOLS, llm=LLM())
#                 except TypeError:
#                     wrapped = create_wrapped_agents(LLM())
#             elif create_inst_wrappers is not None:
#                 try:
#                     wrapped = create_inst_wrappers(tools=TOOLS, llm=LLM())
#                 except TypeError:
#                     wrapped = create_inst_wrappers(LLM())
#             else:
#                 logger.warning("No agent factory function available; starting empty AgentRunner.")
#                 wrapped = {}

#             # Register returned agents safely
#             for name, agent in (wrapped or {}).items():
#                 try:
#                     runner.register(name, agent)
#                 except Exception as e:
#                     logger.exception("Failed to register agent %s: %s", name, e)

#             st.session_state.agent_runner = runner
#             st.session_state.wrapped_agents = list((wrapped or {}).keys())
#             logger.info("AgentRunner initialized with agents: %s", st.session_state.wrapped_agents)

#         except Exception:
#             logger.exception("Failed to initialize wrapped agents")
#             st.session_state.agent_runner = runner
#             st.session_state.wrapped_agents = []
#     return st.session_state.agent_runner


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


# # -------------------------
# # Streamlit Page
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
#         st.markdown("Agents:")
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
#         status.info("1/6 — Fetch canonical price data (via TOOLS)")
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

#         # TECHNICAL
#         status.info("2/6 — Running Technical ")
#         progress.progress(20)
#         tech_raw = runner.run("technical", {"ticker": ticker, "start": start_date, "end": end_date})
#         tech_norm = normalize_response(tech_raw)
#         tech_payload = extract_payload(tech_norm, preferred_keys=("technical", "result"))
#         if isinstance(tech_payload, dict) and "df" not in tech_payload and isinstance(price_df, pd.DataFrame):
#             tech_payload["df"] = price_df

#         latest_close = latest_close or safe_get_latest_close(tech_payload)

#         # NEWS
#         status.info("3/6 — Running News ")
#         progress.progress(40)
#         news_raw = runner.run("news", {"ticker": ticker, "start": start_date, "end": end_date})
#         news_norm = normalize_response(news_raw)
#         news_payload = extract_payload(news_norm)

#         # RISK
#         status.info("4/6 — Running Risk ")
#         progress.progress(55)
#         risk_input = {
#             "ticker": ticker,
#             "start": start_date,
#             "end": end_date,
#             "technical_confidence": tech_payload.get("confidence", 50) if isinstance(tech_payload, dict) else 50,
#             "df": tech_payload.get("df", price_df),
#             "current_price": latest_close
#         }
#         risk_raw = runner.run("risk", risk_input)
#         risk_norm = normalize_response(risk_raw)
#         risk_payload = extract_payload(risk_norm)

#         # PORTFOLIO
#         status.info("5/6 — Running Portfolio ")
#         progress.progress(75)
#         port_input = {
#             "ticker": ticker,
#             "current_price": latest_close,
#             "technical_signal": tech_payload if isinstance(tech_payload, dict) else {},
#             "risk_metrics": risk_payload if isinstance(risk_payload, dict) else {},
#             "df": tech_payload.get("df", price_df)
#         }
#         port_raw = runner.run("portfolio", port_input)
#         port_norm = normalize_response(port_raw)
#         port_payload = extract_payload(port_norm)

#         # DEBATE (optional)
#         debate_payload = None
#         if show_debate:
#             status.info("6/6 — Running Debate Agent")
#             progress.progress(85)
#             debate_raw = runner.run("debate", {
#                 "ticker": ticker,
#                 "technical_result": tech_payload,
#                 "risk_metrics": risk_payload,
#                 "df": port_payload.get("df", tech_payload.get("df", price_df))
#             })
#             debate_norm = normalize_response(debate_raw)
#             debate_payload = extract_payload(debate_norm)
#             progress.progress(90)

#         # MASTER
#         status.info("Finalizing — Running Master Agent")
#         master_raw = runner.run("master", {
#             "ticker": ticker,
#             "technical_result": tech_payload,
#             "sentiment_result": debate_payload or news_payload,
#             "risk_metrics": risk_payload,
#             "portfolio_metrics": port_payload,
#             "current_price": latest_close
#         })
#         master_norm = normalize_response(master_raw)
#         # master can be nested under 'master' key or returned flat
#         master_payload = master_norm.get("master") if isinstance(master_norm, dict) and "master" in master_norm else extract_payload(master_norm, preferred_keys=("master", "result"))

#         progress.progress(100)
#         status.empty()
#         progress.empty()
#         st.success("✅ Analysis Complete")

#         # ---------- DISPLAY ----------
#         tab_all, tab_master, tab_debate, tab_news = st.tabs(["All Agents", "Master", "Debate", "News"])

#         with tab_all:
#             col1, col2 = st.columns(2)
#             with col1:
#                 display_json_friendly("Technical ", tech_payload)
#                 st.markdown("---")
#                 display_json_friendly("Risk ", risk_payload)
#             with col2:
#                 display_json_friendly("Portfolio ", port_payload)
#                 st.markdown("---")
#                 if debate_payload:
#                     display_json_friendly("Debate Agent", debate_payload)

#         with tab_master:
#             # Use the new clean display for Master Agent
#             display_master_agent_analysis(master_payload)

#         with tab_debate:
#             if debate_payload:
#                 display_json_friendly("Debate Output", debate_payload)
#             else:
#                 st.info("No debate output")

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

# trading_bot/ui/ai_agents.py
import os
from datetime import datetime, timedelta
import logging
import streamlit as st
import pandas as pd

from data.data_fetcher import fetch_data  # kept for potential future use

# Runner + factories (robust imports)
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


# -------------------------
# Helpers: normalize runner responses
# -------------------------
def normalize_response(resp):
    """
    Convert possible agent responses into a plain dict.
    - dict -> returned as shallow copy
    - pandas.DataFrame -> {'status':'OK','df': df}
    - objects with .dict() -> use .dict()
    - objects with __dict__ -> use vars()
    - string/other -> {'status':'OK','text': str(resp)}
    - None -> {'status':'ERROR', 'error': 'No response'}
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
    Safely display payload in Streamlit:
    - dict -> st.json (with dataframe shown separately)
    - DataFrame -> st.dataframe
    - list -> st.write
    - string -> st.write
    """
    st.markdown(f"### {label}")
    if payload is None:
        st.info("No output")
        return

    if isinstance(payload, dict):
        # show dataframe preview if present but don't mutate original dict
        df = payload.get("df", None)
        try:
            st.json({k: v for k, v in payload.items() if k != "df"})
        except Exception:
            st.write(payload)
        if df is not None:
            if isinstance(df, pd.DataFrame):
                st.markdown("**Data (preview)**")
                st.dataframe(df.head(10))
            else:
                st.write(df)
    elif isinstance(payload, pd.DataFrame):
        st.dataframe(payload.head(10))
    elif isinstance(payload, list):
        st.write(payload)
    else:
        st.write(str(payload))


def display_master_agent_analysis(master_payload: dict):
    """
    Clean, professional display for the new Master Agent
    """
    if not isinstance(master_payload, dict):
        st.error("No master analysis available")
        return

    # Extract display data or use main payload
    display_data = master_payload.get("display_data", {})
    if not display_data:
        display_data = {
            "final_decision": master_payload.get("action", "HOLD"),
            "confidence": master_payload.get("confidence", 50),
            "reasoning": master_payload.get("reasoning", "No reasoning provided"),
            "narrative": master_payload.get("narrative", "No analysis available"),
            "quantity": master_payload.get("quantity", 0),
            "stop_loss": master_payload.get("risk_management", {}).get("stop_loss"),
            "take_profit": master_payload.get("risk_management", {}).get("take_profit"),
            "current_price": master_payload.get("current_price", 0)
        }

    st.markdown("---")
    
    # Main Decision Card
    col1, col2, col3 = st.columns(3)
    
    decision = display_data.get("final_decision", "HOLD")
    confidence = display_data.get("confidence", 50)
    
    with col1:
        if decision == "BUY":
            st.success(f"## 🟢 {decision}")
        elif decision == "SELL":
            st.error(f"## 🔴 {decision}")  
        else:
            st.info(f"## ⚪ {decision}")
            
    with col2:
        st.metric("Confidence", f"{confidence}%")
        
    with col3:
        st.metric("Position Size", display_data.get("quantity", 0))
    
    # Current Price
    current_price = display_data.get("current_price")
    if current_price:
        st.write(f"**Current Price:** ${current_price:.2f}")
    
    # Narrative (Main story)
    st.markdown("### 📋 AI Analysis Summary")
    narrative = display_data.get("narrative", "No analysis available")
    st.info(narrative)
    
    # Risk Management
    st.markdown("### 🛡️ Risk Management")
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        stop_loss = display_data.get("stop_loss")
        if stop_loss:
            st.metric("Stop Loss", f"${stop_loss:.2f}")
        else:
            st.write("Stop Loss: N/A")
            
    with risk_col2:
        take_profit = display_data.get("take_profit") 
        if take_profit:
            st.metric("Take Profit", f"${take_profit:.2f}")
        else:
            st.write("Take Profit: N/A")
    
    # Signal Breakdown (expandable)
    with st.expander("🔍 Signal Analysis Details"):
        reasoning = display_data.get("reasoning", "No detailed reasoning")
        st.write(f"**Decision Logic:** {reasoning}")
        
        # Show signal scores if available
        signals = master_payload.get("signals", {})
        if signals:
            col1, col2 = st.columns(2)
            with col1:
                buy_score = signals.get("buy_score", 0)
                st.metric("Buy Score", f"{buy_score:.1f}")
            with col2:
                sell_score = signals.get("sell_score", 0)
                st.metric("Sell Score", f"{sell_score:.1f}")
            
            # Show arguments
            buy_args = signals.get("buy_arguments", [])
            sell_args = signals.get("sell_arguments", [])
            
            if buy_args:
                st.write("**Bullish Factors:**")
                for arg in buy_args:
                    st.write(f"• {arg}")
            
            if sell_args:
                st.write("**Bearish Factors:**")
                for arg in sell_args:
                    st.write(f"• {arg}")


# -------------------------
# Runner initialization - UPDATED WITH FALLBACK
# -------------------------
def ensure_session_runner():
    """
    Initialize AgentRunner with guaranteed agent registration
    """
    if "agent_runner" not in st.session_state:
        runner = AgentRunner()
        
        # Force register core agents if empty (double safety)
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
            for k in ("latest_close", "latest", "close", "price"):
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


# -------------------------
# Streamlit Page
# -------------------------
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
        status.info("1/6 — Fetch canonical price data (via TOOLS)")
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

        # TECHNICAL
        status.info("2/6 — Running Technical ")
        progress.progress(20)
        tech_raw = runner.run("technical", {"ticker": ticker, "start": start_date, "end": end_date})
        tech_norm = normalize_response(tech_raw)
        tech_payload = extract_payload(tech_norm, preferred_keys=("technical", "result"))
        if isinstance(tech_payload, dict) and "df" not in tech_payload and isinstance(price_df, pd.DataFrame):
            tech_payload["df"] = price_df

        latest_close = latest_close or safe_get_latest_close(tech_payload)

        # NEWS
        status.info("3/6 — Running News ")
        progress.progress(40)
        news_raw = runner.run("news", {"ticker": ticker, "start": start_date, "end": end_date})
        news_norm = normalize_response(news_raw)
        news_payload = extract_payload(news_norm)

        # RISK
        status.info("4/6 — Running Risk ")
        progress.progress(55)
        risk_input = {
            "ticker": ticker,
            "start": start_date,
            "end": end_date,
            "technical_confidence": tech_payload.get("confidence", 50) if isinstance(tech_payload, dict) else 50,
            "df": tech_payload.get("df", price_df),
            "current_price": latest_close
        }
        risk_raw = runner.run("risk", risk_input)
        risk_norm = normalize_response(risk_raw)
        risk_payload = extract_payload(risk_norm)

        # PORTFOLIO
        status.info("5/6 — Running Portfolio ")
        progress.progress(75)
        port_input = {
            "ticker": ticker,
            "current_price": latest_close,
            "technical_signal": tech_payload if isinstance(tech_payload, dict) else {},
            "risk_metrics": risk_payload if isinstance(risk_payload, dict) else {},
            "df": tech_payload.get("df", price_df)
        }
        port_raw = runner.run("portfolio", port_input)
        port_norm = normalize_response(port_raw)
        port_payload = extract_payload(port_norm)

        # DEBATE (optional)
        debate_payload = None
        if show_debate:
            status.info("6/6 — Running Debate Agent")
            progress.progress(85)
            debate_raw = runner.run("debate", {
                "ticker": ticker,
                "technical_result": tech_payload,
                "risk_metrics": risk_payload,
                "df": port_payload.get("df", tech_payload.get("df", price_df))
            })
            debate_norm = normalize_response(debate_raw)
            debate_payload = extract_payload(debate_norm)
            progress.progress(90)

        # MASTER
        status.info("Finalizing — Running Master Agent")
        master_raw = runner.run("master", {
            "ticker": ticker,
            "technical_result": tech_payload,
            "sentiment_result": debate_payload or news_payload,
            "risk_metrics": risk_payload,
            "portfolio_metrics": port_payload,
            "current_price": latest_close
        })
        master_norm = normalize_response(master_raw)
        # master can be nested under 'master' key or returned flat
        master_payload = master_norm.get("master") if isinstance(master_norm, dict) and "master" in master_norm else extract_payload(master_norm, preferred_keys=("master", "result"))

        progress.progress(100)
        status.empty()
        progress.empty()
        st.success("✅ Analysis Complete")

        # ---------- DISPLAY ----------
        tab_all, tab_master, tab_debate, tab_news = st.tabs(["All Agents", "Master", "Debate", "News"])

        with tab_all:
            col1, col2 = st.columns(2)
            with col1:
                display_json_friendly("Technical ", tech_payload)
                st.markdown("---")
                display_json_friendly("Risk ", risk_payload)
            with col2:
                display_json_friendly("Portfolio ", port_payload)
                st.markdown("---")
                if debate_payload:
                    display_json_friendly("Debate Agent", debate_payload)

        with tab_master:
            # Use the new clean display for Master Agent
            display_master_agent_analysis(master_payload)

        with tab_debate:
            if debate_payload:
                display_json_friendly("Debate Output", debate_payload)
            else:
                st.info("No debate output")

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
        if isinstance(master_payload, dict) and (master_payload.get("action") or master_payload.get("recommendation")):
            master_action = master_payload.get("action", master_payload.get("recommendation", "HOLD"))
            if master_action != "HOLD":
                qty_default = int(port_payload.get("suggested_quantity", port_payload.get("quantity", 1)) if isinstance(port_payload, dict) else 1)
                qty = st.number_input("Quantity", min_value=1, value=qty_default)
                if st.button(f"Simulate {master_action}"):
                    st.success(f"Simulated {master_action} {qty} @ {latest_close:.2f}")
            else:
                st.info("Master recommends HOLD")

    except Exception as e:
        logger.exception("Pipeline failure")
        st.error(f"Pipeline error: {e}")
        with st.expander("Trace"):
            import traceback
            st.code(traceback.format_exc())