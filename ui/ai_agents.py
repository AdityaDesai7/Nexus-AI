# # ui/ai_agents.py
# import streamlit as st
# from datetime import datetime
# import time
# from main import get_trading_system, fetch_and_enhance


# def show_ai_agents_page():
#     """Display AI agents analysis page"""
    
#     # Check if user is logged in
#     if not st.session_state.get('logged_in'):
#         st.error("⚠️ Please login first to use AI agents")
#         if st.button("Go to Login"):
#             st.session_state.page = 'login'
#             st.rerun()
#         return
    
#     ticker = st.session_state.get('ticker', 'RELIANCE.NS')
    
#     # Header
#     col1, col2 = st.columns([1, 6])
#     with col1:
#         if st.button("← Back to Analysis"):
#             st.session_state.page = 'analysis'
#             st.rerun()
#     with col2:
#         st.markdown(f"<h1 style='text-align:center; color:#00b894;'>🤖 AI Agent Analysis for {ticker}</h1>", unsafe_allow_html=True)
    
#     # Initialize trading system
#     if 'trading_system' not in st.session_state:
#         st.session_state.trading_system = get_trading_system(portfolio_value=1000000)
    
#     ts = st.session_state.trading_system
    
#     # Date range
#     col1, col2, col3 = st.columns([2, 2, 1])
#     with col1:
#         start_date = st.date_input("Start Date", datetime(2023, 1, 1))
#     with col2:
#         end_date = st.date_input("End Date", datetime.now())
#     with col3:
#         run_analysis = st.button("🚀 Run AI Agents", type="primary", use_container_width=True)
    
#     if run_analysis:
#         # Fetch data first
#         with st.spinner("Fetching market data..."):
#             try:
#                 df = fetch_and_enhance(ticker, start_date, end_date)
#                 latest = df.iloc[-1]
#             except Exception as e:
#                 st.error(f"Error fetching data: {str(e)}")
#                 return
        
#         # Create tabs
#         tab1, tab2, tab3 = st.tabs(["🤖 Agent Activity", "💡 Final Decision", "📊 Details"])
        
#         with tab1:
#             st.markdown("## Multi-Agent Analysis Pipeline")
            
#             # Progress tracking
#             progress_bar = st.progress(0)
#             status_container = st.empty()
            
#             agent_sequence = [
#                 ("Technical Agent", "Analyzing price patterns and indicators..."),
#                 ("Sentiment Agent", "Evaluating market sentiment..."),
#                 ("Risk Agent", "Assessing volatility and risk..."),
#                 ("Portfolio Agent", "Calculating position sizing..."),
#                 ("Master Agent", "Synthesizing recommendations...")
#             ]
            
#             # Animate execution
#             for idx, (agent_name, description) in enumerate(agent_sequence):
#                 progress = idx / len(agent_sequence)
#                 progress_bar.progress(progress)
                
#                 status_html = f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; margin: 12px 0; border-left: 4px solid #00b894;'>
#                     <span style='display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; background: #00b894; box-shadow: 0 0 10px #00b894;'></span>
#                     <strong style='font-size:1.1rem; color:#00b894;'>{agent_name}</strong>
#                     <p style='margin:8px 0 0 22px; color:#b2bec3;'>{description}</p>
#                 </div>
#                 """
#                 status_container.markdown(status_html, unsafe_allow_html=True)
#                 time.sleep(0.4)
            
#             # Run analysis
#             try:
#                 result = ts.run_analysis(ticker, start_date, end_date)
#                 st.session_state.analysis_result = result
#             except Exception as e:
#                 st.error(f"Error running analysis: {str(e)}")
#                 return
            
#             progress_bar.progress(1.0)
#             time.sleep(0.2)
#             progress_bar.empty()
            
#             st.success("✅ Multi-agent analysis complete!")
            
#             # Show recommendations
#             st.markdown("### 📋 Agent Recommendations")
            
#             recommendations = result['recommendations']
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 tech = recommendations['technical']
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; border-left: 4px solid #00b894;'>
#                     <h4 style='color:#00b894;'>📊 Technical Agent</h4>
#                     <p><strong>Action:</strong> {tech['action']}</p>
#                     <p><strong>Confidence:</strong> {tech['confidence']:.1f}%</p>
#                     <p><strong>Signals:</strong></p>
#                     <ul>{"".join([f"<li>{s}</li>" for s in tech['signals'][:3]])}</ul>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 risk = recommendations['risk']
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; border-left: 4px solid #e17055; margin-top: 12px;'>
#                     <h4 style='color:#e17055;'>⚠️ Risk Agent</h4>
#                     <p><strong>Risk Level:</strong> {risk['risk_level']}</p>
#                     <p><strong>Position Size:</strong> {risk['position_size']}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             with col2:
#                 sent = recommendations['sentiment']
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; border-left: 4px solid #6c5ce7;'>
#                     <h4 style='color:#6c5ce7;'>📰 Sentiment Agent</h4>
#                     <p><strong>Action:</strong> {sent['action']}</p>
#                     <p><strong>Confidence:</strong> {sent['confidence']:.1f}%</p>
#                     <p><strong>Signals:</strong></p>
#                     <ul>{"".join([f"<li>{s}</li>" for s in sent['signals'][:3]])}</ul>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 port = recommendations['portfolio']
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; border-left: 4px solid #00cec9; margin-top: 12px;'>
#                     <h4 style='color:#00cec9;'>💼 Portfolio Agent</h4>
#                     <p><strong>Allocation:</strong> {port['allocation_pct']}%</p>
#                     <p><strong>Quantity:</strong> {port['suggested_quantity']} shares</p>
#                 </div>
#                 """, unsafe_allow_html=True)
        
#         with tab2:
#             st.markdown("## 💡 Master Agent Final Decision")
            
#             master = result['recommendations']['master']
            
#             col1, col2, col3 = st.columns([2, 1, 1])
            
#             with col1:
#                 action_color = "#00b894" if master['action'] == "BUY" else "#ff7675" if master['action'] == "SELL" else "#fdcb6e"
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, {action_color}22, {action_color}11); padding: 30px; border-radius: 16px; border: 2px solid {action_color};'>
#                     <h2 style='color:{action_color}; margin:0; font-size:2.5rem;'>🎯 {master['action']}</h2>
#                     <p style='color:#b2bec3; margin:10px 0 0 0; font-size:1.1rem;'>Master Agent Recommendation</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             with col2:
#                 st.metric("Confidence", f"{master['confidence']:.1f}%")
#                 st.metric("Quantity", f"{master['quantity']} shares")
            
#             with col3:
#                 st.metric("Risk Level", master['risk_level'])
#                 estimated_value = master['quantity'] * latest['Close']
#                 st.metric("Est. Value", f"₹{estimated_value:,.0f}")
            
#             st.markdown("---")
#             st.markdown("### 📝 Reasoning")
#             st.info(master['reasoning'])
            
#             # ============================================
#             # TRADE EXECUTION SECTION
#             # ============================================
#             st.markdown("---")
            
#             if master['action'] != "HOLD":
#                 st.markdown("### 💼 Execute Trade")
                
#                 # Debug info (remove this later)
#                 with st.expander("🔍 Debug Info"):
#                     st.write(f"Logged in: {st.session_state.get('logged_in')}")
#                     st.write(f"User ID: {st.session_state.get('user_id')}")
#                     st.write(f"Username: {st.session_state.get('username')}")
#                     st.write(f"Ticker: {ticker}")
#                     st.write(f"Current Price: ₹{latest['Close']:.2f}")
                
#                 col1, col2 = st.columns([2, 1])
                
#                 with col1:
#                     # Allow user to modify quantity
#                     suggested_qty = master['quantity']
                    
#                     # Use unique key with ticker
#                     trade_qty = st.number_input(
#                         "Quantity to trade",
#                         min_value=1,
#                         value=suggested_qty,
#                         step=1,
#                         key=f"trade_qty_{ticker}"
#                     )
                    
#                     estimated_value = trade_qty * latest['Close']
#                     st.info(f"💰 Estimated transaction value: ₹{estimated_value:,.2f}")
                
#                 with col2:
#                     st.markdown("<br>", unsafe_allow_html=True)
                    
#                     # Use unique key for button
#                     execute_button = st.button(
#                         f"✅ Execute {master['action']} Order",
#                         type="primary",
#                         use_container_width=True,
#                         key=f"execute_{ticker}"
#                     )
                    
#                     if execute_button:
#                         # Verify user is logged in
#                         if not st.session_state.get('logged_in') or not st.session_state.get('user_id'):
#                             st.error("❌ Not logged in! Please login first.")
#                             st.stop()
                        
#                         try:
#                             from models.database import get_database
                            
#                             db = get_database()
#                             user_id = st.session_state.user_id
                            
#                             st.info(f"📊 Executing trade for User ID: {user_id}")
                            
#                             # Prepare agent recommendation summary
#                             agent_rec = f"{master['action']} - Confidence: {master['confidence']:.1f}% - {master['reasoning'][:100]}"
                            
#                             # Execute trade
#                             with st.spinner("⏳ Executing trade..."):
#                                 success, message = db.execute_trade(
#                                     user_id=user_id,
#                                     ticker=ticker,
#                                     action=master['action'],
#                                     quantity=trade_qty,
#                                     price=latest['Close'],
#                                     agent_rec=agent_rec
#                                 )
                            
#                             if success:
#                                 st.balloons()
#                                 st.success(f"✅ {message}")
                                
#                                 # Show redirect message
#                                 st.success("🎉 Redirecting to portfolio in 3 seconds...")
#                                 time.sleep(3)
                                
#                                 # Clear cache and redirect to portfolio
#                                 st.session_state.page = 'portfolio'
#                                 st.rerun()
#                             else:
#                                 st.error(f"❌ {message}")
                        
#                         except Exception as e:
#                             st.error(f"❌ Error executing trade: {str(e)}")
#                             st.exception(e)  # Show full error for debugging
#             else:
#                 st.markdown("---")
#                 st.warning("⏸️ Master Agent recommends HOLD. No trade execution needed at this time.")
        
#         with tab3:
#             st.markdown("## 📊 Detailed Analysis")
            
#             if 'analysis_result' in st.session_state:
#                 st.json(st.session_state.analysis_result['recommendations'])
#             else:
#                 st.info("Run the analysis first to see details")
    
#     else:
#         st.info("⚙️ Configure date range and click **Run AI Agents** to start the multi-agent analysis")
        
#         # Show instructions
#         with st.expander("📖 How to use AI Agents"):
#             st.markdown("""
#             ### Steps:
            
#             1. **Select Date Range** - Choose start and end dates for analysis
#             2. **Click "Run AI Agents"** - This will trigger the multi-agent system
#             3. **View Agent Activity** - Watch as 5 specialized agents analyze the stock
#             4. **Review Final Decision** - See the master agent's recommendation
#             5. **Execute Trade** - If you agree, click the execute button
#             6. **View Portfolio** - Check your updated holdings
            
#             **Note:** Make sure you're logged in to execute trades!
#             """)


# # ui/ai_agents.py
# import streamlit as st
# from datetime import datetime
# import time
# from main import get_trading_system, fetch_and_enhance


# def show_ai_agents_page():
#     """Display AI agents analysis page"""
    
#     # Check if user is logged in
#     if not st.session_state.get('logged_in'):
#         st.error("⚠️ Please login first to use AI agents")
#         if st.button("Go to Login"):
#             st.session_state.page = 'login'
#             st.rerun()
#         return
    
#     ticker = st.session_state.get('ticker', 'RELIANCE.NS')
    
#     # Header
#     col1, col2 = st.columns([1, 6])
#     with col1:
#         if st.button("← Back to Analysis"):
#             st.session_state.page = 'analysis'
#             st.rerun()
#     with col2:
#         st.markdown(f"<h1 style='text-align:center; color:#00b894;'>🤖 AI Agent Analysis for {ticker}</h1>", unsafe_allow_html=True)
    
#     # Initialize trading system
#     if 'trading_system' not in st.session_state:
#         st.session_state.trading_system = get_trading_system(portfolio_value=1000000)
    
#     ts = st.session_state.trading_system
    
#     # Date range - FIX: Convert date to datetime
#     col1, col2, col3 = st.columns([2, 2, 1])
#     with col1:
#         start_date_input = st.date_input("Start Date", datetime(2023, 1, 1))
#         # Convert date to datetime
#         start_date = datetime.combine(start_date_input, datetime.min.time())
#     with col2:
#         end_date_input = st.date_input("End Date", datetime.now())
#         # Convert date to datetime
#         end_date = datetime.combine(end_date_input, datetime.max.time())
#     with col3:
#         run_analysis = st.button("🚀 Run AI Agents", type="primary", use_container_width=True)
    
#     if run_analysis:
#         # Fetch data first
#         with st.spinner("Fetching market data..."):
#             try:
#                 df = fetch_and_enhance(ticker, start_date, end_date)
#                 latest = df.iloc[-1]
#             except Exception as e:
#                 st.error(f"Error fetching data: {str(e)}")
#                 return
        
#         # Create tabs
#         tab1, tab2, tab3 = st.tabs(["🤖 Agent Activity", "💡 Final Decision", "📊 Details"])
        
#         with tab1:
#             st.markdown("## Multi-Agent Analysis Pipeline")
            
#             # Progress tracking
#             progress_bar = st.progress(0)
#             status_container = st.empty()
            
#             agent_sequence = [
#                 ("Technical Agent", "Analyzing price patterns and indicators..."),
#                 ("Sentiment Agent", "Evaluating market sentiment..."),
#                 ("Risk Agent", "Assessing volatility and risk..."),
#                 ("Portfolio Agent", "Calculating position sizing..."),
#                 ("Master Agent", "Synthesizing recommendations...")
#             ]
            
#             # Animate execution
#             for idx, (agent_name, description) in enumerate(agent_sequence):
#                 progress = idx / len(agent_sequence)
#                 progress_bar.progress(progress)
                
#                 status_html = f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; margin: 12px 0; border-left: 4px solid #00b894;'>
#                     <span style='display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; background: #00b894; box-shadow: 0 0 10px #00b894;'></span>
#                     <strong style='font-size:1.1rem; color:#00b894;'>{agent_name}</strong>
#                     <p style='margin:8px 0 0 22px; color:#b2bec3;'>{description}</p>
#                 </div>
#                 """
#                 status_container.markdown(status_html, unsafe_allow_html=True)
#                 time.sleep(0.4)
            
#             # Run analysis
#             try:
#                 result = ts.run_analysis(ticker, start_date, end_date)
#                 st.session_state.analysis_result = result
#             except Exception as e:
#                 st.error(f"Error running analysis: {str(e)}")
#                 return
            
#             progress_bar.progress(1.0)
#             time.sleep(0.2)
#             progress_bar.empty()
            
#             st.success("✅ Multi-agent analysis complete!")
            
#             # Show recommendations
#             st.markdown("### 📋 Agent Recommendations")
            
#             recommendations = result['recommendations']
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 tech = recommendations['technical']
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; border-left: 4px solid #00b894;'>
#                     <h4 style='color:#00b894;'>📊 Technical Agent</h4>
#                     <p><strong>Action:</strong> {tech['action']}</p>
#                     <p><strong>Confidence:</strong> {tech['confidence']:.1f}%</p>
#                     <p><strong>Signals:</strong></p>
#                     <ul>{"".join([f"<li>{s}</li>" for s in tech['signals'][:3]])}</ul>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 risk = recommendations['risk']
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; border-left: 4px solid #e17055; margin-top: 12px;'>
#                     <h4 style='color:#e17055;'>⚠️ Risk Agent</h4>
#                     <p><strong>Risk Level:</strong> {risk['risk_level']}</p>
#                     <p><strong>Position Size:</strong> {risk['position_size']}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             with col2:
#                 sent = recommendations['sentiment']
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; border-left: 4px solid #6c5ce7;'>
#                     <h4 style='color:#6c5ce7;'>📰 Sentiment Agent</h4>
#                     <p><strong>Action:</strong> {sent['action']}</p>
#                     <p><strong>Confidence:</strong> {sent['confidence']:.1f}%</p>
#                     <p><strong>Signals:</strong></p>
#                     <ul>{"".join([f"<li>{s}</li>" for s in sent['signals'][:3]])}</ul>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 port = recommendations['portfolio']
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; border-left: 4px solid #00cec9; margin-top: 12px;'>
#                     <h4 style='color:#00cec9;'>💼 Portfolio Agent</h4>
#                     <p><strong>Allocation:</strong> {port['allocation_pct']}%</p>
#                     <p><strong>Quantity:</strong> {port['suggested_quantity']} shares</p>
#                 </div>
#                 """, unsafe_allow_html=True)
        
#         with tab2:
#             st.markdown("## 💡 Master Agent Final Decision")
            
#             master = result['recommendations']['master']
            
#             col1, col2, col3 = st.columns([2, 1, 1])
            
#             with col1:
#                 action_color = "#00b894" if master['action'] == "BUY" else "#ff7675" if master['action'] == "SELL" else "#fdcb6e"
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, {action_color}22, {action_color}11); padding: 30px; border-radius: 16px; border: 2px solid {action_color};'>
#                     <h2 style='color:{action_color}; margin:0; font-size:2.5rem;'>🎯 {master['action']}</h2>
#                     <p style='color:#b2bec3; margin:10px 0 0 0; font-size:1.1rem;'>Master Agent Recommendation</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             with col2:
#                 st.metric("Confidence", f"{master['confidence']:.1f}%")
#                 st.metric("Quantity", f"{master['quantity']} shares")
            
#             with col3:
#                 st.metric("Risk Level", master['risk_level'])
#                 estimated_value = master['quantity'] * latest['Close']
#                 st.metric("Est. Value", f"₹{estimated_value:,.0f}")
            
#             st.markdown("---")
#             st.markdown("### 📝 Reasoning")
#             st.info(master['reasoning'])
            
#             # ============================================
#             # TRADE EXECUTION SECTION
#             # ============================================
#             st.markdown("---")
            
#             if master['action'] != "HOLD":
#                 st.markdown("### 💼 Execute Trade")
                
#                 col1, col2 = st.columns([2, 1])
                
#                 with col1:
#                     # Allow user to modify quantity
#                     suggested_qty = master['quantity']
                    
#                     # Use unique key with ticker
#                     trade_qty = st.number_input(
#                         "Quantity to trade",
#                         min_value=1,
#                         value=suggested_qty,
#                         step=1,
#                         key=f"trade_qty_{ticker}"
#                     )
                    
#                     estimated_value = trade_qty * latest['Close']
#                     st.info(f"💰 Estimated transaction value: ₹{estimated_value:,.2f}")
                
#                 with col2:
#                     st.markdown("<br>", unsafe_allow_html=True)
                    
#                     # Use unique key for button
#                     execute_button = st.button(
#                         f"✅ Execute {master['action']} Order",
#                         type="primary",
#                         use_container_width=True,
#                         key=f"execute_{ticker}"
#                     )
                    
#                     if execute_button:
#                         # Verify user is logged in
#                         if not st.session_state.get('logged_in') or not st.session_state.get('user_id'):
#                             st.error("❌ Not logged in! Please login first.")
#                             st.stop()
                        
#                         try:
#                             from models.database import get_database
                            
#                             db = get_database()
#                             user_id = st.session_state.user_id
                            
#                             # Prepare agent recommendation summary
#                             agent_rec = f"{master['action']} - Confidence: {master['confidence']:.1f}% - {master['reasoning'][:100]}"
                            
#                             # Execute trade
#                             with st.spinner("⏳ Executing trade..."):
#                                 success, message = db.execute_trade(
#                                     user_id=user_id,
#                                     ticker=ticker,
#                                     action=master['action'],
#                                     quantity=trade_qty,
#                                     price=latest['Close'],
#                                     agent_rec=agent_rec
#                                 )
                            
#                             if success:
#                                 st.balloons()
#                                 st.success(f"✅ {message}")
                                
#                                 # Show redirect message
#                                 st.success("🎉 Redirecting to portfolio in 3 seconds...")
#                                 time.sleep(3)
                                
#                                 # Clear cache and redirect to portfolio
#                                 st.session_state.page = 'portfolio'
#                                 st.rerun()
#                             else:
#                                 st.error(f"❌ {message}")
                        
#                         except Exception as e:
#                             st.error(f"❌ Error executing trade: {str(e)}")
#             else:
#                 st.markdown("---")
#                 st.warning("⏸️ Master Agent recommends HOLD. No trade execution needed at this time.")
        
#         with tab3:
#             st.markdown("## 📊 Detailed Analysis")
            
#             if 'analysis_result' in st.session_state:
#                 st.json(st.session_state.analysis_result['recommendations'])
#             else:
#                 st.info("Run the analysis first to see details")
    
#     else:
#         st.info("⚙️ Configure date range and click **Run AI Agents** to start the multi-agent analysis")
        
#         # Show instructions
#         with st.expander("📖 How to use AI Agents"):
#             st.markdown("""
#             ### Steps:
            
#             1. **Select Date Range** - Choose start and end dates for analysis
#             2. **Click "Run AI Agents"** - This will trigger the multi-agent system
#             3. **View Agent Activity** - Watch as 5 specialized agents analyze the stock
#             4. **Review Final Decision** - See the master agent's recommendation
#             5. **Execute Trade** - If you agree, click the execute button
#             6. **View Portfolio** - Check your updated holdings
            
#             **Note:** Make sure you're logged in to execute trades!
#             """)






# # ui/ai_agents.py
# import streamlit as st
# from datetime import datetime
# import time
# from main import get_trading_system, fetch_and_enhance
# from agents.technical_agent import TechnicalAnalysisAgent


# def show_ai_agents_page():
#     """Display AI agents analysis page with AUTONOMOUS TECHNICAL AGENT"""
    
#     # Check if user is logged in
#     if not st.session_state.get('logged_in'):
#         st.error("⚠️ Please login first to use AI agents")
#         if st.button("Go to Login"):
#             st.session_state.page = 'login'
#             st.rerun()
#         return
    
#     ticker = st.session_state.get('ticker', 'RELIANCE.NS')
    
#     # Header
#     col1, col2 = st.columns([1, 6])
#     with col1:
#         if st.button("← Back to Analysis"):
#             st.session_state.page = 'analysis'
#             st.rerun()
#     with col2:
#         st.markdown(f"<h1 style='text-align:center; color:#00b894;'>🤖 AI Agent Analysis for {ticker}</h1>", unsafe_allow_html=True)
    
#     # Initialize trading system
#     if 'trading_system' not in st.session_state:
#         st.session_state.trading_system = get_trading_system(portfolio_value=1000000)
    
#     ts = st.session_state.trading_system
    
#     # Initialize technical agent (AUTONOMOUS - NO LLM)
#     if 'tech_agent' not in st.session_state:
#         st.session_state.tech_agent = TechnicalAnalysisAgent()
    
#     tech_agent = st.session_state.tech_agent
    
#     # Date range
#     col1, col2, col3 = st.columns([2, 2, 1])
#     with col1:
#         start_date_input = st.date_input("Start Date", datetime(2023, 1, 1))
#         start_date = datetime.combine(start_date_input, datetime.min.time())
#     with col2:
#         end_date_input = st.date_input("End Date", datetime.now())
#         end_date = datetime.combine(end_date_input, datetime.max.time())
#     with col3:
#         run_analysis = st.button("🚀 Run AI Agents", type="primary", use_container_width=True)
    
#     if run_analysis:
#         # Fetch data first
#         with st.spinner("Fetching market data..."):
#             try:
#                 df = fetch_and_enhance(ticker, start_date, end_date)
#                 latest = df.iloc[-1]
#             except Exception as e:
#                 st.error(f"Error fetching data: {str(e)}")
#                 return
        
#         # Create tabs
#         tab1, tab2, tab3 = st.tabs(["🤖 Agent Activity", "💡 Final Decision", "📊 Details"])
        
#         with tab1:
#             st.markdown("## Multi-Agent Analysis Pipeline")
            
#             # Progress tracking
#             progress_bar = st.progress(0)
#             status_container = st.empty()
            
#             agent_sequence = [
#                 ("Technical Agent", "Analyzing price patterns with MACD, BB, LOFS, RSI..."),
#                 ("Sentiment Agent", "Evaluating market sentiment..."),
#                 ("Risk Agent", "Assessing volatility and risk..."),
#                 ("Portfolio Agent", "Calculating position sizing..."),
#                 ("Master Agent", "Synthesizing recommendations...")
#             ]
            
#             # Animate execution
#             for idx, (agent_name, description) in enumerate(agent_sequence):
#                 progress = idx / len(agent_sequence)
#                 progress_bar.progress(progress)
                
#                 status_html = f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; margin: 12px 0; border-left: 4px solid #00b894;'>
#                     <span style='display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; background: #00b894; box-shadow: 0 0 10px #00b894;'></span>
#                     <strong style='font-size:1.1rem; color:#00b894;'>{agent_name}</strong>
#                     <p style='margin:8px 0 0 22px; color:#b2bec3;'>{description}</p>
#                 </div>
#                 """
#                 status_container.markdown(status_html, unsafe_allow_html=True)
#                 time.sleep(0.4)
            
#             # Run TECHNICAL ANALYSIS (AUTONOMOUS)
#             try:
#                 tech_result = tech_agent.analyze(ticker, start_date, end_date)
#                 st.session_state.tech_result = tech_result
#             except Exception as e:
#                 st.error(f"Error in technical analysis: {str(e)}")
#                 return
            
#             # Run other agents
#             try:
#                 result = ts.run_analysis(ticker, start_date, end_date)
#                 st.session_state.analysis_result = result
#             except Exception as e:
#                 st.error(f"Error running other agents: {str(e)}")
#                 return
            
#             progress_bar.progress(1.0)
#             time.sleep(0.2)
#             progress_bar.empty()
            
#             st.success("✅ Multi-agent analysis complete!")
            
#             # Show recommendations
#             st.markdown("### 📋 Agent Recommendations")
            
#             recommendations = result['recommendations']
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 # TECHNICAL AGENT - NOW AUTONOMOUS
#                 action_color_tech = "#00b894" if "BUY" in tech_result.recommendation else "#ff7675" if "SELL" in tech_result.recommendation else "#fdcb6e"
                
#                 # Extract confidence and signals from recommendation string
#                 rec_text = tech_result.recommendation
#                 signals_list = rec_text.split("Signals: ")[1].split(" - ")[0] if "Signals:" in rec_text else "No signals"
                
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; border-left: 4px solid {action_color_tech};'>
#                     <h4 style='color:{action_color_tech};'>📊 Technical Agent (AUTONOMOUS)</h4>
#                     <p><strong>Signal:</strong> {rec_text.split("(")[0].strip()}</p>
#                     <p><strong>Confidence:</strong> {rec_text.split("Confidence: ")[1].split("%")[0]}%</p>
#                     <p><strong>Indicators:</strong></p>
#                     <ul>
#                         <li>MACD: {tech_result.macd:.4f} (Signal: {tech_result.macd_signal:.4f})</li>
#                         <li>RSI: {tech_result.rsi:.2f}</li>
#                         <li>BB Upper: {tech_result.bollinger_upper:.2f}</li>
#                         <li>BB Lower: {tech_result.bollinger_lower:.2f}</li>
#                     </ul>
#                     <p><strong>Triggered:</strong> {signals_list}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 risk = recommendations['risk']
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; border-left: 4px solid #e17055; margin-top: 12px;'>
#                     <h4 style='color:#e17055;'>⚠️ Risk Agent</h4>
#                     <p><strong>Risk Level:</strong> {risk['risk_level']}</p>
#                     <p><strong>Position Size:</strong> {risk['position_size']}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             with col2:
#                 sent = recommendations['sentiment']
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; border-left: 4px solid #6c5ce7;'>
#                     <h4 style='color:#6c5ce7;'>📰 Sentiment Agent</h4>
#                     <p><strong>Action:</strong> {sent['action']}</p>
#                     <p><strong>Confidence:</strong> {sent['confidence']:.1f}%</p>
#                     <p><strong>Signals:</strong></p>
#                     <ul>{"".join([f"<li>{s}</li>" for s in sent['signals'][:3]])}</ul>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 port = recommendations['portfolio']
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 18px; border-radius: 12px; border-left: 4px solid #00cec9; margin-top: 12px;'>
#                     <h4 style='color:#00cec9;'>💼 Portfolio Agent</h4>
#                     <p><strong>Allocation:</strong> {port['allocation_pct']}%</p>
#                     <p><strong>Quantity:</strong> {port['suggested_quantity']} shares</p>
#                 </div>
#                 """, unsafe_allow_html=True)
        
#         with tab2:
#             st.markdown("## 💡 Master Agent Final Decision")
            
#             master = result['recommendations']['master']
            
#             col1, col2, col3 = st.columns([2, 1, 1])
            
#             with col1:
#                 action_color = "#00b894" if master['action'] == "BUY" else "#ff7675" if master['action'] == "SELL" else "#fdcb6e"
#                 st.markdown(f"""
#                 <div style='background: linear-gradient(135deg, {action_color}22, {action_color}11); padding: 30px; border-radius: 16px; border: 2px solid {action_color};'>
#                     <h2 style='color:{action_color}; margin:0; font-size:2.5rem;'>🎯 {master['action']}</h2>
#                     <p style='color:#b2bec3; margin:10px 0 0 0; font-size:1.1rem;'>Master Agent Recommendation</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             with col2:
#                 st.metric("Confidence", f"{master['confidence']:.1f}%")
#                 st.metric("Quantity", f"{master['quantity']} shares")
            
#             with col3:
#                 st.metric("Risk Level", master['risk_level'])
#                 estimated_value = master['quantity'] * latest['Close']
#                 st.metric("Est. Value", f"₹{estimated_value:,.0f}")
            
#             st.markdown("---")
#             st.markdown("### 📝 Reasoning")
#             st.info(master['reasoning'])
            
#             # Show technical agent's autonomous decision
#             st.markdown("---")
#             st.markdown("### 🔍 Technical Agent Analysis (Autonomous)")
#             tech_result = st.session_state.get('tech_result')
#             if tech_result:
#                 st.info(f"**Autonomous Signal:** {tech_result.recommendation}")
            
#             # ============================================
#             # TRADE EXECUTION SECTION
#             # ============================================
#             st.markdown("---")
            
#             if master['action'] != "HOLD":
#                 st.markdown("### 💼 Execute Trade")
                
#                 col1, col2 = st.columns([2, 1])
                
#                 with col1:
#                     suggested_qty = master['quantity']
                    
#                     trade_qty = st.number_input(
#                         "Quantity to trade",
#                         min_value=1,
#                         value=suggested_qty,
#                         step=1,
#                         key=f"trade_qty_{ticker}"
#                     )
                    
#                     estimated_value = trade_qty * latest['Close']
#                     st.info(f"💰 Estimated transaction value: ₹{estimated_value:,.2f}")
                
#                 with col2:
#                     st.markdown("<br>", unsafe_allow_html=True)
                    
#                     execute_button = st.button(
#                         f"✅ Execute {master['action']} Order",
#                         type="primary",
#                         use_container_width=True,
#                         key=f"execute_{ticker}"
#                     )
                    
#                     if execute_button:
#                         # Verify user is logged in
#                         if not st.session_state.get('logged_in') or not st.session_state.get('user_id'):
#                             st.error("❌ Not logged in! Please login first.")
#                             st.stop()
                        
#                         try:
#                             from models.database import get_database
                            
#                             db = get_database()
#                             user_id = st.session_state.user_id
                            
#                             # Prepare agent recommendation summary (including autonomous tech signal)
#                             tech_rec = st.session_state.get('tech_result').recommendation if st.session_state.get('tech_result') else "N/A"
#                             agent_rec = f"{master['action']} - Tech: {tech_rec[:80]} - Master Confidence: {master['confidence']:.1f}%"
                            
#                             # Execute trade
#                             with st.spinner("⏳ Executing trade..."):
#                                 success, message = db.execute_trade(
#                                     user_id=user_id,
#                                     ticker=ticker,
#                                     action=master['action'],
#                                     quantity=trade_qty,
#                                     price=latest['Close'],
#                                     agent_rec=agent_rec
#                                 )
                            
#                             if success:
#                                 st.balloons()
#                                 st.success(f"✅ {message}")
                                
#                                 st.success("🎉 Redirecting to portfolio in 3 seconds...")
#                                 time.sleep(3)
                                
#                                 st.session_state.page = 'portfolio'
#                                 st.rerun()
#                             else:
#                                 st.error(f"❌ {message}")
                        
#                         except Exception as e:
#                             st.error(f"❌ Error executing trade: {str(e)}")
#             else:
#                 st.markdown("---")
#                 st.warning("⏸️ Master Agent recommends HOLD. No trade execution needed at this time.")
        
#         with tab3:
#             st.markdown("## 📊 Detailed Analysis")
            
#             # Show technical agent details
#             if 'tech_result' in st.session_state:
#                 tech = st.session_state['tech_result']
#                 st.subheader("Technical Agent (AUTONOMOUS)")
#                 st.json({
#                     "recommendation": tech.recommendation,
#                     "rsi": float(tech.rsi),
#                     "macd": float(tech.macd),
#                     "macd_signal": float(tech.macd_signal),
#                     "bollinger_upper": float(tech.bollinger_upper),
#                     "bollinger_lower": float(tech.bollinger_lower),
#                     "support": float(tech.support),
#                     "resistance": float(tech.resistance)
#                 })
            
#             # Show other agents
#             if 'analysis_result' in st.session_state:
#                 st.subheader("Other Agents")
#                 st.json(st.session_state.analysis_result['recommendations'])
    
#     else:
#         st.info("⚙️ Configure date range and click **Run AI Agents** to start the multi-agent analysis")
        
#         with st.expander("📖 How to use AI Agents"):
#             st.markdown("""
#             ### New Features:
            
#             ✨ **Autonomous Technical Agent** - No LLM dependency!
#             - Uses MACD crossover detection (35% weight)
#             - Bollinger Bands bounce signals (35% weight)
#             - Loss Following Strategy (20% weight)
#             - RSI extremes confirmation (10% weight)
            
#             ### Steps:
            
#             1. **Select Date Range** - Choose start and end dates for analysis
#             2. **Click "Run AI Agents"** - Triggers the multi-agent system
#             3. **View Agent Activity** - Watch as 5 specialized agents analyze
#             4. **Review Final Decision** - See master agent's recommendation
#             5. **Execute Trade** - If you agree, click the execute button
#             6. **View Portfolio** - Check your updated holdings
            
#             **Note:** Technical agent makes autonomous decisions based on pure logic!
#             """)
# ui/ai_agents.py - UPDATED WITH "New News" SECTION

# import streamlit as st
# from datetime import datetime, timedelta
# import time
# from main import get_trading_system, fetch_and_enhance
# from agents.technical_agent import TechnicalAnalysisAgent
# from agents.sentiment_agent import SentimentAnalysisAgent
# from agents.risk_agent import RiskManagementAgent
# from agents.portfolio_agent import PortfolioManagerAgent
# from agents.master_agent import MasterAgent
# from agents.debate_agent import DebateAgent
# from agents.data_collection_agent import DataCollectionAgent
# from data.news_fetcher import NewsFetcher
# # 🆕 New imports
# from agents.new_news_agent import NewNews       # make sure this file exists
# from trading_bot.data.new_news_fetcher import NewNewsFetcher  # make sure this file exists
# import logging
# import os

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
#         st.markdown(f"<h1 style='text-align:center; color:#00b894;'>🤖 Multi-Agent Analysis</h1>", unsafe_allow_html=True)
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
#         # 🆕 New News components
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

#             # 🆕 Step 1.1: Fetch New News articles (Tavily + NewsAPI)
#             # Using ticker as company_name fallback (replace with actual company name if you have it)
#             status_container.markdown("""
#             <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #0984e3;'>
#                 <strong style='color:#74b9ff;'>🗞️ Fetching New News...</strong>
#             </div>
#             """, unsafe_allow_html=True)
#             progress_bar.progress(20)

#             new_news_fetcher = st.session_state.new_news_fetcher
#             new_news_articles = new_news_fetcher.get_news(
#                 company_name=ticker,  # swap with real company name if available
#                 ticker=ticker,
#                 max_results=16,
#                 days=7,
#                 dedupe=True
#             )
#             st.session_state['new_news_articles'] = new_news_articles  # persist across reruns

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

#             # 🆕 Step 3.1: New News Agent (deterministic + Groq-ready)
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
#             # 🆕 Added "🗞️ New News" tab
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

# ui/ai_agents.py - UPDATED WITH "New News" TAB

import os
import logging
import time
from datetime import datetime, timedelta
import streamlit as st

from main import get_trading_system, fetch_and_enhance
from agents.technical_agent import TechnicalAnalysisAgent
from agents.sentiment_agent import SentimentAnalysisAgent
from agents.risk_agent import RiskManagementAgent
from agents.portfolio_agent import PortfolioManagerAgent
from agents.master_agent import MasterAgent
from agents.debate_agent import DebateAgent
from agents.data_collection_agent import DataCollectionAgent

# Legacy fetcher (keep for compatibility); provide a safe fallback if missing
try:
    from data.news_fetcher import NewsFetcher
except Exception:
    class NewsFetcher:
        def get_stock_news(self, ticker, company_name, max_results=10):
            return []

# New News imports (make sure these files exist)
from agents.new_news_agent import NewNews
from data.new_news_fetcher import NewNewsFetcher

logger = logging.getLogger(__name__)


def show_ai_agents_page():
    """Display AI agents analysis page"""
    if not st.session_state.get('logged_in'):
        st.error("⚠️ Please login first")
        if st.button("Go to Login"):
            st.session_state.page = 'login'
            st.rerun()
        return
    
    ticker = st.session_state.get('ticker', 'RELIANCE.NS')
    
    # Header
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 'analysis'
            st.rerun()
    with col2:
        st.markdown("<h1 style='text-align:center; color:#00b894;'>🤖 Multi-Agent Analysis</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>{ticker}</p>", unsafe_allow_html=True)
    
    # Initialize agents
    if 'agents_initialized' not in st.session_state:
        st.session_state.tech_agent = TechnicalAnalysisAgent()
        st.session_state.sentiment_agent = SentimentAnalysisAgent()
        st.session_state.risk_agent = RiskManagementAgent()
        st.session_state.portfolio_agent = PortfolioManagerAgent()
        st.session_state.master_agent = MasterAgent()
        st.session_state.debate_agent = DebateAgent()
        st.session_state.data_collector = DataCollectionAgent()
        st.session_state.news_fetcher = NewsFetcher()
        # New News components
        st.session_state.new_news_fetcher = NewNewsFetcher()
        st.session_state.new_news_agent = NewNews(groq_api_key=os.getenv("GROQ_API_KEY"))
        st.session_state.agents_initialized = True
    
    # Date range
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])
    with col1:
        start_date_input = st.date_input("Start Date", value=datetime.now() - timedelta(days=180), key="start_date_key")
        start_date = datetime.combine(start_date_input, datetime.min.time())
    with col2:
        end_date_input = st.date_input("End Date", value=datetime.now(), key="end_date_key")
        end_date = datetime.combine(end_date_input, datetime.max.time())
    with col3:
        show_debate = st.checkbox("Show Debate", value=True)
    with col4:
        run_analysis = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    # Main analysis
    if run_analysis:
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        try:
            # Step 1: Collect Data
            status_container.markdown("""
            <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #00b894;'>
                <strong style='color:#00b894;'>📊 Collecting Data...</strong>
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(10)
            
            data_collector = st.session_state.data_collector
            collection = data_collector.collect(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                include_news=True,
                include_fundamentals=True
            )
            
            tech_data = collection.get('technical_data', {})
            news_data = collection.get('news_data', [])
            
            if tech_data.get('status') != 'SUCCESS':
                st.error(f"❌ Failed to fetch data: {tech_data.get('error', 'Unknown error')}")
                return
            
            price_df = tech_data.get('dataframe')
            latest_close = tech_data.get('latest_close', 0)

            # Step 1.1: Fetch New News articles (Tavily + NewsAPI)
            status_container.markdown("""
            <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #0984e3;'>
                <strong style='color:#74b9ff;'>🗞️ Fetching New News...</strong>
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(20)

            new_news_fetcher = st.session_state.new_news_fetcher
            new_news_articles = new_news_fetcher.get_news(
                company_name=ticker,  # replace with actual company name if available
                ticker=ticker,
                max_results=16,
                days=7,
                dedupe=True
            )
            st.session_state['new_news_articles'] = new_news_articles  # persist

            # Step 2: Technical Analysis
            status_container.markdown("""
            <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #00b894;'>
                <strong style='color:#00b894;'>📈 Technical Agent...</strong>
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(30)
            time.sleep(0.3)
            
            tech_agent = st.session_state.tech_agent
            tech_result = tech_agent.analyze(ticker, start_date, end_date)
            tech_action = "BUY" if "BUY" in tech_result.recommendation else ("SELL" if "SELL" in tech_result.recommendation else "HOLD")
            tech_confidence = tech_result.overall_confidence if hasattr(tech_result, 'overall_confidence') else 50
            
            # Step 3: Sentiment Analysis (legacy)
            status_container.markdown("""
            <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #6c5ce7;'>
                <strong style='color:#6c5ce7;'>📰 Sentiment Agent...</strong>
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(45)
            time.sleep(0.3)
            
            sentiment_agent = st.session_state.sentiment_agent
            if not news_data:
                news_fetcher = st.session_state.news_fetcher
                news_data = news_fetcher.get_stock_news(ticker, ticker, max_results=10)
            sentiment_result = sentiment_agent.analyze(ticker=ticker, news_data=news_data if news_data else [])
            sent_action = sentiment_result.get('overall_sentiment', 'neutral')
            sent_confidence = sentiment_result.get('overall_confidence', 50)

            # Step 3.1: New News Agent (deterministic + Groq-ready)
            status_container.markdown("""
            <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #00cec9;'>
                <strong style='color:#00cec9;'>🤖 New News Agent...</strong>
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(55)
            time.sleep(0.2)

            new_news_agent = st.session_state.new_news_agent
            new_news_agent.set_context(ticker=ticker, company_name=ticker, news_data=new_news_articles or [])
            new_news_result = new_news_agent.analyze(ticker=ticker, news_data=new_news_articles or [])
            st.session_state['new_news_result'] = new_news_result

            # Step 4: Risk Assessment
            status_container.markdown("""
            <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #e17055;'>
                <strong style='color:#e17055;'>⚠️ Risk Agent...</strong>
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(60)
            time.sleep(0.3)
            
            risk_agent = st.session_state.risk_agent
            risk_metrics = risk_agent.evaluate(
                ticker=ticker,
                df=price_df,
                current_price=latest_close,
                technical_confidence=tech_confidence,
                sentiment_confidence=sent_confidence
            )
            risk_dict = risk_metrics.dict() if hasattr(risk_metrics, 'dict') else dict(risk_metrics) if not isinstance(risk_metrics, dict) else risk_metrics
            risk_level = risk_dict.get('risk_level', 'MEDIUM')
            
            # Step 5: Portfolio Allocation
            status_container.markdown("""
            <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #00cec9;'>
                <strong style='color:#00cec9;'>💼 Portfolio Agent...</strong>
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(75)
            time.sleep(0.3)
            
            portfolio_agent = st.session_state.portfolio_agent
            portfolio_action, portfolio_qty, portfolio_meta = portfolio_agent.decide(
                ticker=ticker,
                current_price=latest_close,
                technical_signal={'recommendation': tech_result.recommendation, 'action': tech_action},
                sentiment_signal=sentiment_result,
                risk_metrics=risk_dict
            )
            
            # Step 6: Debate Analysis
            debate_result = None
            if show_debate:
                status_container.markdown("""
                <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #fd79a8;'>
                    <strong style='color:#fd79a8;'>🐂🐻 Debate Agent...</strong>
                </div>
                """, unsafe_allow_html=True)
                progress_bar.progress(85)
                time.sleep(0.3)
                
                debate_agent = st.session_state.debate_agent
                debate_result = debate_agent.debate(
                    ticker=ticker,
                    technical_result=dict(tech_result) if hasattr(tech_result, '__dict__') else tech_result,
                    risk_metrics=risk_dict,
                    price_data=price_df,
                    sentiment_score=sentiment_result.get('overall_score', 50)
                )
            
            # Step 7: Master Decision
            status_container.markdown("""
            <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd93d;'>
                <strong style='color:#ffd93d;'>🎯 Master Agent...</strong>
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(95)
            time.sleep(0.3)
            
            master_agent = st.session_state.master_agent
            master_result = master_agent.synthesize(
                ticker=ticker,
                technical_result=dict(tech_result) if hasattr(tech_result, '__dict__') else tech_result,
                sentiment_result=sentiment_result,
                risk_metrics=risk_dict,
                portfolio_metrics={'quantity': portfolio_qty},
                current_price=latest_close
            )
            master_dict = master_result.dict() if hasattr(master_result, 'dict') else dict(master_result) if not isinstance(master_result, dict) else master_result
            master_action = master_dict.get('action', 'HOLD')
            master_conf = master_dict.get('confidence', 50)
            
            progress_bar.progress(100)
            time.sleep(0.2)
            progress_bar.empty()
            status_container.empty()
            
            st.success("✅ Analysis Complete!")
            
            # Display results
            tab_all, tab_master, tab_debate, tab_details, tab_newnews = st.tabs(
                ["🤖 All Agents", "🎯 Master Decision", "🐂🐻 Debate", "📊 Details", "🗞️ New News"]
            )
            
            # TAB 1: ALL AGENTS
            with tab_all:
                st.markdown("## Individual Agent Recommendations")
                col1, col2 = st.columns(2)
                
                # Technical
                with col1:
                    action_color = "#00b894" if "BUY" in tech_action else "#ff7675" if "SELL" in tech_action else "#fdcb6e"
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {action_color};'>
                        <h3 style='color:{action_color}; margin-top:0;'>📊 Technical Agent</h3>
                        <p><strong>Signal:</strong> {tech_action}</p>
                        <p><strong>Confidence:</strong> {tech_confidence:.1f}%</p>
                        <hr style='margin: 10px 0; opacity: 0.3;'>
                        <p><strong>Key Indicators:</strong></p>
                        <ul style='margin: 10px 0; padding-left: 20px;'>
                            <li>MACD: {tech_result.macd:.4f}</li>
                            <li>RSI: {tech_result.rsi:.2f}</li>
                            <li>BB Upper: {tech_result.bollinger_upper:.2f}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Sentiment (legacy)
                with col2:
                    sent_color = "#00b894" if sent_action == "positive" else "#ff7675" if sent_action == "negative" else "#fdcb6e"
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {sent_color};'>
                        <h3 style='color:{sent_color}; margin-top:0;'>📰 Sentiment Agent</h3>
                        <p><strong>Sentiment:</strong> {sent_action.upper()}</p>
                        <p><strong>Confidence:</strong> {sent_confidence:.1f}%</p>
                        <p><strong>Summary:</strong></p>
                        <p style='font-size: 0.9em; color: #b2bec3;'>{sentiment_result.get('summary', 'No summary')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                col3, col4 = st.columns(2)
                
                # Risk
                with col3:
                    risk_color = "#00b894" if risk_level == "LOW" else "#fdcb6e" if risk_level == "MEDIUM" else "#ff7675"
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {risk_color};'>
                        <h3 style='color:{risk_color}; margin-top:0;'>⚠️ Risk Agent</h3>
                        <p><strong>Risk Level:</strong> {risk_level}</p>
                        <p><strong>Volatility:</strong> {risk_dict.get('volatility', 0):.2%}</p>
                        <p><strong>Position Size:</strong> {risk_dict.get('position_size', 0):.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Portfolio
                with col4:
                    port_color = "#00b894" if portfolio_action == "BUY" else "#ff7675" if portfolio_action == "SELL" else "#fdcb6e"
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {port_color};'>
                        <h3 style='color:{port_color}; margin-top:0;'>💼 Portfolio Agent</h3>
                        <p><strong>Action:</strong> {portfolio_action}</p>
                        <p><strong>Quantity:</strong> {portfolio_qty} shares</p>
                        <p><strong>Allocation:</strong> ₹{portfolio_qty * latest_close:,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # TAB 2: MASTER DECISION
            with tab_master:
                st.markdown("## 🎯 Master Agent Final Decision")
                master_color = "#00b894" if master_action == "BUY" else "#ff7675" if master_action == "SELL" else "#fdcb6e"
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {master_color}22, {master_color}11); padding: 40px; border-radius: 16px; border: 3px solid {master_color};'>
                        <h1 style='color:{master_color}; margin:0; font-size:3rem; text-align:center;'>{master_action}</h1>
                        <p style='color:#b2bec3; text-align:center; margin:10px 0 0 0; font-size:1.2rem;'>Master Recommendation</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.metric("Confidence", f"{master_conf:.1f}%")
                    st.metric("Quantity", f"{portfolio_qty}")
                with col3:
                    st.metric("Risk Level", risk_level)
                    st.metric("Entry Price", f"₹{latest_close:.2f}")
                st.markdown("---")
                st.markdown("### 📝 Master Agent Reasoning")
                st.info(master_dict.get('reasoning', 'No reasoning'))
            
            # TAB 3: DEBATE
            if show_debate and debate_result:
                with tab_debate:
                    st.markdown("## 🐂 vs 🐻 Bull vs Bear Debate")
                    debate_dict = debate_result.dict() if hasattr(debate_result, 'dict') else dict(debate_result) if not isinstance(debate_result, dict) else debate_result
                    col1, col2 = st.columns(2)
                    with col1:
                        bull = debate_dict.get('bull_case', {})
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid #00b894;'>
                            <h3 style='color:#00b894; margin-top:0;'>🐂 BULL CASE</h3>
                            <p><strong>Strength:</strong> {bull.get('strength', 0):.0f}/100</p>
                            <p><strong>Arguments:</strong></p>
                            <ul style='margin: 10px 0; padding-left: 20px;'>
                        """
                        + "".join([f"<li>{arg}</li>" for arg in bull.get('arguments', [])[:5]])
                        + """
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        bear = debate_dict.get('bear_case', {})
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid #ff7675;'>
                            <h3 style='color:#ff7675; margin-top:0;'>🐻 BEAR CASE</h3>
                            <p><strong>Strength:</strong> {bear.get('strength', 0):.0f}/100</p>
                            <p><strong>Arguments:</strong></p>
                            <ul style='margin: 10px 0; padding-left: 20px;'>
                        """
                        + "".join([f"<li>{arg}</li>" for arg in bear.get('arguments', [])[:5]])
                        + """
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
            
            # TAB 4: DETAILS
            with tab_details:
                st.markdown("## 📊 Detailed Data")
                subtab1, subtab2, subtab3 = st.tabs(["Technical", "Sentiment", "Risk"])
                with subtab1:
                    st.json(dict(tech_result) if hasattr(tech_result, '__dict__') else tech_result)
                with subtab2:
                    st.json(sentiment_result)
                with subtab3:
                    st.json(risk_dict)

            # TAB 5: NEW NEWS (Groq + Chat)
            with tab_newnews:
                st.markdown("## 🗞️ New News (Tavily + NewsAPI + Groq)")
                
                new_news_result = st.session_state.get('new_news_result', {})
                new_news_articles = st.session_state.get('new_news_articles', [])

                # Cards: Deterministic sentiment
                nn_sent = new_news_result.get('overall_sentiment', 'neutral')
                nn_conf = new_news_result.get('overall_confidence', 0)
                nn_score = new_news_result.get('overall_score', 0.0)
                nn_color = "#00b894" if nn_sent == "positive" else "#ff7675" if nn_sent == "negative" else "#fdcb6e"

                colA, colB = st.columns([2, 3])
                with colA:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {nn_color};'>
                        <h3 style='color:{nn_color}; margin-top:0;'>🤖 New News Sentiment</h3>
                        <p><strong>Sentiment:</strong> {nn_sent.upper()}</p>
                        <p><strong>Score:</strong> {nn_score:.3f}</p>
                        <p><strong>Confidence:</strong> {nn_conf:.1f}%</p>
                        <p><strong>Summary:</strong></p>
                        <p style='font-size: 0.9em; color: #b2bec3;'>{new_news_result.get('summary', 'No summary')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with colB:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid #0984e3;'>
                        <h3 style='color:#74b9ff; margin-top:0;'>📰 Latest Articles</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    if new_news_articles:
                        for i, a in enumerate(new_news_articles[:10], start=1):
                            title = a.get("title", "N/A")
                            src = a.get("source", "Unknown")
                            date = a.get("publishedAt", "Unknown")
                            url = a.get("url", "")
                            st.markdown(f"- [{title}]({url}) — {src} • {date}")
                    else:
                        st.warning("No articles found for New News.")

                st.markdown("---")

                # Groq Executive Summary
                groq_ready = getattr(st.session_state.new_news_agent, "client", None) is not None
                col_groq1, col_groq2 = st.columns([1, 3])
                with col_groq1:
                    gen_summary = st.button("⚡ Generate Groq Executive Summary", use_container_width=True)
                with col_groq2:
                    if not groq_ready:
                        st.info("Set GROQ_API_KEY to enable AI summary and chat.")
                if gen_summary and groq_ready:
                    with st.spinner("Analyzing with Groq..."):
                        summary = st.session_state.new_news_agent.groq_summary(top_k=10)
                        st.session_state['new_news_groq_summary'] = summary or "No AI summary available."
                summary_text = st.session_state.get('new_news_groq_summary')
                if summary_text:
                    st.markdown("### 🧠 AI Executive Summary")
                    st.markdown(summary_text)

                st.markdown("---")

                # Grounded Chatbot
                st.markdown("### 💬 New News Chatbot (Grounded with Citations)")
                chat_q = st.text_input("Ask about catalysts, risks, revenue impact, etc.", key=f"nn_chat_{ticker}")
                colAsk1, colAsk2 = st.columns([1, 3])
                with colAsk1:
                    ask = st.button("Ask", key=f"nn_chat_btn_{ticker}", use_container_width=True)
                with colAsk2:
                    if not groq_ready:
                        st.info("Groq not configured.")
                if ask and chat_q and groq_ready:
                    with st.spinner("Thinking..."):
                        resp = st.session_state.new_news_agent.chat(chat_q, max_articles=12, temperature=0.2, return_citations=True)
                        st.session_state['nn_last_answer'] = resp

                resp = st.session_state.get('nn_last_answer')
                if resp:
                    st.markdown("#### Answer")
                    st.markdown(resp.get("answer", ""))
                    cites = resp.get("citations", [])
                    if cites:
                        st.markdown("#### Citations")
                        for c in cites:
                            st.markdown(f"- [{c['index']}] [{c['title']}]({c['url']}) — {c.get('source','Unknown')} • {c.get('publishedAt','Unknown')}")
            
            # Trade execution
            st.markdown("---")
            st.markdown("## 💼 Trade Execution")
            if master_action != "HOLD":
                col1, col2 = st.columns([2, 1])
                with col1:
                    trade_qty = st.number_input("Quantity", min_value=1, value=portfolio_qty, step=1, key=f"qty_{ticker}")
                    st.info(f"💰 Value: ₹{trade_qty * latest_close:,.2f}")
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button(f"✅ Execute {master_action}", type="primary", use_container_width=True, key=f"exe_{ticker}"):
                        if not st.session_state.get('user_id'):
                            st.error("❌ Not logged in!")
                            st.stop()
                        try:
                            from models.database import get_database
                            db = get_database()
                            agent_rec = f"{master_action} - Tech: {tech_action} | Sentiment: {sent_action} | Risk: {risk_level}"
                            with st.spinner("Executing..."):
                                success, message = db.execute_trade(
                                    user_id=st.session_state.user_id,
                                    ticker=ticker,
                                    action=master_action,
                                    quantity=trade_qty,
                                    price=latest_close,
                                    agent_rec=agent_rec
                                )
                            if success:
                                st.balloons()
                                st.success(f"✅ {message}")
                                time.sleep(2)
                                st.session_state.page = 'portfolio'
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            logger.error(f"Trade execution error: {e}")
            else:
                st.warning("⏸️ Master recommends HOLD")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Analysis error: {e}")
            import traceback
            with st.expander("🔍 Debug Info"):
                st.code(traceback.format_exc())
    else:
        st.info("👆 Select dates and click Analyze")