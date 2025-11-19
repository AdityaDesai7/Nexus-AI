# app.py - Enhanced Streamlit frontend with Institutional Analysis (ALL 13 AGENTS)
import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta
import time  # ✅ ADD THIS
from main import fetch_and_enhance  # ✅ ADD THIS


# Load environment variables
load_dotenv()

# Check for required API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("⚠️ Missing API Keys! Please add GROQ_API_KEY and TAVILY_API_KEY to your .env file")
    st.stop()

# Import main analysis system
from main import get_trading_system, analyze_stock, get_detailed_analysis

# Page config
st.set_page_config(
    page_title="ProTrader AI - 13 Institutional Agents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 100%);
        color: #fafafa;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00b894, #00d4aa);
        color: white;
        border-radius: 12px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0, 184, 148, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'ticker' not in st.session_state:
    st.session_state.ticker = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Sidebar with user info (if logged in)
if st.session_state.logged_in:
    with st.sidebar:
        if st.session_state.is_admin:
            st.markdown(f"### 👨‍💼 {st.session_state.username} (Admin)")
        else:
            st.markdown(f"### 👤 Welcome, {st.session_state.username}!")
        
        st.markdown("---")
        st.markdown("### 📍 Navigation")
        
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        
        if st.button("📊 Stock Analysis", use_container_width=True):
            st.session_state.page = 'analysis'
            st.rerun()
        
        if st.button("🏦 Institutional Analysis", use_container_width=True):
            st.session_state.page = 'institutional'
            st.rerun()
        
        if st.button("💼 My Portfolio", use_container_width=True):
            st.session_state.page = 'portfolio'
            st.rerun()
        
        if st.button("🤖 AI Agents", use_container_width=True):
            st.session_state.page = 'ai_agents'
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📚 Education")
        
        if st.button("📖 Learn About Agents", use_container_width=True):
            st.session_state.page = 'education'
            st.rerun()
        
        if st.session_state.is_admin:
            st.markdown("---")
            st.markdown("### 👨‍💼 Admin")
            if st.button("Admin Dashboard", use_container_width=True, type="primary"):
                st.session_state.page = 'admin'
                st.rerun()
        
        st.markdown("---")
        
        if st.button("🔐 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.is_admin = False
            st.session_state.page = 'login'
            st.rerun()

# Import page modules
from ui.login import show_login_page
from ui.home import show_home_page
from ui.stock_analysis import show_stock_analysis
from ui.ai_agents import show_ai_agents_page
from ui.portfolio import show_portfolio_page
from ui.admin import show_admin_dashboard


# ============================================
# PAGE 1: EDUCATION - Learn About All 13 Agents
# ============================================

def show_agent_education_page():
    """Show educational guide for all 13 agents"""
    
    from agents.agents_educational_guide import AgentEducationalGuide
    
    st.markdown("# 📚 Understanding the 13 Institutional Agents")
    st.markdown("**Complete Educational Guide for Investors - Learn How Each Agent Works**")
    
    agents = AgentEducationalGuide.get_all_agents_explanation()
    
    # Create a selector
    agent_num = st.selectbox(
        "📌 Select an Agent to Learn About:",
        options=list(agents.keys()),
        format_func=lambda x: f"Agent {x}: {agents[x]['name']}"
    )
    
    agent = agents[agent_num]
    
    # Display
    st.markdown(f"## {agent['name']}")
    st.markdown(f"**Category:** `{agent['category']}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📖 What It Does")
        st.info(agent['what_it_does'])
        
        st.markdown("### 🎯 Real-World Example")
        st.success(agent['real_world_example'])
    
    with col2:
        st.markdown("### 📥 Inputs (Data It Uses)")
        for input_name, input_desc in agent['inputs'].items():
            st.markdown(f"**{input_name}**")
            st.write(input_desc)
            st.markdown("")
    
    st.markdown("---")
    
    st.markdown("### 🧠 Logic (How It Decides)")
    st.code(agent['logic'], language="text")
    
    st.markdown("---")
    
    st.markdown("### 📤 Output Example (Real Numbers)")
    st.success(agent['output_example'])
    
    # Show all agents list
    st.markdown("---")
    st.markdown("## 📋 All 13 Agents Overview")
    
    agent_list = []
    for i, data in agents.items():
        agent_list.append({
            "Agent": f"**{i}. {data['name']}**",
            "Type": data['category']
        })
    
    df = pd.DataFrame(agent_list)
    st.table(df)


# ============================================
# PAGE 2: INSTITUTIONAL ANALYSIS - Real-time Analysis
# ============================================

def show_institutional_analysis_page():
    """Display 9 Institutional Agents Analysis - MINIMAL UPDATE"""
    
    st.markdown("# 🏛️ 9 Institutional Agents")
    st.markdown("Real-time institutional investor detection")
    
    ticker = st.text_input("Stock Ticker:", value=st.session_state.get('ticker', 'RELIANCE.NS'))
    days = st.slider("Days:", 30, 365, 180)
    analyze_btn = st.button("🚀 Analyze 9 Agents", type="primary", use_container_width=True)
    
    if analyze_btn:
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        try:
            status_container.markdown("""
            <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #9b59b6;'>
                <strong style='color:#9b59b6;'>📊 Loading market data...</strong>
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(10)
            
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
            
            df = fetch_and_enhance(ticker, start_date, end_date)
            
            if df is None or len(df) == 0:
                st.error("❌ Failed to fetch data")
                return
            
            progress_bar.progress(40)
            
            status_container.markdown("""
            <div style='background: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid #9b59b6;'>
                <strong style='color:#9b59b6;'>🤖 Running 9 institutional agents...</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize 9 agents
            from agents.institutional_agents import MasterInstitutionalAggregator
            
            master_agg = MasterInstitutionalAggregator()
            master_result = master_agg.aggregate(
                fii_data={'today': 100, '30_day_avg': 80},
                order_data=[],
                price_data=df['Close'],
                volume_data=df['Volume']
            )
            
            progress_bar.progress(90)
            st.success("✅ Analysis Complete!")
            time.sleep(0.2)
            progress_bar.empty()
            status_container.empty()
            
            # Display 9 Agents in 3x3 Grid
            st.markdown("## 9 Individual Agents")
            
            agent_info = {
                'fii': {'emoji': '💰', 'color': '#e74c3c'},
                'execution': {'emoji': '⚡', 'color': '#f39c12'},
                'volume': {'emoji': '📈', 'color': '#3498db'},
                'ifi': {'emoji': '📊', 'color': '#2ecc71'},
                'accumulation': {'emoji': '🔝', 'color': '#9b59b6'},
                'liquidity': {'emoji': '💧', 'color': '#1abc9c'},
                'smart_money': {'emoji': '🧠', 'color': '#34495e'},
                'block_orders': {'emoji': '📦', 'color': '#16a085'},
                'breakout': {'emoji': '🚀', 'color': '#c0392b'},
            }
            
            cols = st.columns(3)
            
            for idx, (agent_name, signal) in enumerate(master_result.agent_breakdown.items()):
                col_idx = idx % 3
                
                if col_idx == 0 and idx != 0:
                    cols = st.columns(3)
                
                if signal:
                    info = agent_info.get(agent_name, {'emoji': '🤖', 'color': '#95a5a6'})
                    action_color = "#00b894" if "BUY" in signal.action else "#ff7675" if "SELL" in signal.action else "#fdcb6e"
                    
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #1e1e2e, #2a2a3e); padding: 20px; border-radius: 12px; border-left: 4px solid {info["color"]};'>
                            <h4 style='color:{info["color"]}; margin-top:0;'>{info["emoji"]} {agent_name.replace('_', ' ').title()}</h4>
                            <p><strong>Vote:</strong> <span style='color:{action_color};'>{signal.action}</span></p>
                            <p><strong>Score:</strong> {signal.score:.1f}</p>
                            <p><strong>Confidence:</strong> {signal.confidence:.0f}%</p>
                            <hr style='margin: 10px 0; opacity: 0.3;'>
                            <p style='font-size: 0.85em; color: #b2bec3;'>{signal.reason}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Master Decision
            st.markdown("---")
            st.markdown("## 🎯 Master Consensus")
            
            master_color = "#00b894" if "BUY" in master_result.recommendation else "#ff7675" if "SELL" in master_result.recommendation else "#fdcb6e"
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {master_color}22, {master_color}11); padding: 40px; border-radius: 16px; border: 3px solid {master_color};'>
                    <h1 style='color:{master_color}; margin:0; font-size:3rem; text-align:center;'>{master_result.recommendation}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Score", f"{master_result.final_score:.1f}/100")
                st.metric("Confidence", f"{master_result.confidence:.1f}%")
            
            st.markdown("---")
            st.info(f"**Reasoning:** {master_result.reasoning}")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Debug"):
                st.code(traceback.format_exc())


# ============================================
# NAVIGATION LOGIC
# ============================================

if not st.session_state.logged_in:
    show_login_page()
else:
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'analysis':
        show_stock_analysis()
    elif st.session_state.page == 'institutional':
        show_institutional_analysis_page()
    elif st.session_state.page == 'education':
        show_agent_education_page()
    elif st.session_state.page == 'ai_agents':
        show_ai_agents_page()
    elif st.session_state.page == 'portfolio':
        show_portfolio_page()
    elif st.session_state.page == 'admin':
        show_admin_dashboard()
