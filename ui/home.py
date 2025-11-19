# ui/home.py
import streamlit as st

def show_home_page():
    """Display the home page with basic info and ticker input"""
    
    # Hero Section
    st.markdown("""
    <div style='text-align:center; margin-top:50px;'>
        <img src='https://img.icons8.com/fluency/96/artificial-intelligence.png' width='120'/>
        <h1 style='color:#00b894; font-size:3.5rem; margin-top:20px;'>ProTrader AI</h1>
        <p style='color:#888; font-size:1.3rem; margin-top:10px;'>
            Multi-Agent Trading Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color:#00b894;'>📊 Deep Analysis</h3>
            <p style='color:#b2bec3; margin-top:10px;'>
                Comprehensive technical & fundamental analysis with real-time data from Yahoo Finance
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color:#6c5ce7;'>📰 Market News</h3>
            <p style='color:#b2bec3; margin-top:10px;'>
                Latest news and sentiment analysis powered by AI to keep you informed
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color:#e17055;'>🤖 AI Agents</h3>
            <p style='color:#b2bec3; margin-top:10px;'>
                5 specialized AI agents working together to provide trading recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stock Input Section
    st.markdown("<h2 style='text-align:center; color:#00b894; margin-top:60px;'>Get Started</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888; margin-bottom:30px;'>Enter a stock ticker to begin your analysis</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        ticker = st.text_input(
            "",
            placeholder="Enter Indian Stock Ticker (e.g., RELIANCE.NS, TCS.NS, INFY.NS)",
            key="ticker_input",
            label_visibility="collapsed"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🚀 Analyze Stock", type="primary", use_container_width=True):
                if ticker and len(ticker.strip()) > 0:
                    st.session_state.ticker = ticker.strip().upper()
                    st.session_state.page = 'analysis'
                    st.rerun()
                else:
                    st.error("⚠️ Please enter a valid stock ticker")
    
    st.markdown("---")
    
    # How It Works
    st.markdown("## 🎯 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Stock Analysis
        - **Technical Indicators**: RSI, MACD, Bollinger Bands, EMAs, and more
        - **Fundamental Data**: Financial ratios, income statements, balance sheets, cash flows
        - **Valuation Models**: DCF, DDM, FCFE, FCFF, and relative valuations
        - **Historical Trends**: Multi-year performance analysis
        """)
        
        st.markdown("""
        ### 2️⃣ Market Intelligence
        - **Real-time News**: Latest news articles about your stock
        - **Sentiment Analysis**: AI-powered sentiment from news and market data
        - **Analyst Ratings**: Price targets and recommendations
        """)
    
    with col2:
        st.markdown("""
        ### 3️⃣ AI Agent System (Optional)
        Our multi-agent system provides intelligent trading recommendations:
        
        - **📊 Technical Agent**: Analyzes chart patterns and indicators
        - **📰 Sentiment Agent**: Evaluates market sentiment and momentum
        - **⚠️ Risk Agent**: Assesses volatility and risk metrics
        - **💼 Portfolio Agent**: Optimizes position sizing
        - **🎯 Master Agent**: Synthesizes all recommendations
        
        *Paper trading with ₹10,00,000 virtual capital*
        """)
    
    st.markdown("---")
    
    # Sample Tickers
    st.markdown("### 💡 Popular Indian Stocks to Try")
    
    sample_tickers = [
        ("RELIANCE.NS", "Reliance Industries"),
        ("TCS.NS", "Tata Consultancy Services"),
        ("INFY.NS", "Infosys"),
        ("HDFCBANK.NS", "HDFC Bank"),
        ("ICICIBANK.NS", "ICICI Bank"),
        ("NTPC.NS", "NTPC Limited"),
        ("BHARTIARTL.NS", "Bharti Airtel"),
        ("ITC.NS", "ITC Limited"),
    ]
    
    cols = st.columns(4)
    for idx, (ticker, name) in enumerate(sample_tickers):
        with cols[idx % 4]:
            if st.button(f"{ticker}\n{name}", key=f"sample_{ticker}", use_container_width=True):
                st.session_state.ticker = ticker
                st.session_state.page = 'analysis'
                st.rerun()
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align:center; color:#666; margin-top:60px; margin-bottom:20px;'>
        <p>Built with ❤️ using Streamlit, yfinance, Groq AI, and Tavily</p>
        <p style='font-size:0.9rem; margin-top:10px;'>
            ⚠️ For educational purposes only. Not financial advice.
        </p>
    </div>
    """, unsafe_allow_html=True)
