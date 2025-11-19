# ui/portfolio.py
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from models.database import get_database
import pandas as pd

def show_portfolio_page():
    """Display user's portfolio"""
    
    if not st.session_state.get('logged_in'):
        st.warning("Please login to view your portfolio")
        return
    
    user_id = st.session_state.user_id
    username = st.session_state.username
    db = get_database()
    
    # Header
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("← Home"):
            st.session_state.page = 'home'
            st.rerun()
    with col2:
        st.markdown(f"<h1 style='text-align:center; color:#00b894;'>💼 {username}'s Portfolio</h1>", unsafe_allow_html=True)
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Get portfolio data
    with st.spinner("Loading portfolio data..."):
        portfolio_df = db.get_portfolio(user_id)
        balance = db.get_user_balance(user_id)
    
    # Update portfolio with current prices
    current_prices = {}
    if not portfolio_df.empty:
        with st.spinner("Fetching current market prices..."):
            for ticker in portfolio_df['ticker']:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    # Try multiple price fields
                    price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
                    current_prices[ticker] = price
                except Exception as e:
                    st.warning(f"Could not fetch price for {ticker}: {str(e)}")
                    current_prices[ticker] = 0
        
        # Update database with current prices
        db.update_portfolio_values(user_id, current_prices)
        
        # Refresh stats after updating prices
        stats = db.get_user_stats(user_id)
    else:
        stats = db.get_user_stats(user_id)
    
    # Portfolio Summary Cards
    st.markdown("## 💰 Portfolio Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        pl_delta = stats['profit_loss_pct']
        delta_color = "normal" if pl_delta >= 0 else "inverse"
        st.metric(
            "Total Value",
            f"₹{stats['total_value']:,.2f}",
            delta=f"{stats['profit_loss_pct']:+.2f}%",
            delta_color=delta_color
        )
    
    with col2:
        st.metric("Available Cash", f"₹{stats['available_cash']:,.2f}")
    
    with col3:
        st.metric("Invested", f"₹{stats['invested_amount']:,.2f}")
    
    with col4:
        st.metric("Portfolio Value", f"₹{stats['portfolio_value']:,.2f}")
    
    with col5:
        pl_color = "🟢" if stats['profit_loss'] > 0 else "🔴" if stats['profit_loss'] < 0 else "⚪"
        st.metric(
            "P&L",
            f"₹{stats['profit_loss']:,.2f}",
            delta=f"{pl_color}"
        )
    
    st.markdown("---")
    
    # Portfolio visualization
    if not portfolio_df.empty and len(current_prices) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio allocation pie chart
            portfolio_df['current_value'] = portfolio_df.apply(
                lambda row: row['quantity'] * current_prices.get(row['ticker'], row['avg_price']),
                axis=1
            )
            
            fig_pie = px.pie(
                portfolio_df,
                values='current_value',
                names='ticker',
                title='Portfolio Allocation by Value',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # P&L by stock
            portfolio_df['pl'] = portfolio_df.apply(
                lambda row: (current_prices.get(row['ticker'], row['avg_price']) * row['quantity']) - row['total_invested'],
                axis=1
            )
            portfolio_df['pl_pct'] = (portfolio_df['pl'] / portfolio_df['total_invested']) * 100
            
            fig_bar = px.bar(
                portfolio_df,
                x='ticker',
                y='pl',
                title='Profit/Loss by Stock',
                color='pl',
                color_continuous_scale=['red', 'yellow', 'green'],
                text='pl'
            )
            fig_bar.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
            fig_bar.update_layout(template="plotly_dark", height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Holdings table
    st.markdown("## 📊 Current Holdings")
    
    if not portfolio_df.empty:
        display_df = portfolio_df.copy()
        
        # Add current prices and P&L
        display_df['current_price'] = display_df['ticker'].map(current_prices)
        display_df['current_value'] = display_df['quantity'] * display_df['current_price']
        display_df['pl'] = display_df['current_value'] - display_df['total_invested']
        display_df['pl_pct'] = (display_df['pl'] / display_df['total_invested']) * 100
        
        # Format for display
        display_table = pd.DataFrame({
            'Ticker': display_df['ticker'],
            'Quantity': display_df['quantity'],
            'Avg Price': display_df['avg_price'].apply(lambda x: f"₹{x:.2f}"),
            'Current Price': display_df['current_price'].apply(lambda x: f"₹{x:.2f}"),
            'Invested': display_df['total_invested'].apply(lambda x: f"₹{x:,.2f}"),
            'Current Value': display_df['current_value'].apply(lambda x: f"₹{x:,.2f}"),
            'P&L': display_df['pl'].apply(lambda x: f"₹{x:+,.2f}"),
            'P&L %': display_df['pl_pct'].apply(lambda x: f"{x:+.2f}%")
        })
        
        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
            height=min(400, (len(display_table) + 1) * 35 + 3)
        )
    else:
        st.info("📭 Your portfolio is empty. Start analyzing stocks and use AI agents to make your first trade!")
        
        # Show getting started guide
        with st.expander("🚀 How to get started"):
            st.markdown("""
            ### Steps to start trading:
            
            1. **Go to Home** - Click the "← Home" button
            2. **Enter a ticker** - Type in a stock ticker (e.g., RELIANCE.NS, TCS.NS)
            3. **Analyze the stock** - View technical analysis, fundamentals, and news
            4. **Run AI Agents** - Click the "🤖 AI Agents" button for recommendations
            5. **Execute trade** - Follow the AI agent's recommendation
            6. **View portfolio** - Come back here to see your positions!
            """)
    
    st.markdown("---")
    
    # Transaction history
    st.markdown("## 📜 Transaction History")
    
    history_df = db.get_transaction_history(user_id, limit=50)
    
    if not history_df.empty:
        display_history = pd.DataFrame({
            'Date/Time': pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M'),
            'Ticker': history_df['ticker'],
            'Action': history_df['action'],
            'Quantity': history_df['quantity'],
            'Price': history_df['price'].apply(lambda x: f"₹{x:.2f}"),
            'Total Amount': history_df['total_amount'].apply(lambda x: f"₹{x:,.2f}"),
            'Agent Rec': history_df['agent_recommendation'].apply(lambda x: str(x)[:50] + '...' if x and len(str(x)) > 50 else str(x))
        })
        
        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True,
            height=min(400, (len(display_history) + 1) * 35 + 3)
        )
    else:
        st.info("No transactions yet. Make your first trade to see history here!")
    
    # Quick actions
    st.markdown("---")
    st.markdown("## ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Analyze New Stock", use_container_width=True, type="primary"):
            st.session_state.page = 'home'
            st.rerun()
    
    with col2:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
    
    with col3:
        if st.button("🔐 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.page = 'login'
            st.rerun()
