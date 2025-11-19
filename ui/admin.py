# ui/admin.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from models.database import get_database
import pandas as pd
import yfinance as yf

def show_admin_dashboard():
    """Display admin dashboard"""
    
    # Check if user is admin
    if not st.session_state.get('logged_in') or not st.session_state.get('is_admin'):
        st.error("⛔ Access Denied! Admin privileges required.")
        return
    
    username = st.session_state.username
    db = get_database()
    
    # Header
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("← Home"):
            st.session_state.page = 'home'
            st.rerun()
    with col2:
        st.markdown(f"<h1 style='text-align:center; color:#00b894;'>👨‍💼 Admin Dashboard</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:#888;'>Logged in as: {username} (Admin)</p>", unsafe_allow_html=True)
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "👥 All Users",
        "💼 User Portfolios",
        "📜 All Transactions",
        "🏆 Leaderboard"
    ])
    
    # ========================================
    # TAB 1: SYSTEM OVERVIEW
    # ========================================
    with tab1:
        st.markdown("## 📊 System Statistics")
        
        with st.spinner("Loading system stats..."):
            stats = db.get_system_stats()
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Users", stats['total_users'])
        with col2:
            st.metric("Active Users", stats['active_users'])
        with col3:
            st.metric("Total Trades", stats['total_transactions'])
        with col4:
            st.metric("Total Capital", f"₹{stats['total_capital']:,.0f}")
        with col5:
            st.metric("Total Invested", f"₹{stats['total_invested']:,.0f}")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Top traded stocks
            if stats['top_stocks']:
                st.markdown("### 📈 Most Traded Stocks")
                stock_data = pd.DataFrame(stats['top_stocks'], columns=['Ticker', 'Trade Count'])
                
                fig = px.bar(
                    stock_data,
                    x='Ticker',
                    y='Trade Count',
                    title='Top 5 Most Traded Stocks',
                    color='Trade Count',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Buy vs Sell
            if stats['buy_sell_stats']:
                st.markdown("### 💹 Buy vs Sell Activity")
                bs_data = pd.DataFrame(stats['buy_sell_stats'], columns=['Action', 'Count'])
                
                fig = px.pie(
                    bs_data,
                    values='Count',
                    names='Action',
                    title='Buy vs Sell Distribution',
                    color_discrete_map={'BUY': '#00b894', 'SELL': '#ff7675'}
                )
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity
        st.markdown("---")
        st.markdown("### 📜 Recent System Activity")
        
        recent_transactions = db.get_all_transactions_admin(limit=10)
        
        if not recent_transactions.empty:
            display_recent = pd.DataFrame({
                'Time': pd.to_datetime(recent_transactions['timestamp']).dt.strftime('%Y-%m-%d %H:%M'),
                'User': recent_transactions['username'],
                'Action': recent_transactions['action'],
                'Ticker': recent_transactions['ticker'],
                'Quantity': recent_transactions['quantity'],
                'Price': recent_transactions['price'].apply(lambda x: f"₹{x:.2f}"),
                'Amount': recent_transactions['total_amount'].apply(lambda x: f"₹{x:,.2f}")
            })
            
            st.dataframe(display_recent, use_container_width=True, hide_index=True)
        else:
            st.info("No transactions yet")
    
    # ========================================
    # TAB 2: ALL USERS
    # ========================================
    with tab2:
        st.markdown("## 👥 All Registered Users")
        
        with st.spinner("Loading user data..."):
            users_df = db.get_all_users()
        
        if not users_df.empty:
            st.markdown(f"**Total Users:** {len(users_df)}")
            
            # Format data for display
            display_users = pd.DataFrame({
                'User ID': users_df['user_id'],
                'Username': users_df['username'],
                'Email': users_df['email'],
                'Registered': pd.to_datetime(users_df['created_at']).dt.strftime('%Y-%m-%d'),
                'Initial Capital': users_df['initial_capital'].apply(lambda x: f"₹{x:,.0f}"),
                'Available Cash': users_df['available_cash'].apply(lambda x: f"₹{x:,.0f}"),
                'Invested': users_df['invested_amount'].apply(lambda x: f"₹{x:,.0f}"),
                'Portfolio Value': users_df['total_portfolio_value'].apply(lambda x: f"₹{x:,.0f}"),
                'Total Value': users_df['total_value'].apply(lambda x: f"₹{x:,.0f}"),
                'P&L': users_df.apply(lambda row: f"₹{(row['total_value'] - row['initial_capital']):+,.0f}", axis=1)
            })
            
            st.dataframe(display_users, use_container_width=True, hide_index=True, height=400)
            
            # User distribution chart
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                # Users by registration date
                users_df['reg_date'] = pd.to_datetime(users_df['created_at']).dt.date
                user_growth = users_df.groupby('reg_date').size().cumsum().reset_index()
                user_growth.columns = ['Date', 'Total Users']
                
                fig = px.line(
                    user_growth,
                    x='Date',
                    y='Total Users',
                    title='User Growth Over Time',
                    markers=True
                )
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Portfolio value distribution
                fig = px.histogram(
                    users_df,
                    x='total_value',
                    title='Portfolio Value Distribution',
                    nbins=20,
                    labels={'total_value': 'Portfolio Value'}
                )
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No users registered yet")
    
    # ========================================
    # TAB 3: USER PORTFOLIOS
    # ========================================
    with tab3:
        st.markdown("## 💼 View User Portfolio")
        
        # User selector
        users_df = db.get_all_users()
        
        if not users_df.empty:
            user_options = {f"{row['username']} (ID: {row['user_id']})": row['user_id'] 
                          for _, row in users_df.iterrows()}
            
            selected_user = st.selectbox(
                "Select User",
                options=list(user_options.keys())
            )
            
            if selected_user:
                user_id = user_options[selected_user]
                
                with st.spinner("Loading user portfolio..."):
                    user_data = db.get_user_portfolio_admin(user_id)
                
                if user_data:
                    # User info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Username", user_data['username'])
                    with col2:
                        st.metric("Email", user_data['email'])
                    with col3:
                        st.metric("Member Since", pd.to_datetime(user_data['created_at']).strftime('%Y-%m-%d'))
                    with col4:
                        st.metric("Total Trades", user_data['transaction_count'])
                    
                    st.markdown("---")
                    
                    # Portfolio summary
                    balance = user_data['balance']
                    total_value = balance['available_cash'] + balance['total_portfolio_value']
                    profit_loss = total_value - user_data['initial_capital']
                    pl_pct = (profit_loss / user_data['initial_capital']) * 100
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Value", f"₹{total_value:,.2f}")
                    with col2:
                        st.metric("Available Cash", f"₹{balance['available_cash']:,.2f}")
                    with col3:
                        st.metric("Invested", f"₹{balance['invested_amount']:,.2f}")
                    with col4:
                        st.metric("Portfolio Value", f"₹{balance['total_portfolio_value']:,.2f}")
                    with col5:
                        st.metric("P&L", f"₹{profit_loss:+,.2f}", delta=f"{pl_pct:+.2f}%")
                    
                    st.markdown("---")
                    
                    # Holdings
                    portfolio_df = user_data['portfolio']
                    
                    if not portfolio_df.empty:
                        st.markdown("### 📊 Holdings")
                        
                        # Fetch current prices
                        current_prices = {}
                        for ticker in portfolio_df['ticker']:
                            try:
                                stock = yf.Ticker(ticker)
                                info = stock.info
                                price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
                                current_prices[ticker] = price
                            except:
                                current_prices[ticker] = 0
                        
                        # Calculate current values
                        portfolio_df['current_price'] = portfolio_df['ticker'].map(current_prices)
                        portfolio_df['current_value'] = portfolio_df['quantity'] * portfolio_df['current_price']
                        portfolio_df['pl'] = portfolio_df['current_value'] - portfolio_df['total_invested']
                        portfolio_df['pl_pct'] = (portfolio_df['pl'] / portfolio_df['total_invested']) * 100
                        
                        display_portfolio = pd.DataFrame({
                            'Ticker': portfolio_df['ticker'],
                            'Quantity': portfolio_df['quantity'],
                            'Avg Price': portfolio_df['avg_price'].apply(lambda x: f"₹{x:.2f}"),
                            'Current Price': portfolio_df['current_price'].apply(lambda x: f"₹{x:.2f}"),
                            'Invested': portfolio_df['total_invested'].apply(lambda x: f"₹{x:,.2f}"),
                            'Current Value': portfolio_df['current_value'].apply(lambda x: f"₹{x:,.2f}"),
                            'P&L': portfolio_df['pl'].apply(lambda x: f"₹{x:+,.2f}"),
                            'P&L %': portfolio_df['pl_pct'].apply(lambda x: f"{x:+.2f}%")
                        })
                        
                        st.dataframe(display_portfolio, use_container_width=True, hide_index=True)
                        
                        # Portfolio chart
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.pie(
                                portfolio_df,
                                values='current_value',
                                names='ticker',
                                title='Portfolio Allocation'
                            )
                            fig.update_layout(template="plotly_dark", height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.bar(
                                portfolio_df,
                                x='ticker',
                                y='pl',
                                title='P&L by Stock',
                                color='pl',
                                color_continuous_scale=['red', 'yellow', 'green']
                            )
                            fig.update_layout(template="plotly_dark", height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("This user has no holdings")
        else:
            st.info("No users registered yet")
    
    # ========================================
    # TAB 4: ALL TRANSACTIONS
    # ========================================
    with tab4:
        st.markdown("## 📜 All System Transactions")
        
        # Transaction limit selector
        limit = st.selectbox("Show transactions", [50, 100, 200, 500, 1000], index=1)
        
        with st.spinner("Loading transactions..."):
            transactions_df = db.get_all_transactions_admin(limit=limit)
        
        if not transactions_df.empty:
            st.markdown(f"**Showing {len(transactions_df)} most recent transactions**")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_user = st.multiselect(
                    "Filter by User",
                    options=transactions_df['username'].unique().tolist()
                )
            
            with col2:
                filter_action = st.multiselect(
                    "Filter by Action",
                    options=['BUY', 'SELL']
                )
            
            with col3:
                filter_ticker = st.multiselect(
                    "Filter by Ticker",
                    options=transactions_df['ticker'].unique().tolist()
                )
            
            # Apply filters
            filtered_df = transactions_df.copy()
            if filter_user:
                filtered_df = filtered_df[filtered_df['username'].isin(filter_user)]
            if filter_action:
                filtered_df = filtered_df[filtered_df['action'].isin(filter_action)]
            if filter_ticker:
                filtered_df = filtered_df[filtered_df['ticker'].isin(filter_ticker)]
            
            # Display filtered data
            display_transactions = pd.DataFrame({
                'ID': filtered_df['transaction_id'],
                'Time': pd.to_datetime(filtered_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M'),
                'User': filtered_df['username'],
                'Action': filtered_df['action'],
                'Ticker': filtered_df['ticker'],
                'Quantity': filtered_df['quantity'],
                'Price': filtered_df['price'].apply(lambda x: f"₹{x:.2f}"),
                'Amount': filtered_df['total_amount'].apply(lambda x: f"₹{x:,.2f}"),
                'Agent Rec': filtered_df['agent_recommendation'].apply(
                    lambda x: str(x)[:30] + '...' if x and len(str(x)) > 30 else str(x)
                )
            })
            
            st.dataframe(display_transactions, use_container_width=True, hide_index=True, height=400)
            
            # Transaction analytics
            st.markdown("---")
            st.markdown("### 📊 Transaction Analytics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_buy = filtered_df[filtered_df['action'] == 'BUY']['total_amount'].sum()
                st.metric("Total Buy Volume", f"₹{total_buy:,.0f}")
            
            with col2:
                total_sell = filtered_df[filtered_df['action'] == 'SELL']['total_amount'].sum()
                st.metric("Total Sell Volume", f"₹{total_sell:,.0f}")
            
            with col3:
                avg_transaction = filtered_df['total_amount'].mean()
                st.metric("Avg Transaction", f"₹{avg_transaction:,.0f}")
        else:
            st.info("No transactions yet")
    
    # ========================================
    # TAB 5: LEADERBOARD
    # ========================================
    with tab5:
        st.markdown("## 🏆 User Leaderboard")
        st.markdown("*Ranked by Profit & Loss*")
        
        with st.spinner("Loading leaderboard..."):
            leaderboard_df = db.get_user_leaderboard()
        
        if not leaderboard_df.empty:
            # Add rank
            leaderboard_df['rank'] = range(1, len(leaderboard_df) + 1)
            
            # Format for display
            display_leaderboard = pd.DataFrame({
                'Rank': leaderboard_df['rank'],
                'Username': leaderboard_df['username'],
                'Total Value': leaderboard_df['total_value'].apply(lambda x: f"₹{x:,.0f}"),
                'P&L': leaderboard_df['profit_loss'].apply(lambda x: f"₹{x:+,.0f}"),
                'P&L %': leaderboard_df['profit_loss_pct'].apply(lambda x: f"{x:+.2f}%"),
                'Total Trades': leaderboard_df['total_trades']
            })
            
            # Color-code by rank
            def highlight_rank(row):
                if row['Rank'] == 1:
                    return ['background-color: #FFD700'] * len(row)  # Gold
                elif row['Rank'] == 2:
                    return ['background-color: #C0C0C0'] * len(row)  # Silver
                elif row['Rank'] == 3:
                    return ['background-color: #CD7F32'] * len(row)  # Bronze
                return [''] * len(row)
            
            st.dataframe(
                display_leaderboard,
                use_container_width=True,
                hide_index=True,
                height=500
            )
            
            # Leaderboard chart
            st.markdown("---")
            
            fig = px.bar(
                leaderboard_df.head(10),
                x='username',
                y='profit_loss',
                title='Top 10 Users by P&L',
                color='profit_loss',
                color_continuous_scale=['red', 'yellow', 'green'],
                labels={'profit_loss': 'Profit/Loss (₹)', 'username': 'User'}
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No users to rank yet")
