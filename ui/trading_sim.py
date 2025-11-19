import streamlit as st
from graphs.trading_graph import trading_graph
from simulation.portfolio_manager import PortfolioManager
from models.simulation import Trade

def show_trading_sim():
    st.title("AI Trading Simulation ")
    ticker = st.text_input("Enter Stock for Analysis (e.g., NTPC.NS)", "NTPC.NS")  # Default to NTPC.NS

    # Initialize session state for result if not present
    if "result" not in st.session_state:
        st.session_state.result = None

    # Run agents and store result
    if st.button("Run Agents") and ticker:
        state = {"ticker": ticker}
        st.session_state.result = trading_graph.invoke(state)
        trade = st.session_state.result.get("trade")  # Handle None case
        if trade:
            portfolio = PortfolioManager()
            updated = portfolio.execute_trade(trade)
            st.write(f"Agent Decision: {trade.action} {trade.quantity} shares of {ticker} at ₹{trade.price * 83:.2f}")
            st.write(f"Portfolio Balance: ₹{updated.balance:.2f}")
            st.write(f"Total Value: ₹{updated.total_value:.2f}")
        else:
            st.write("No trade decision made (e.g., Hold recommendation).")

    # Backtest strategy using stored result
    if st.button("Backtest Strategy") and ticker and st.session_state.result:
        backtest_result = st.session_state.result.get("backtest", {})
        if backtest_result and "error" not in backtest_result:
            st.write("Backtest Results:")
            st.write(f"Initial Capital: ₹{backtest_result['initial_capital']:.2f}")
            st.write(f"Ending Capital: ₹{backtest_result['ending_capital']:.2f}")
            st.write(f"Total Return: {backtest_result['total_return']:.2f}%")
            st.write(f"Sharpe Ratio: {backtest_result['sharpe_ratio']:.2f}")
            st.write(f"Max Drawdown: {backtest_result['max_drawdown']:.2f}%")
            st.write(f"Analysis: {backtest_result['analysis']}")
        else:
            st.error("Backtest failed or not available: " + str(backtest_result.get("error", "Run Agents first to generate data")))
