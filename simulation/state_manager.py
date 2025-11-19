import streamlit as st
from models.simulation import PortfolioState

def get_portfolio_state(user_id: str) -> PortfolioState:
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = PortfolioState()
    return st.session_state.portfolio