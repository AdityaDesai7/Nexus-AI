# Trading Bot
A multi-agent trading system for Indian stocks built with Python, yfinance, and Streamlit.

## Setup
1. Install Python 3.10+.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `.\venv\Scripts\Activate.ps1` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Add API keys to `.env` (GROQ_API_KEY, TAVILY_API_KEY).
6. Run the app: `streamlit run app.py`

## Usage
- **Public Access**: Enter an Indian stock ticker (e.g., NTPC.NS) on the Home tab to see a candlestick chart and news.
- **Login**: Use "user"/"pass" to access the Trading Sim tab with ₹10,00,000 virtual capital.
- **Trading Simulation**: Enter a ticker (e.g., NTPC.NS), click "Run Agents" to see agent-driven trade decisions, and check portfolio updates.
- **Backtesting**: Click "Backtest Strategy" to evaluate the strategy's historical performance.
- **Test Backend**: Run `python main.py` to debug the agent workflow.

## Notes
- For NSE stocks (e.g., NTPC.NS, RELIANCE.NS), yfinance support may vary—use BSE codes (e.g., NTPC.BO) if needed.
- This is for educational purposes only, not financial advice.