from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Dict
from agents.data_collection_agent import DataCollectionAgent
from agents.technical_agent import TechnicalAnalysisAgent
from agents.sentiment_agent import SentimentAnalysisAgent
from agents.debate_agent import DebateAgent
from agents.risk_agent import RiskManagementAgent
from agents.portfolio_agent import PortfolioManagerAgent
from agents.master_agent import MasterAgent
from agents.backtesting_agent import BacktestingAgent

from pydantic import BaseModel

class Trade(BaseModel):
    ticker: str
    action: str
    quantity: int
    price: float

class TradingState(TypedDict):
    ticker: str
    data: Dict
    technical: Dict
    sentiment: float
    debate: str
    risk: Dict
    trade: Trade  # Updated to expect Trade object
    portfolio: Dict
    master: Dict
    backtest: Dict

def create_graph():
    workflow = StateGraph(TradingState)
    
    # Initialize agents
    data_agent = DataCollectionAgent()
    technical_agent = TechnicalAnalysisAgent()
    sentiment_agent = SentimentAnalysisAgent()
    debate_agent = DebateAgent()
    risk_agent = RiskManagementAgent()
    portfolio_agent = PortfolioManagerAgent()
    master_agent = MasterAgent()
    backtest_agent = BacktestingAgent()

    # Add nodes to the graph
    workflow.add_node("data", lambda state: {"data": data_agent.collect(state["ticker"])})
    workflow.add_node("technical", lambda state: {"technical": technical_agent.analyze(state["ticker"]).dict()})
    workflow.add_node("sentiment", lambda state: {"sentiment": sentiment_agent.analyze(state["ticker"], state["data"])})
    workflow.add_node("debate", lambda state: {"debate": debate_agent.debate(state)})
    workflow.add_node("risk", lambda state: {"risk": risk_agent.evaluate(state, state.get("portfolio", {}))})
    workflow.add_node("portfolio", lambda state: {"trade": portfolio_agent.decide(state, state["risk"])})  # Removed .dict()
    workflow.add_node("master", lambda state: {"master": master_agent.oversee(state)})
    workflow.add_node("backtest", lambda state: {"backtest": backtest_agent.backtest(state["ticker"])})

    # Define the edges (workflow sequence)
    workflow.add_edge(START, "data")
    workflow.add_edge("data", "technical")
    workflow.add_edge("technical", "sentiment")
    workflow.add_edge("sentiment", "debate")
    workflow.add_edge("debate", "risk")
    workflow.add_edge("risk", "portfolio")
    workflow.add_edge("portfolio", "master")
    workflow.add_edge("master", "backtest")
    workflow.add_edge("backtest", END)

    return workflow.compile()

trading_graph = create_graph()