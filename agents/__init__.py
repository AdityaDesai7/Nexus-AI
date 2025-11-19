# agents/__init__.py
"""
Agents Package - Autonomous Trading System
All agents are completely autonomous with ZERO LLM dependency
"""

# Import autonomous agents
from .technical_agent import TechnicalAnalysisAgent
from .sentiment_agent import SentimentAnalysisAgent
from .risk_agent import RiskManagementAgent
from .portfolio_agent import PortfolioManagerAgent
from .master_agent import MasterAgent
from .debate_agent import DebateAgent
from .data_collection_agent import DataCollectionAgent

# Only import institutional agents if they exist
try:
    from .institutional_agents import (
        FFIMomentumAgent,
        ExecutionBreakdownDetector,
        VolumePatternRecognition,
        IFICalculator,
        AccumulationDetector,
        AgentSignal,
        AggregatedSignal
    )
except ImportError as e:
    print(f"⚠️ Warning: Could not import institutional agents: {e}")
    print("Some advanced institutional analysis features may not be available")

__all__ = [
    # Core Autonomous Agents
    'TechnicalAnalysisAgent',
    'SentimentAnalysisAgent',
    'RiskManagementAgent',
    'PortfolioManagerAgent',
    'MasterAgent',
    'DebateAgent',
    'DataCollectionAgent',
    
    # Institutional Agents (if available)
    'FFIMomentumAgent',
    'ExecutionBreakdownDetector',
    'VolumePatternRecognition',
    'IFICalculator',
    'AccumulationDetector',
    'AgentSignal',
    'AggregatedSignal',
]

print("✅ All autonomous agents loaded successfully")
