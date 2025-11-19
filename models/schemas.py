# from pydantic import BaseModel

# class TechnicalAnalysisOutput(BaseModel):
#     ticker: str
#     rsi: float
#     macd: float
#     macd_signal: float
#     bollinger_upper: float
#     bollinger_lower: float
#     support: float
#     resistance: float
#     recommendation: str


# models/schemas.py - COMPLETE FILE

from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime


class AgentMessage(BaseModel):
    agent: str
    timestamp: datetime
    message: str
    data: Optional[Dict] = None


class TradingState(BaseModel):
    ticker: str
    start_date: datetime
    end_date: datetime
    portfolio: Dict
    agent_messages: List[AgentMessage]
    recommendations: Dict[str, Any]
    final_decision: Optional[Dict] = None
    
    class Config:
        arbitrary_types_allowed = True


# ADD THIS:
class TechnicalAnalysisOutput(BaseModel):
    ticker: str
    rsi: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    support: float
    resistance: float
    recommendation: str
