# trading_bot/agents/inst_wrappers.py
"""
Wrappers for the 9 institutional agents so they behave like BaseAgent-driven agents.
They call tools from TOOLS and then call the institutional agent logic.
"""

import logging
from typing import Dict, Any
from agents.base_agent import BaseAgent
from agents.institutional_agents import MasterInstitutionalAggregator, AgentSignal, AggregatedSignal
from tools.toolbox import TOOLS
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class InstWrapper(BaseAgent):
    def __init__(self, name="institutional_agent", tools: Dict[str, Any] = None, llm=None, system_prompt: str = ""):
        super().__init__(name=name, tools=tools or TOOLS, llm=llm, system_prompt=system_prompt)
        self.logic = MasterInstitutionalAggregator()

    def plan(self, user_input: Dict) -> Dict:
        # Expect ticker and optional start/end
        return {"action": "aggregate", **user_input}

    def act(self, plan: Dict) -> Dict:
        ticker = plan.get("ticker")
        start = plan.get("start")
        end = plan.get("end")
        # Defensive defaults: last 90 days if not provided
        if start is None or end is None:
            end = end or datetime.now()
            start = start or (end - pd.Timedelta(days=180))

        # Call tools (each tool is defensive)
        try:
            price_df = self.call_tool("fetch_price", ticker, start, end)
            # tool_fetch_price returns DataFrame
            if isinstance(price_df, dict) and price_df.get("status") == "ERROR":
                raise RuntimeError(price_df.get("error"))
            if price_df is None:
                raise RuntimeError("Price fetch returned None")

            # volume series
            volume_series = self.call_tool("fetch_volume", ticker, start, end)
        except Exception as e:
            logger.warning("Institutional wrapper: price/volume fetch failed: %s", e)
            # Return "no data available" gracefully
            return {"ticker": ticker, "institutional": {"status": "NO_DATA", "reason": str(e)}}

        # FII and orders (best-effort)
        try:
            fii = self.call_tool("fetch_fii", ticker)
        except Exception as e:
            logger.warning("fetch_fii failed: %s", e)
            fii = {"available": False, "reason": str(e)}

        try:
            orders = self.call_tool("fetch_orders", ticker)
        except Exception as e:
            logger.warning("fetch_orders failed: %s", e)
            orders = []

        # convert structures if necessary
        price_series = price_df["Close"] if hasattr(price_df, "__getitem__") and "Close" in price_df.columns else None

        # call aggregator
        agg = self.logic.aggregate(fii_data=fii, order_data=orders, price_data=price_series, volume_data=volume_series)

        # convert dataclasses to dicts for transport
        try:
            breakdown = {}
            for k, v in agg.agent_breakdown.items():
                # v is AgentSignal dataclass; convert to dict
                breakdown[k] = {
                    "agent_name": v.agent_name,
                    "score": v.score,
                    "confidence": v.confidence,
                    "action": v.action,
                    "reason": v.reason,
                    "details": v.details
                }
        except Exception:
            breakdown = str(agg.agent_breakdown)

        result = {
            "ticker": ticker,
            "institutional": {
                "final_score": agg.final_score,
                "recommendation": agg.recommendation,
                "confidence": agg.confidence,
                "reasoning": agg.reasoning,
                "breakdown": breakdown
            }
        }
        return result

# Convenience factory
def create_inst_wrappers(tools: Dict[str, Any] = None, llm = None):
    tools = tools or TOOLS
    return {"institutional": InstWrapper(tools=tools, llm=llm)}
