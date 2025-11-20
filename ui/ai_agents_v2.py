# trading_bot/ui/ai_agents_v2.py
import logging
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

from agent_runner import AgentRunner
from agents.inst_wrappers import create_inst_wrappers     # ← USE ONLY INSTITUTIONAL WRAPPERS
from tools.toolbox import TOOLS
from llm.llm_wrapper import LLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def show_ai_agents_page_v2():
    """
    Institutional Agents UI (Only 9 institutional AI agents)
    Clean and separated from old ai_agents.py
    """
    st.set_page_config(layout="wide", page_title="🏦 Institutional AI Agents")

    st.title("🏦 Institutional AI Agents (V2)")
    st.caption("This page runs ONLY the 9 institutional agents. Classical agents are NOT used here.")

    # ---------------------------------------------------------
    # INITIALIZE RUNNER (ONLY INSTITUTIONAL AGENTS)
    # ---------------------------------------------------------
    if "inst_runner" not in st.session_state:
        runner = AgentRunner()

        # Load ONLY institutional wrappers
        wrapped_agents = create_inst_wrappers(tools=TOOLS, llm=LLM())

        # Register only these agents
        for name, ag in wrapped_agents.items():
            runner.register(name, ag)

        st.session_state.inst_runner = runner
        st.session_state.inst_agent_names = list(wrapped_agents.keys())

        logger.info("Institutional Agents v2 loaded: %s", st.session_state.inst_agent_names)

    runner = st.session_state.inst_runner
    inst_agents = st.session_state.inst_agent_names

    # ---------------------------------------------------------
    # SIDEBAR INPUTS
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("Run Institutional Agents")

        ticker = st.text_input("Ticker (e.g. RELIANCE.NS)", value="RELIANCE.NS")

        start_date_input = st.date_input(
            "Start date",
            value=datetime.now() - timedelta(days=200)
        )
        end_date_input = st.date_input("End date", value=datetime.now())

        run_btn = st.button("🚀 Run Institutional Agents")

    if not run_btn:
        st.info("Configure inputs and click Run.")
        return

    start = datetime.combine(start_date_input, datetime.min.time())
    end = datetime.combine(end_date_input, datetime.max.time())

    st.markdown("---")
    st.subheader("📊 Agent Outputs")

    # ---------------------------------------------------------
    # RUN ALL 9 INSTITUTIONAL AGENTS
    # ---------------------------------------------------------
    results = {}

    for agent_name in inst_agents:
        st.write(f"### 🔹 Running `{agent_name}` agent…")

        try:
            output = runner.run(agent_name, {
                "ticker": ticker,
                "start": start,
                "end": end
            })

            results[agent_name] = output
            st.json(output)

        except Exception as e:
            st.error(f"{agent_name} failed: {e}")

    st.success("✅ All Institutional Agents Completed")

    # ---------------------------------------------------------
    # MASTER AGGREGATOR (optional)
    # ---------------------------------------------------------
    if "master_institutional" in inst_agents:
        st.markdown("---")
        st.subheader("🏁 Institutional Master Result")

        master_out = results.get("master_institutional")
        if master_out:
            st.json(master_out)
        else:
            st.info("Master agent did not run.")

