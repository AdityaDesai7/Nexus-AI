# # agent_runner.py
# # Central registry + unified execution layer for all agents

# import logging
# from typing import Dict, Any

# # FIXED IMPORTS (correct for your folder structure)
# from agents.base_agent import BaseAgent
# from tools.toolbox import TOOLS
# from llm.llm_wrapper import LLM   # your Groq/Dummy unified wrapper
# from agents.inst_wrappers import create_inst_wrappers


# logger = logging.getLogger(__name__)


# class AgentRunner:
#     """
#     Central manager for all agents.
#     - Registers agents by name
#     - Injects tools + LLM into every agent
#     - Runs agents safely
#     """

#     def __init__(self):
#         self.agents: Dict[str, BaseAgent] = {}

#         # shared tools + shared LLM wrapper (Groq or Dummy)
#         self.tools = TOOLS
#         self.llm = LLM()

#         logger.info("AgentRunner initialized with unified tools + LLM.")

#     # --------------------------------------------------------------
#     # REGISTER AGENT
#     # --------------------------------------------------------------
#     def register(self, name: str, agent: BaseAgent):
#         if not isinstance(name, str):
#             raise ValueError("Agent name must be a string")

#         if not isinstance(agent, BaseAgent):
#             raise TypeError(f"Agent '{name}' must inherit BaseAgent")

#         self.agents[name] = agent
#         logger.info(f"Registered agent: {name}")

#     # --------------------------------------------------------------
#     # RUN AGENT
#     # --------------------------------------------------------------
#     def run(self, name: str, user_input: Dict[str, Any]):
#         """
#         Run one agent and return its output.
#         Inject tools + LLM automatically.
#         """
#         agent = self.agents.get(name)
#         if not agent:
#             raise RuntimeError(f"Unknown agent: {name}")

#         logger.info(f"Running agent: {name}")

#         # Inject dependencies if missing
#         agent.tools = getattr(agent, "tools", None) or self.tools
#         agent.llm = getattr(agent, "llm", None) or self.llm

#         # Execute safely
#         try:
#             result = agent.run(user_input)
#             logger.info(f"Agent '{name}' completed successfully.")
#             return result

#         except Exception as e:
#             logger.error(f"Agent '{name}' failed: {e}", exc_info=True)
#             return {
#                 "status": "ERROR",
#                 "agent": name,
#                 "error": str(e)
#             }


# agent_runner.py
# Central registry + unified execution layer for all agents

import logging
from typing import Dict, Any

# FIXED IMPORTS (correct for your folder structure)
from agents.base_agent import BaseAgent
from tools.toolbox import TOOLS
from llm.llm_wrapper import LLM   # your Groq/Dummy unified wrapper

# IMPORTANT: institutional wrappers (9 agents)
from agents.inst_wrappers import create_inst_wrappers


logger = logging.getLogger(__name__)


class AgentRunner:
    """
    Central manager for all agents.
    - Registers agents by name
    - Injects tools + LLM into every agent
    - Runs agents safely
    - Supports both: regular 5 agents + 9 institutional agents
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}

        # shared tools + shared LLM wrapper (Groq or Dummy)
        self.tools = TOOLS
        self.llm = LLM()

        logger.info("AgentRunner initialized with unified tools + LLM.")

        # --------------------------------------------------------------
        # AUTO-REGISTER 9 INSTITUTIONAL AGENTS
        # --------------------------------------------------------------
        try:
            inst_agents = create_inst_wrappers(tools=self.tools, llm=self.llm)

            for name, agent in inst_agents.items():
                self.register(name, agent)

            logger.info(f"Institutional agents registered: {list(inst_agents.keys())}")

        except Exception as e:
            logger.error(f"Failed to register institutional agents: {e}", exc_info=True)

    # --------------------------------------------------------------
    # REGISTER ANY AGENT (5 normal agents use this)
    # --------------------------------------------------------------
    def register(self, name: str, agent: BaseAgent):
        if not isinstance(name, str):
            raise ValueError("Agent name must be a string")

        if not isinstance(agent, BaseAgent):
            raise TypeError(f"Agent '{name}' must inherit BaseAgent")

        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")

    # --------------------------------------------------------------
    # RUN ONE AGENT
    # --------------------------------------------------------------
    def run(self, name: str, user_input: Dict[str, Any]):
        """
        Run one agent and return its output.
        Inject tools + LLM automatically.
        """
        agent = self.agents.get(name)
        if not agent:
            raise RuntimeError(f"Unknown agent: {name}")

        logger.info(f"Running agent: {name}")

        # Inject dependencies if missing
        agent.tools = getattr(agent, "tools", None) or self.tools
        agent.llm = getattr(agent, "llm", None) or self.llm

        # Execute safely
        try:
            result = agent.run(user_input)
            logger.info(f"Agent '{name}' completed successfully.")
            return result

        except Exception as e:
            logger.error(f"Agent '{name}' failed: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "agent": name,
                "error": str(e)
            }

    # --------------------------------------------------------------
    # RUN ALL AGENTS (optional but recommended)
    # --------------------------------------------------------------
    def run_all(self, user_input: Dict[str, Any]):
        """
        Runs EVERY registered agent.
        Returns a dict: { agent_name: result }
        """
        outputs = {}
        logger.info(f"Running all agents: {list(self.agents.keys())}")

        for name, agent in self.agents.items():
            try:
                outputs[name] = self.run(name, user_input)
            except Exception as e:
                logger.error(f"Agent '{name}' crashed inside run_all: {e}")
                outputs[name] = {
                    "status": "ERROR",
                    "agent": name,
                    "error": str(e)
                }

        return outputs

