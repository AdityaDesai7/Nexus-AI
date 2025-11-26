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


# # agent_runner.py
# # Central registry + unified execution layer for all agents

# import logging
# from typing import Dict, Any

# # FIXED IMPORTS (correct for your folder structure)
# from agents.base_agent import BaseAgent
# from tools.toolbox import TOOLS
# from llm.llm_wrapper import LLM   # your Groq/Dummy unified wrapper

# # IMPORTANT: institutional wrappers (9 agents)
# from agents.inst_wrappers import create_inst_wrappers


# logger = logging.getLogger(__name__)


# class AgentRunner:
#     """
#     Central manager for all agents.
#     - Registers agents by name
#     - Injects tools + LLM into every agent
#     - Runs agents safely
#     - Supports both: regular 5 agents + 9 institutional agents
#     """

#     def __init__(self):
#         self.agents: Dict[str, BaseAgent] = {}

#         # shared tools + shared LLM wrapper (Groq or Dummy)
#         self.tools = TOOLS
#         self.llm = LLM()

#         logger.info("AgentRunner initialized with unified tools + LLM.")

#         # --------------------------------------------------------------
#         # AUTO-REGISTER 9 INSTITUTIONAL AGENTS
#         # --------------------------------------------------------------
#         try:
#             inst_agents = create_inst_wrappers(tools=self.tools, llm=self.llm)

#             for name, agent in inst_agents.items():
#                 self.register(name, agent)

#             logger.info(f"Institutional agents registered: {list(inst_agents.keys())}")

#         except Exception as e:
#             logger.error(f"Failed to register institutional agents: {e}", exc_info=True)

#     # --------------------------------------------------------------
#     # REGISTER ANY AGENT (5 normal agents use this)
#     # --------------------------------------------------------------
#     def register(self, name: str, agent: BaseAgent):
#         if not isinstance(name, str):
#             raise ValueError("Agent name must be a string")

#         if not isinstance(agent, BaseAgent):
#             raise TypeError(f"Agent '{name}' must inherit BaseAgent")

#         self.agents[name] = agent
#         logger.info(f"Registered agent: {name}")

#     # --------------------------------------------------------------
#     # RUN ONE AGENT
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

#     # --------------------------------------------------------------
#     # RUN ALL AGENTS (optional but recommended)
#     # --------------------------------------------------------------
#     def run_all(self, user_input: Dict[str, Any]):
#         """
#         Runs EVERY registered agent.
#         Returns a dict: { agent_name: result }
#         """
#         outputs = {}
#         logger.info(f"Running all agents: {list(self.agents.keys())}")

#         for name, agent in self.agents.items():
#             try:
#                 outputs[name] = self.run(name, user_input)
#             except Exception as e:
#                 logger.error(f"Agent '{name}' crashed inside run_all: {e}")
#                 outputs[name] = {
#                     "status": "ERROR",
#                     "agent": name,
#                     "error": str(e)
#                 }

#         return outputs

# # agent_runner.py
# # Central registry + unified execution layer for all agents

# import logging
# from typing import Dict, Any

# # FIXED IMPORTS (correct for your folder structure)
# from agents.base_agent import BaseAgent
# from tools.toolbox import TOOLS
# from llm.llm_wrapper import LLM   # your Groq/Dummy unified wrapper

# # Try to import institutional wrapper factory if present
# try:
#     from agents.inst_wrappers import create_inst_wrappers
# except Exception:
#     create_inst_wrappers = None

# # Try to import regular wrapper factory if present
# try:
#     from agents.wrappers import create_wrapped_agents
# except Exception:
#     create_wrapped_agents = None


# logger = logging.getLogger(__name__)


# class AgentRunner:
#     """
#     Central manager for all agents.
#     - Registers agents by name
#     - Injects tools + LLM into every agent
#     - Runs agents safely
#     - Supports both: regular 5 agents + 9 institutional agents
#     """

#     def __init__(self):
#         self.agents: Dict[str, BaseAgent] = {}

#         # shared tools + shared LLM wrapper (Groq or Dummy)
#         self.tools = TOOLS
#         self.llm = LLM()

#         logger.info("AgentRunner initialized with unified tools + LLM.")

#         # --------------------------------------------------------------
#         # AUTO-REGISTER institutional agents (try inst_wrappers first,
#         # fallback to create_wrapped_agents if available)
#         # --------------------------------------------------------------
#         try:
#             inst_agents = {}
#             if create_inst_wrappers is not None:
#                 # prefer institutional wrappers factory if present
#                 try:
#                     # try calling with signature (tools=..., llm=...)
#                     inst_agents = create_inst_wrappers(tools=self.tools, llm=self.llm)
#                 except TypeError:
#                     # fallback: maybe factory only expects llm
#                     inst_agents = create_inst_wrappers(self.llm)
#             elif create_wrapped_agents is not None:
#                 try:
#                     inst_agents = create_wrapped_agents(tools=self.tools, llm=self.llm)
#                 except TypeError:
#                     inst_agents = create_wrapped_agents(self.llm)
#             else:
#                 logger.info("No agent factory found (create_inst_wrappers/create_wrapped_agents). Skipping auto-register.")

#             # register returned agents
#             for name, agent in (inst_agents or {}).items():
#                 try:
#                     self.register(name, agent)
#                 except Exception as e:
#                     logger.exception("Failed to register agent %s: %s", name, e)

#             if inst_agents:
#                 logger.info(f"Institutional agents registered: {list(inst_agents.keys())}")

#         except Exception as e:
#             logger.error(f"Failed to register institutional agents: {e}", exc_info=True)

#     # --------------------------------------------------------------
#     # REGISTER ANY AGENT (5 normal agents use this)
#     # --------------------------------------------------------------
#     def register(self, name: str, agent: BaseAgent):
#         if not isinstance(name, str):
#             raise ValueError("Agent name must be a string")

#         if not isinstance(agent, BaseAgent):
#             raise TypeError(f"Agent '{name}' must inherit BaseAgent")

#         self.agents[name] = agent
#         logger.info(f"Registered agent: {name}")

#     # --------------------------------------------------------------
#     # RUN ONE AGENT
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

#     # --------------------------------------------------------------
#     # RUN ALL AGENTS (optional but recommended)
#     # --------------------------------------------------------------
#     def run_all(self, user_input: Dict[str, Any]):
#         """
#         Runs EVERY registered agent.
#         Returns a dict: { agent_name: result }
#         """
#         outputs = {}
#         logger.info(f"Running all agents: {list(self.agents.keys())}")

#         for name, agent in self.agents.items():
#             try:
#                 outputs[name] = self.run(name, user_input)
#             except Exception as e:
#                 logger.error(f"Agent '{name}' crashed inside run_all: {e}")
#                 outputs[name] = {
#                     "status": "ERROR",
#                     "agent": name,
#                     "error": str(e)
#                 }

#         return outputs

# # agent_runner.py
# # Central registry + unified execution layer for all agents

# import logging
# from typing import Dict, Any
# import time

# # FIXED IMPORTS (correct for your folder structure)
# from agents.base_agent import BaseAgent
# from tools.toolbox import TOOLS
# from llm.llm_wrapper import LLM   # your Groq/Dummy unified wrapper

# # Try to import institutional wrapper factory if present
# try:
#     from agents.inst_wrappers import create_inst_wrappers
# except Exception:
#     create_inst_wrappers = None

# # Try to import regular wrapper factory if present
# try:
#     from agents.wrappers import create_wrapped_agents
# except Exception:
#     create_wrapped_agents = None


# logger = logging.getLogger(__name__)


# class AgentRunner:
#     """
#     Central manager for all agents.
#     - Registers agents by name
#     - Injects tools + LLM into every agent
#     - Runs agents safely
#     - Supports both: regular 5 agents + 9 institutional agents
#     """

#     def __init__(self):
#         self.agents: Dict[str, BaseAgent] = {}

#         # shared tools + shared LLM wrapper (Groq or Dummy)
#         self.tools = TOOLS
#         self.llm = LLM()

#         logger.info("🤖 AgentRunner initialized with unified tools + LLM")

#         # --------------------------------------------------------------
#         # AUTO-REGISTER institutional agents (try inst_wrappers first,
#         # fallback to create_wrapped_agents if available)
#         # --------------------------------------------------------------
#         try:
#             inst_agents = {}
#             if create_inst_wrappers is not None:
#                 # prefer institutional wrappers factory if present
#                 try:
#                     # try calling with signature (tools=..., llm=...)
#                     inst_agents = create_inst_wrappers(tools=self.tools, llm=self.llm)
#                 except TypeError:
#                     # fallback: maybe factory only expects llm
#                     inst_agents = create_inst_wrappers(self.llm)
#             elif create_wrapped_agents is not None:
#                 try:
#                     inst_agents = create_wrapped_agents(tools=self.tools, llm=self.llm)
#                 except TypeError:
#                     inst_agents = create_wrapped_agents(self.llm)
#             else:
#                 logger.info("No agent factory found. Starting empty AgentRunner.")

#             # register returned agents
#             for name, agent in (inst_agents or {}).items():
#                 try:
#                     self.register(name, agent)
#                 except Exception as e:
#                     logger.exception("Failed to register agent %s: %s", name, e)

#             if inst_agents:
#                 logger.info(f"✅ Agents registered: {list(inst_agents.keys())}")

#         except Exception as e:
#             logger.error(f"Failed to register agents: {e}", exc_info=True)

#     # --------------------------------------------------------------
#     # REGISTER ANY AGENT (5 normal agents use this)
#     # --------------------------------------------------------------
#     def register(self, name: str, agent: BaseAgent):
#         if not isinstance(name, str):
#             raise ValueError("Agent name must be a string")

#         if not isinstance(agent, BaseAgent):
#             raise TypeError(f"Agent '{name}' must inherit BaseAgent")

#         self.agents[name] = agent
#         logger.info(f"📋 Registered agent: {name}")

#     # --------------------------------------------------------------
#     # RUN ONE AGENT
#     # --------------------------------------------------------------
#     def run(self, name: str, user_input: Dict[str, Any]):
#         """
#         Run one agent and return its output.
#         Inject tools + LLM automatically.
#         """
#         agent = self.agents.get(name)
#         if not agent:
#             raise RuntimeError(f"Unknown agent: {name}")

#         logger.info(f"🚀 Running agent: {name}")

#         # Inject dependencies if missing
#         agent.tools = getattr(agent, "tools", None) or self.tools
#         agent.llm = getattr(agent, "llm", None) or self.llm

#         # Execute safely with timing
#         start_time = time.time()
#         try:
#             result = agent.run(user_input)
#             elapsed = time.time() - start_time
#             logger.info(f"✅ Agent '{name}' completed in {elapsed:.2f}s")
#             return result

#         except Exception as e:
#             elapsed = time.time() - start_time
#             logger.error(f"❌ Agent '{name}' failed after {elapsed:.2f}s: {e}", exc_info=True)
#             return {
#                 "status": "ERROR",
#                 "agent": name,
#                 "error": str(e)
#             }

#     # --------------------------------------------------------------
#     # RUN ALL AGENTS (optional but recommended)
#     # --------------------------------------------------------------
#     def run_all(self, user_input: Dict[str, Any]):
#         """
#         Runs EVERY registered agent.
#         Returns a dict: { agent_name: result }
#         """
#         outputs = {}
#         logger.info(f"Running all agents: {list(self.agents.keys())}")

#         for name, agent in self.agents.items():
#             try:
#                 outputs[name] = self.run(name, user_input)
#             except Exception as e:
#                 logger.error(f"Agent '{name}' crashed inside run_all: {e}")
#                 outputs[name] = {
#                     "status": "ERROR",
#                     "agent": name,
#                     "error": str(e)
#                 }

#         return outputs
    
#     # agent_runner.py - ADD THIS FUNCTION

# def register_core_agents(self):
#     """Manually register core agents as fallback"""
#     try:
#         from agents.wrappers import (
#             TechnicalAgent, RiskAgent, PortfolioAgent, 
#             DebateAgent, MasterAgent, NewsAgent
#         )
        
#         core_agents = {
#             "technical": TechnicalAgent(tools=self.tools, llm=self.llm),
#             "risk": RiskAgent(tools=self.tools, llm=self.llm),
#             "portfolio": PortfolioAgent(tools=self.tools, llm=self.llm),
#             "debate": DebateAgent(tools=self.tools, llm=self.llm),
#             "master": MasterAgent(tools=self.tools, llm=self.llm),
#             "news": NewsAgent(tools=self.tools, llm=self.llm),
#         }
        
#         for name, agent in core_agents.items():
#             self.register(name, agent)
            
#         logger.info(f"✅ Manually registered core agents: {list(core_agents.keys())}")
        
#     except Exception as e:
#         logger.error(f"❌ Failed to manually register core agents: {e}")

# agent_runner.py
# Central registry + unified execution layer for all agents

import logging
from typing import Dict, Any
import time

# FIXED IMPORTS (correct for your folder structure)
from agents.base_agent import BaseAgent
from tools.toolbox import TOOLS
from llm.llm_wrapper import LLM   # your Groq/Dummy unified wrapper

# Try to import institutional wrapper factory if present
try:
    from agents.inst_wrappers import create_inst_wrappers
except Exception:
    create_inst_wrappers = None

# Try to import regular wrapper factory if present
try:
    from agents.wrappers import create_wrapped_agents
except Exception:
    create_wrapped_agents = None

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

        logger.info("🤖 AgentRunner initialized with unified tools + LLM")

        # Debug registration first
        self.debug_agent_registration()
        self.registerAllAgents()
        
        # Try auto-registration first
        self.auto_register_agents()
        
        # If no agents registered, use manual fallback
        if not self.agents:
            logger.warning("No agents auto-registered, using manual fallback")
            self.register_core_agents()
        
        logger.info(f"🎯 Final registered agents: {list(self.agents.keys())}")

    def debug_agent_registration(self):
        """Debug function to check agent registration status"""
        logger.info("🔍 Debugging agent registration:")
        logger.info(f"   - create_inst_wrappers available: {create_inst_wrappers is not None}")
        logger.info(f"   - create_wrapped_agents available: {create_wrapped_agents is not None}")
        
        # Test if we can import the wrappers directly
        try:
            from agents.wrappers import create_wrapped_agents as test_create
            test_result = test_create(tools=self.tools, llm=self.llm)
            logger.info(f"   - Direct wrapper test: {list(test_result.keys())}")
        except Exception as e:
            logger.info(f"   - Direct wrapper test failed: {e}")

    def auto_register_agents(self):
        """Try to auto-register agents from factories"""
        try:
            inst_agents = {}
            if create_inst_wrappers is not None:
                try:
                    inst_agents = create_inst_wrappers(tools=self.tools, llm=self.llm)
                except TypeError:
                    inst_agents = create_inst_wrappers(self.llm)
            elif create_wrapped_agents is not None:
                try:
                    inst_agents = create_wrapped_agents(tools=self.tools, llm=self.llm)
                except TypeError:
                    inst_agents = create_wrapped_agents(self.llm)
            else:
                logger.info("No agent factory found.")
            
            # Register returned agents
            for name, agent in (inst_agents or {}).items():
                try:
                    self.register(name, agent)
                except Exception as e:
                    logger.exception("Failed to register agent %s: %s", name, e)
            
            if inst_agents:
                logger.info(f"✅ Auto-registered agents: {list(inst_agents.keys())}")
                
        except Exception as e:
            logger.error(f"Auto-registration failed: {e}")


    def registerAllAgents(self):


        from agents.wrappers import (
            TechnicalAgent, RiskAgent, PortfolioAgent, 
            DebateAgent, MasterAgent, NewsAgent,
            ProfessionalSentimentAgent
            )
            
        core_agents = {
            "technical": TechnicalAgent(tools=self.tools, llm=self.llm),
            "risk": RiskAgent(tools=self.tools, llm=self.llm),
            "portfolio": PortfolioAgent(tools=self.tools, llm=self.llm),
            "debate": DebateAgent(tools=self.tools, llm=self.llm),
            "master": MasterAgent(tools=self.tools, llm=self.llm),
            "news": NewsAgent(tools=self.tools, llm=self.llm),
            "sentiment": ProfessionalSentimentAgent(tools=self.tools, llm=self.llm),
        }
        
        for name, agent in core_agents.items():
                self.register(name,agent)

    def register_core_agents(self):
        """Manually register core agents as fallback"""
        try:
            from agents.wrappers import (
                TechnicalAgent, RiskAgent, PortfolioAgent, 
                DebateAgent, MasterAgent, NewsAgent,
                ProfessionalSentimentAgent
            )
            
            core_agents = {
                "technical": TechnicalAgent(tools=self.tools, llm=self.llm),
                "risk": RiskAgent(tools=self.tools, llm=self.llm),
                "portfolio": PortfolioAgent(tools=self.tools, llm=self.llm),
                "debate": DebateAgent(tools=self.tools, llm=self.llm),
                "master": MasterAgent(tools=self.tools, llm=self.llm),
                "news": NewsAgent(tools=self.tools, llm=self.llm),
                "sentiment": ProfessionalSentimentAgent(tools=self.tools, llm=self.llm),
            }
            
            for name, agent in core_agents.items():
                self.register(name, agent)
                
            logger.info(f"✅ Manually registered core agents: {list(core_agents.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Failed to manually register core agents: {e}")

    # --------------------------------------------------------------
    # REGISTER ANY AGENT (5 normal agents use this)
    # --------------------------------------------------------------
    def register(self, name: str, agent: BaseAgent):
        if not isinstance(name, str):
            raise ValueError("Agent name must be a string")

        if not isinstance(agent, BaseAgent):
            raise TypeError(f"Agent '{name}' must inherit BaseAgent")

        self.agents[name] = agent
        logger.info(f"📋 Registered agent: {name}")



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
            print(f"self.agents: {self.agents}")
            raise RuntimeError(f"Unknown agent: {name}")
            

        logger.info(f"🚀 Running agent: {name}")

        # Inject dependencies if missing
        agent.tools = getattr(agent, "tools", None) or self.tools
        agent.llm = getattr(agent, "llm", None) or self.llm

        # Execute safely with timing
        start_time = time.time()
        try:
            result = agent.run(user_input)
            elapsed = time.time() - start_time
            logger.info(f"✅ Agent '{name}' completed in {elapsed:.2f}s")
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Agent '{name}' failed after {elapsed:.2f}s: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "agent": name,
                "error": str(e)
            }
        

    # Add this to agent_runner.py for testing
def test_wrapper_factory(self):
    """Test if the wrapper factory is working"""
    try:
        from agents.wrappers import create_wrapped_agents
        test_agents = create_wrapped_agents(tools=self.tools, llm=self.llm)
        logger.info(f"🧪 Factory test: Created {len(test_agents)} agents: {list(test_agents.keys())}")
        return True
    except Exception as e:
        logger.error(f"🧪 Factory test failed: {e}")
        return False

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