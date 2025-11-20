# trading_bot/agents/base_agent.py
from typing import Any, Dict
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BaseAgent:
    """
    Minimal, deterministic Agent base class.
    - plan(input) -> plan (dict)
    - act(plan) -> result (dict)
    - run(input) -> result (dict)  (does logging + trace)
    """

    def __init__(self, name: str, tools: Dict[str, Any] = None, llm: Any = None, system_prompt: str = ""):
        self.name = name
        self.tools = tools or {}
        self.llm = llm
        self.system_prompt = system_prompt
        self.trace = []

    def call_tool(self, tool_name: str, *args, **kwargs):
        """Call a registered tool. Tools are pure functions or callables."""
        tool = self.tools.get(tool_name)
        if tool is None:
            raise RuntimeError(f"[{self.name}] Unknown tool: {tool_name}")
        t0 = time.time()
        try:
            res = tool(*args, **kwargs)
            dur = time.time() - t0
            entry = {"tool": tool_name, "args": args, "kwargs": kwargs, "result": "OK", "duration": dur}
            self.trace.append(entry)
            logger.info(f"[{self.name}] tool {tool_name} called in {dur:.3f}s")
            return res
        except Exception as e:
            dur = time.time() - t0
            entry = {"tool": tool_name, "args": args, "kwargs": kwargs, "result": f"ERROR: {str(e)}", "duration": dur}
            self.trace.append(entry)
            logger.exception(f"[{self.name}] tool {tool_name} raised")
            raise

    def plan(self, user_input: Dict) -> Dict:
        """Return plan dict describing required tool calls & intents. Override in subclasses."""
        raise NotImplementedError

    def act(self, plan: Dict) -> Dict:
        """Execute the plan. Override in subclasses."""
        raise NotImplementedError

    def run(self, user_input: Dict) -> Dict:
        """Top-level runner: plan -> act, with trace and safety guard."""
        logger.info(f"[{self.name}] run started with input keys: {list(user_input.keys())}")
        plan = self.plan(user_input)
        if not isinstance(plan, dict):
            raise RuntimeError(f"[{self.name}] plan() must return dict")
        # attach plan to trace
        self.trace.append({"phase": "plan", "plan": plan})
        result = self.act(plan)
        self.trace.append({"phase": "result", "result": result})
        logger.info(f"[{self.name}] run finished")
        return {"agent": self.name, "result": result, "trace": self.trace}
