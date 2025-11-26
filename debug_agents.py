# debug_agents.py
import sys
import os
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing critical imports...")
    
    tests = [
        ("agents.base_agent", "BaseAgent"),
        ("agents.technical_agent", "TechnicalAnalysisAgent"), 
        ("agents.wrappers", "create_wrapped_agents"),
        ("agent_runner", "AgentRunner"),
        ("tools.toolbox", "TOOLS"),
        ("llm.llm_wrapper", "LLM"),
    ]
    
    for module, item in tests:
        try:
            imported_module = __import__(module, fromlist=[item])
            obj = getattr(imported_module, item)
            print(f"✅ {module}.{item}")
        except Exception as e:
            print(f"❌ {module}.{item}: {e}")

def test_agent_runner():
    """Test AgentRunner initialization"""
    print("\n🧪 Testing AgentRunner...")
    try:
        from agent_runner import AgentRunner
        runner = AgentRunner()
        print(f"✅ AgentRunner created with {len(runner.agents)} agents")
        print(f"   Agents: {list(runner.agents.keys())}")
        return runner
    except Exception as e:
        print(f"❌ AgentRunner failed: {e}")
        return None

def test_technical_agent_directly():
    """Test technical agent directly"""
    print("\n🧪 Testing TechnicalAgent directly...")
    try:
        from agents.technical_agent import TechnicalAnalysisAgent
        agent = TechnicalAnalysisAgent()
        print("✅ TechnicalAnalysisAgent created")
        
        # Test analysis
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=30)
        result = agent.analyze("AAPL", start, end)
        print(f"✅ Technical analysis: {result.get('action', 'UNKNOWN')}")
        return True
    except Exception as e:
        print(f"❌ TechnicalAgent failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Starting agent diagnostics...")
    test_imports()
    runner = test_agent_runner()
    test_technical_agent_directly()
    
    if runner and "technical" in runner.agents:
        print("\n🎉 SUCCESS: Technical agent is properly registered!")
    else:
        print("\n❌ FAILED: Technical agent is NOT registered!")