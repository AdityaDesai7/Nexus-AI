# agents/test_master_ai_agent.py
import logging
import json
import os
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# ===== REPLACE THIS WITH YOUR ACTUAL GROQ API KEY =====
GROQ_API_KEY = "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"  # ⬅️ REPLACE THE ... WITH YOUR FULL KEY
# =======================================================

print("🚀 TEST MASTER AGENT - STARTING...")
print(f"🔑 API Key length: {len(GROQ_API_KEY)}")
print(f"🔑 API Key starts with: {GROQ_API_KEY[:10]}...")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
    print("✅ Groq package is available")
except ImportError as e:
    print(f"❌ Groq import failed: {e}")
    GROQ_AVAILABLE = False

class TestMasterAgent:
    """
    TEST MASTER AGENT - For debugging Groq API integration
    """

    def __init__(self):
        self.client = None
        # Current working Groq models
        self.model = "llama-3.1-8b-instant"  # This should work
        
        print("\n🔧 ===== TEST MASTER AGENT INITIALIZATION =====")
        print(f"🔧 Using model: {self.model}")
        
        if not GROQ_AVAILABLE:
            print("❌ Groq package not available")
            return
        
        try:
            print("🔧 Creating Groq client...")
            self.client = Groq(api_key=GROQ_API_KEY)
            print("✅ Groq client created")
            
            # Test the connection
            print("🔧 Testing Groq connection...")
            test_response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": "Say just 'SUCCESS' in one word."}],
                model=self.model,
                max_tokens=10,
                temperature=0.1
            )
            test_result = test_response.choices[0].message.content
            print(f"✅ Groq connection test successful: {test_result}")
            
        except Exception as e:
            print(f"❌ Groq initialization failed: {e}")
            self.client = None

    def test_simple_call(self):
        """Test a simple Groq API call"""
        print("\n🔧 ===== TEST SIMPLE CALL =====")
        
        if not self.client:
            print("❌ No Groq client available")
            return False
            
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
                model=self.model,
                max_tokens=10
            )
            result = response.choices[0].message.content
            print(f"✅ Simple call successful: {result}")
            return True
        except Exception as e:
            print(f"❌ Simple call failed: {e}")
            return False

    def test_trading_analysis(self, ticker="NTPC.NS", current_price=3162.9):
        """Test trading analysis with Groq"""
        print(f"\n🔧 ===== TEST TRADING ANALYSIS for {ticker} =====")
        
        if not self.client:
            print("❌ No Groq client available")
            return None
            
        # Mock data for testing
        technical_data = {
            "action": "BUY",
            "confidence": 75,
            "rsi": 65.5,
            "macd_hist": 0.5,
            "support": 3100,
            "resistance": 3200
        }
        
        sentiment_data = {
            "overall_sentiment": "positive",
            "overall_confidence": 80
        }
        
        risk_data = {
            "risk_level": "MEDIUM",
            "stop_loss_price": 3094.37,
            "take_profit_price": 3299.96
        }
        
        prompt = self._build_test_prompt(ticker, technical_data, sentiment_data, risk_data, current_price)
        
        try:
            print("📡 Sending trading analysis request...")
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            print("✅ Trading analysis successful")
            print(f"📄 Raw response: {result_text}")
            
            result = json.loads(result_text)
            return result
            
        except Exception as e:
            print(f"❌ Trading analysis failed: {e}")
            return None

    def _build_test_prompt(self, ticker: str, technical: Dict, sentiment: Dict, risk: Dict, current_price: float) -> str:
        """Build test prompt"""
        
        prompt = f"""
You are a professional trading analyst. Analyze {ticker} and provide a trading decision.

Return ONLY a JSON object with EXACTLY these fields:
- ticker (string)
- current_price (number)
- action (string: "BUY", "SELL", or "HOLD")
- confidence (number between 50-95)
- reasoning (string: 1-2 sentences explaining the decision)
- entry_price (number: same as current_price)
- stop_loss (number)
- take_profit (number) 
- risk_reward_ratio (number with 1 decimal)
- quantity (integer)
- position_size (number with 3 decimals)
- risk_level (string: "LOW", "MEDIUM", "HIGH")
- timestamp (string: ISO format)
- status (string: "SUCCESS")
- ai_enhanced (boolean: true)

DATA:
- Ticker: {ticker}
- Current Price: ${current_price:.2f}
- Technical: {technical['action']} signal with {technical['confidence']}% confidence
- RSI: {technical['rsi']:.2f}
- MACD: {technical['macd_hist']:.4f}
- Support: ${technical['support']:.2f}, Resistance: ${technical['resistance']:.2f}
- Sentiment: {sentiment['overall_sentiment']} with {sentiment['overall_confidence']}% confidence
- Risk Level: {risk['risk_level']}
- Stop Loss: ${risk['stop_loss_price']:.2f}
- Take Profit: ${risk['take_profit_price']:.2f}

Return ONLY the JSON object, no other text.
"""
        return prompt

    def run_comprehensive_test(self):
        """Run all tests"""
        print("\n" + "="*50)
        print("🎯 COMPREHENSIVE GROQ API TEST")
        print("="*50)
        
        if not GROQ_AVAILABLE:
            print("❌ FAIL: Groq package not available")
            return False
            
        if not self.client:
            print("❌ FAIL: No Groq client available")
            return False
        
        # Test 1: Simple call
        print("\n1. Testing simple API call...")
        simple_test = self.test_simple_call()
        if not simple_test:
            print("❌ Simple call test FAILED")
            return False
        
        # Test 2: Trading analysis
        print("\n2. Testing trading analysis...")
        trading_result = self.test_trading_analysis()
        
        if trading_result:
            print("✅ Trading analysis test PASSED")
            print(f"📊 Result: {json.dumps(trading_result, indent=2)}")
            return True
        else:
            print("❌ Trading analysis test FAILED")
            return False


# ===== TEST RUNNER =====
if __name__ == "__main__":
    print("🧪 TEST MASTER AGENT - STANDALONE TEST")
    print("="*60)
    
    # Create test instance
    test_agent = TestMasterAgent()
    
    # Run comprehensive test
    success = test_agent.run_comprehensive_test()
    
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS PASSED! Your Groq API is working correctly.")
        print("💡 Now update your main master_ai_agent.py:")
        print("   - Use the same API key")
        print("   - Use model: 'llama-3.1-8b-instant'")
    else:
        print("💥 TESTS FAILED!")
        print("🔧 Possible issues:")
        print("   - API key is incorrect or incomplete")
        print("   - Network connectivity issue")
        print("   - Groq service temporarily down")
    print("="*60)