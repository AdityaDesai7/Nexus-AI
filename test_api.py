# test_groq.py - TEMPORARY SOLUTION
import os
from groq import Groq

# TEMPORARY: Set your key directly here
os.environ["GROQ_API_KEY"] = "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"

def test_groq_key():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    print(f"API Key present: {bool(api_key)}")
    if api_key:
        print(f"Key starts with: {api_key[:8]}...")
    
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say 'Hello World'"}],
            max_tokens=10
        )
        print("✅ API Key is VALID!")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ API Key is INVALID: {e}")

if __name__ == "__main__":
    test_groq_key()