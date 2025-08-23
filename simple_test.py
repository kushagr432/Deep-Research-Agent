#!/usr/bin/env python3
"""
Simple test to verify Yahoo Finance fix
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_simple():
    """Simple test of the fix"""
    try:
        from agents.deep_research_agent import DeepResearchAgent
        
        # Mock LLM client
        class MockLLM:
            def __call__(self, prompt):
                return "Mock response for testing"
        
        # Initialize agent
        agent = DeepResearchAgent(MockLLM())
        
        print("✅ Agent initialized successfully")
        print("🔍 Testing ticker extraction...")
        
        # Test ticker extraction directly
        tickers = agent._extract_tickers_from_query("generate a report on nifty 50 in 2025")
        print(f"📊 Tickers found: {tickers}")
        
        if '^NSEI' in tickers:
            print("✅ Nifty 50 ticker extraction working!")
        else:
            print("❌ Nifty 50 ticker extraction failed")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Simple test of Yahoo Finance fix...")
    asyncio.run(test_simple())
    print("✨ Test completed!")
