#!/usr/bin/env python3
"""
Test script for the FinancialResearchGraph
"""
import asyncio
from app.graph import FinancialResearchGraph

async def test_graph():
    """Test the graph with different queries"""
    graph = FinancialResearchGraph()
    
    # Test 1: Simple banking query
    print("=== Test 1: Banking Query ===")
    result1 = await graph.process("How do I open a savings account?", state={"deep_research": False})
    print(f"Response: {result1.get('response')}")
    print(f"Intent: {result1.get('intent')}")
    print()
    
    # Test 2: Deep research query
    print("=== Test 2: Deep Research Query ===")
    result2 = await graph.process("What are the current trends in cryptocurrency markets?", state={"deep_research": True})
    print(f"Response: {result2.get('response')}")
    print(f"Web Data: {result2.get('web_data', 'N/A')[:200]}...")
    print()
    
    # Test 3: Insufficient query
    print("=== Test 3: Insufficient Query ===")
    result3 = await graph.process("Hi", state={"deep_research": False})
    print(f"Response: {result3.get('response')}")
    print()

if __name__ == "__main__":
    asyncio.run(test_graph())
