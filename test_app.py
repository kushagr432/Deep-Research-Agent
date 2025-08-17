#!/usr/bin/env python3
"""
Simple test script for Financial Research Chatbot
"""
import asyncio
import json
import time
from app.main import app
from app.services.cache import CacheService
from app.services.queue import QueueService
from app.services.vector_db import VectorDBService
from app.agents.finance_agent import FinanceAgent
from app.agents.banking_agent import BankingAgent

async def test_services():
    """Test individual services"""
    print("🧪 Testing Services...")
    
    # Test Cache Service
    print("\n1. Testing Cache Service...")
    cache = CacheService()
    try:
        await cache.connect()
        await cache.set("test_key", "test_value", 60)
        value = await cache.get("test_key")
        print(f"   ✅ Cache test: {value}")
        await cache.disconnect()
    except Exception as e:
        print(f"   ❌ Cache test failed: {e}")
    
    # Test Vector DB Service
    print("\n2. Testing Vector DB Service...")
    vector_db = VectorDBService()
    try:
        await vector_db.connect()
        results = await vector_db.search("investment portfolio", top_k=3)
        print(f"   ✅ Vector DB test: Found {len(results)} results")
        await vector_db.disconnect()
    except Exception as e:
        print(f"   ❌ Vector DB test failed: {e}")
    
    # Test Queue Service
    print("\n3. Testing Queue Service...")
    queue = QueueService()
    try:
        await queue.connect()
        success = await queue.publish_message("test_topic", {"message": "test"})
        print(f"   ✅ Queue test: Message published: {success}")
        await queue.disconnect()
    except Exception as e:
        print(f"   ❌ Queue test failed: {e}")
    
    # Test Agents
    print("\n4. Testing Agents...")
    finance_agent = FinanceAgent()
    banking_agent = BankingAgent()
    
    try:
        finance_info = await finance_agent.get_agent_info()
        banking_info = await banking_agent.get_agent_info()
        print(f"   ✅ Finance Agent: {finance_info['name']}")
        print(f"   ✅ Banking Agent: {banking_info['name']}")
    except Exception as e:
        print(f"   ❌ Agent test failed: {e}")

async def test_api_endpoints():
    """Test API endpoints"""
    print("\n🌐 Testing API Endpoints...")
    
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # Test health endpoint
    try:
        response = client.get("/health")
        print(f"   ✅ Health endpoint: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"      Status: {health_data.get('status', 'unknown')}")
    except Exception as e:
        print(f"   ❌ Health endpoint failed: {e}")
    
    # Test root endpoint
    try:
        response = client.get("/")
        print(f"   ✅ Root endpoint: {response.status_code}")
        if response.status_code == 200:
            root_data = response.json()
            print(f"      Message: {root_data.get('message', 'unknown')}")
    except Exception as e:
        print(f"   ❌ Root endpoint failed: {e}")

def test_imports():
    """Test that all modules can be imported"""
    print("📦 Testing Imports...")
    
    try:
        from app.main import app
        print("   ✅ app.main imported successfully")
    except Exception as e:
        print(f"   ❌ app.main import failed: {e}")
    
    try:
        from app.graph import FinancialResearchGraph
        print("   ✅ app.graph imported successfully")
    except Exception as e:
        print(f"   ❌ app.graph import failed: {e}")
    
    try:
        from app.agents.finance_agent import FinanceAgent
        print("   ✅ FinanceAgent imported successfully")
    except Exception as e:
        print(f"   ❌ FinanceAgent import failed: {e}")
    
    try:
        from app.agents.banking_agent import BankingAgent
        print("   ✅ BankingAgent imported successfully")
    except Exception as e:
        print(f"   ❌ BankingAgent import failed: {e}")
    
    try:
        from app.services.cache import CacheService
        print("   ✅ CacheService imported successfully")
    except Exception as e:
        print(f"   ❌ CacheService import failed: {e}")
    
    try:
        from app.services.queue import QueueService
        print("   ✅ QueueService imported successfully")
    except Exception as e:
        print(f"   ❌ QueueService import failed: {e}")
    
    try:
        from app.services.vector_db import VectorDBService
        print("   ✅ VectorDBService imported successfully")
    except Exception as e:
        print(f"   ❌ VectorDBService import failed: {e}")

async def main():
    """Main test function"""
    print("🚀 Financial Research Chatbot - Test Suite")
    print("=" * 50)
    
    # Test imports first
    test_imports()
    
    # Test services
    await test_services()
    
    # Test API endpoints
    await test_api_endpoints()
    
    print("\n" + "=" * 50)
    print("✨ Test suite completed!")
    print("\nTo run the application:")
    print("  python -m app.main")
    print("\nOr:")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")

if __name__ == "__main__":
    asyncio.run(main())
