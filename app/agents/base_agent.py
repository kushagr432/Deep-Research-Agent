# base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator
import time
import asyncio

class BaseAgent(ABC):
    """Base class for all financial research agents"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.agent_name = self.__class__.__name__
    
    @abstractmethod
    async def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a query and return results"""
        pass
     
    def _log_start(self, operation: str, query: str):
        """Log the start of an operation"""
        print(f"🚀 [{self.agent_name}] Starting {operation}: '{query[:100]}...'")
        return time.time()
    
    def _log_llm_call(self, prompt: str, start_time: float):
        """Log LLM call with timing"""
        print(f"🤖 [{self.agent_name}] Calling LLM with prompt: {prompt[:100]}...")
        return time.time()
    
    def _log_llm_response(self, response: str, start_time: float):
        """Log LLM response with timing"""
        llm_time = time.time() - start_time
        print(f"💡 [{self.agent_name}] LLM response generated in {llm_time:.2f}s")
        print(f"📝 [{self.agent_name}] Response length: {len(response)} characters")
        return response
    
    def _log_completion(self, operation: str, total_time: float):
        """Log completion of operation"""
        print(f"✅ [{self.agent_name}] {operation} completed in {total_time:.2f}s")
    
    def get_agent_info(self) -> Dict[str, str]:
        """Get information about this agent"""
        return {
            "name": self.agent_name,
            "type": "financial_agent",
            "capabilities": self._get_capabilities()
        }
    
    def _get_capabilities(self) -> list:
        """Get list of agent capabilities - override in subclasses"""
        return ["query_processing"]
