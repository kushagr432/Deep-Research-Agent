# finance_agent.py
from .base_agent import BaseAgent
from typing import Dict, Any, AsyncGenerator
import time
import asyncio

class FinanceAgent(BaseAgent):
    """Agent specialized in general finance queries"""
    
    def __init__(self, llm_client):
        super().__init__(llm_client)
        self.specializations = [
            "investment strategies",
            "portfolio management",
            "retirement planning",
            "tax planning",
            "risk management",
            "market analysis",
            "financial education"
        ]
    
    async def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process finance-related queries"""
        start_time = self._log_start("finance analysis", query)
        
        # Create specialized finance prompt
        prompt = self._create_finance_prompt(query)
        
        # Call LLM
        llm_start = self._log_llm_call(prompt, start_time)
        response = self.llm_client(prompt)
        response = self._log_llm_response(response, llm_start)
        
        # Log completion
        total_time = time.time() - start_time
        self._log_completion("finance analysis", total_time)
        
        return {
            "response": response,
            "agent": "finance",
            "specializations": self.specializations,
            "processing_time": total_time
        }
    
    async def process_stream(self, query: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Process finance queries with streaming updates"""
        start_time = self._log_start("finance analysis", query)
        
        # Stream: Starting analysis
        yield {
            "type": "status",
            "message": "Starting financial analysis...",
            "step": "start"
        }
        
        # Stream: Creating prompt
        yield {
            "type": "status", 
            "message": "Analyzing your financial query...",
            "step": "analysis"
        }
        
        # Create specialized finance prompt
        prompt = self._create_finance_prompt(query)
        
        # Stream: Calling LLM
        yield {
            "type": "status",
            "message": "Generating expert financial advice...",
            "step": "llm_call"
        }
        
        # Call LLM
        llm_start = self._log_llm_call(prompt, start_time)
        response = self.llm_client(prompt)
        response = self._log_llm_response(response, llm_start)
        
        # Stream: Response chunks (simulate streaming by splitting into sentences)
        sentences = response.split('. ')
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                yield {
                    "type": "response_chunk",
                    "content": sentence.strip() + ('.' if i < len(sentences) - 1 else ''),
                    "chunk_index": i,
                    "total_chunks": len(sentences)
                }
                await asyncio.sleep(0.1)  # Small delay for realistic streaming effect
        
        # Log completion
        total_time = time.time() - start_time
        self._log_completion("finance analysis", total_time)
        
        # Stream: Final result
        yield {
            "type": "final_result",
            "response": response,
            "agent": "finance",
            "specializations": self.specializations,
            "processing_time": total_time
        }
    
    def _create_finance_prompt(self, query: str) -> str:
        """Create a specialized finance prompt"""
        return f"""You are a financial advisor and investment expert. Provide comprehensive analysis on: {query}

Please include:
1. Key financial concepts and principles
2. Investment strategies and approaches
3. Risk assessment and management
4. Market considerations and trends
5. Long-term planning advice
6. When to seek professional guidance

Focus on providing educational, actionable financial insights that help users build wealth and achieve financial goals."""

    def _get_capabilities(self) -> list:
        """Get finance agent capabilities"""
        return [
            "query_processing",
            "investment_expertise",
            "financial_planning",
            "risk_management",
            "market_analysis"
        ]
