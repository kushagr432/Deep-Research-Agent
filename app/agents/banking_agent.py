# banking_agent.py
from .base_agent import BaseAgent
from typing import Dict, Any, AsyncGenerator
import time
import asyncio

class BankingAgent(BaseAgent):
    """Agent specialized in banking-related queries"""
    
    def __init__(self, llm_client):
        super().__init__(llm_client)
        self.specializations = [
            "account management",
            "loans and mortgages", 
            "credit cards",
            "investment accounts",
            "online banking",
            "financial planning"
        ]
    
    async def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process banking-related queries"""
        start_time = self._log_start("banking analysis", query)
        
        # Create specialized banking prompt
        prompt = self._create_banking_prompt(query)
        
        # Call LLM
        llm_start = self._log_llm_call(prompt, start_time)
        response = self.llm_client(prompt)
        response = self._log_llm_response(response, llm_start)
        
        # Log completion
        total_time = time.time() - start_time
        self._log_completion("banking analysis", total_time)
        
        return {
            "response": response,
            "agent": "banking",
            "specializations": self.specializations,
            "processing_time": total_time
        }
    
    async def process_stream(self, query: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Process banking queries with streaming updates"""
        start_time = self._log_start("banking analysis", query)
        
        # Stream: Starting analysis
        yield {
            "type": "status",
            "message": "Starting banking analysis...",
            "step": "start"
        }
        
        # Stream: Creating prompt
        yield {
            "type": "status", 
            "message": "Analyzing your banking query...",
            "step": "analysis"
        }
        
        # Create specialized banking prompt
        prompt = self._create_banking_prompt(query)
        
        # Stream: Calling LLM
        yield {
            "type": "status",
            "message": "Generating expert banking advice...",
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
        self._log_completion("banking analysis", total_time)
        
        # Stream: Final result
        yield {
            "type": "final_result",
            "response": response,
            "agent": "banking",
            "specializations": self.specializations,
            "processing_time": total_time
        }
    
    def _create_banking_prompt(self, query: str) -> str:
        """Create a specialized banking prompt"""
        return f"""You are a banking expert. Provide comprehensive advice on: {query}

Please include:
1. Key banking concepts and principles
2. Best practices and recommendations
3. Potential risks and considerations
4. Steps to take or next actions
5. When to consult a professional

<Hard Limits>
*Tool Call Budgets* (Prevent excessive searching):
- *Simple queries*: Use 2-3 search tool calls maximum
- *Complex queries*: Use up to 5 search tool calls maximum
- *Always stop*: After 5 search tool calls if you cannot find the right sources

*Stop Immediately When*:
- You can answer the user's question comprehensively
- You have 3+ relevant examples/sources for the question
- Your last 2 searches returned similar information
</Hard Limits>

Focus on practical, actionable banking advice that helps users make informed decisions."""

    def _get_capabilities(self) -> list:
        """Get banking agent capabilities"""
        return [
            "query_processing",
            "banking_expertise", 
            "financial_advice",
            "risk_assessment"
        ]
