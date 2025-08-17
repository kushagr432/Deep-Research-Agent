"""
Finance Agent for handling general financial research queries
"""
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class FinanceAgent:
    """Agent specialized in general financial research and analysis"""
    
    def __init__(self):
        self.name = "FinanceAgent"
        self.specialization = "General Financial Research"
        self.version = "1.0.0"
        
        # Mock knowledge base (in production, this would be more sophisticated)
        self.financial_knowledge = {
            "investment": [
                "Diversification is key to managing investment risk",
                "Compound interest can significantly grow wealth over time",
                "Asset allocation should consider your risk tolerance and time horizon"
            ],
            "retirement": [
                "Start saving for retirement as early as possible",
                "Consider tax-advantaged accounts like 401(k)s and IRAs",
                "Regular rebalancing helps maintain your target asset allocation"
            ],
            "budgeting": [
                "Track your income and expenses to identify spending patterns",
                "Create an emergency fund covering 3-6 months of expenses",
                "Use the 50/30/20 rule: 50% needs, 30% wants, 20% savings"
            ]
        }
    
    async def process_query(self, query: str, vector_results: List[Dict[str, Any]]) -> str:
        """
        Process a financial query using available knowledge and vector search results
        
        Args:
            query: The user's financial question
            vector_results: Relevant documents from vector database
            
        Returns:
            Formatted response to the user's query
        """
        try:
            logger.info(f"FinanceAgent processing query: {query[:100]}...")
            
            # Analyze the query to determine the financial domain
            domain = self._classify_query_domain(query)
            
            # Get relevant knowledge
            knowledge = self._get_relevant_knowledge(query, domain, vector_results)
            
            # Generate response using mock LLM (in production, call actual LLM)
            response = await self._generate_response(query, knowledge, domain)
            
            logger.info(f"FinanceAgent completed processing query")
            return response
            
        except Exception as e:
            logger.error(f"Error in FinanceAgent: {e}")
            return self._get_fallback_response(query)
    
    def _classify_query_domain(self, query: str) -> str:
        """Classify the query into a financial domain"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["invest", "stock", "portfolio", "asset"]):
            return "investment"
        elif any(word in query_lower for word in ["retire", "401k", "ira", "pension"]):
            return "retirement"
        elif any(word in query_lower for word in ["budget", "save", "expense", "income"]):
            return "budgeting"
        elif any(word in query_lower for word in ["tax", "deduction", "credit"]):
            return "taxes"
        elif any(word in query_lower for word in ["debt", "loan", "credit card"]):
            return "debt_management"
        else:
            return "general"
    
    def _get_relevant_knowledge(self, query: str, domain: str, vector_results: List[Dict[str, Any]]) -> List[str]:
        """Get relevant knowledge for the query"""
        knowledge = []
        
        # Add domain-specific knowledge
        if domain in self.financial_knowledge:
            knowledge.extend(self.financial_knowledge[domain])
        
        # Add vector search results (simulated)
        if vector_results:
            for result in vector_results[:3]:  # Top 3 results
                if "content" in result:
                    knowledge.append(result["content"])
                elif "text" in result:
                    knowledge.append(result["text"])
        
        # Add general financial principles
        knowledge.extend([
            "Always consider your personal financial situation before making decisions",
            "Consult with qualified financial professionals for complex matters",
            "Past performance doesn't guarantee future results"
        ])
        
        return knowledge
    
    async def _generate_response(self, query: str, knowledge: List[str], domain: str) -> str:
        """Generate a response using local Llama3:8b model"""
        try:
            # Import Ollama integration
            from langchain_community.llms import Ollama
            
            # Initialize Llama3:8b model
            llm = Ollama(
                model="llama3:8b",
                temperature=0.7,
                top_p=0.9
            )
            
            # Create comprehensive prompt
            prompt = f"""
            You are a knowledgeable financial advisor. Answer the following question clearly and helpfully.

            Question: {query}
            Financial Domain: {domain}
            
            Context Information:
            {chr(10).join([f"- {point}" for point in knowledge[:3]])}
            
            Instructions:
            1. Provide a clear, educational answer about {query}
            2. Use the context information to enhance your response
            3. Include practical examples when relevant
            4. Keep the tone professional but accessible
            5. End with a brief disclaimer about consulting professionals
            
            Answer:
            """
            
            # Generate response using Llama3:8b
            response = await llm.agenerate([prompt])
            generated_text = response.generations[0][0].text.strip()
            
            # Clean up response
            if generated_text:
                # Add domain context if not present
                if domain not in generated_text.lower():
                    generated_text = f"Based on your question about {domain.replace('_', ' ')}:\n\n{generated_text}"
                
                return generated_text
            else:
                return self._get_fallback_response(query)
                
        except Exception as e:
            logger.error(f"Error calling Llama3:8b: {e}")
            # Fallback to mock response if LLM fails
            return self._get_mock_response(query, knowledge, domain)
    
    async def generate_response_streaming(self, query: str, knowledge: List[str], domain: str):
        """Generate real-time streaming response using local Llama3:8b model"""
        try:
            from langchain_community.llms import Ollama
            llm = Ollama(model="llama3:8b", temperature=0.7, top_p=0.9)
            
            # Create context-aware prompt
            knowledge_context = "\n".join(knowledge) if knowledge else "General financial knowledge"
            prompt = f"""You are a financial research expert. Answer the following question with detailed, helpful information.

Question: {query}
Domain: {domain}
Context: {knowledge_context}

Provide a comprehensive, well-structured response that is informative and actionable. Focus on practical insights and clear explanations.

Answer:"""

            try:
                # Stream response and accumulate into natural chunks
                response_stream = llm.stream(prompt)
                current_chunk = ""
                word_count = 0
                
                for chunk in response_stream:
                    if chunk and hasattr(chunk, 'content') and chunk.content:
                        current_chunk += chunk.content
                        word_count += len(chunk.content.split())
                    elif chunk and isinstance(chunk, str):
                        current_chunk += chunk
                        word_count += len(chunk.split())
                    elif chunk and hasattr(chunk, 'text') and chunk.text:
                        current_chunk += chunk.text
                        word_count += len(chunk.text.split())
                    
                    # Yield chunk when we have a complete sentence or reach ~15-20 words
                    if (word_count >= 15 or 
                        (current_chunk.strip() and current_chunk.strip()[-1] in '.!?') or
                        len(current_chunk) >= 100):
                        
                        if current_chunk.strip():
                            yield current_chunk.strip()
                            current_chunk = ""
                            word_count = 0
                
                # Yield any remaining content
                if current_chunk.strip():
                    yield current_chunk.strip()
                    
            except Exception as streaming_error:
                logger.warning(f"Native streaming failed, falling back to chunked response: {streaming_error}")
                response = await llm.agenerate([prompt])
                generated_text = response.generations[0][0].text.strip()
                if generated_text:
                    sentences = self._split_into_sentences(generated_text)
                    for sentence in sentences:
                        if sentence.strip():
                            yield sentence.strip()
                            await asyncio.sleep(0.1)
                else:
                    yield "I apologize, but I couldn't generate a response for your query. Please try rephrasing your question."
                    
        except Exception as e:
            logger.error(f"Error calling Llama3:8b: {e}")
            yield "I encountered an error while processing your request. Please try again."
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for natural chunking"""
        import re
        
        # Split on sentence endings (., !, ?) followed by space or newline
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean up and filter empty sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Only include meaningful sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def _get_mock_response(self, query: str, knowledge: List[str], domain: str) -> str:
        """Fallback mock response if LLM fails"""
        response_parts = []
        
        # Add personalized greeting
        response_parts.append(f"Based on your question about {domain.replace('_', ' ')}...")
        
        # Add relevant knowledge
        if knowledge:
            response_parts.append("\nHere are some key points to consider:")
            for i, point in enumerate(knowledge[:3], 1):
                response_parts.append(f"{i}. {point}")
        
        # Add domain-specific advice
        if domain == "investment":
            response_parts.append("\n💡 Investment Tip: Consider your risk tolerance and time horizon when making investment decisions.")
        elif domain == "retirement":
            response_parts.append("\n💡 Retirement Tip: The earlier you start saving, the more time compound interest has to work in your favor.")
        elif domain == "budgeting":
            response_parts.append("\n💡 Budgeting Tip: Track your spending for at least a month to identify areas where you can cut back.")
        
        # Add disclaimer
        response_parts.append("\n⚠️ Disclaimer: This information is for educational purposes only and should not be considered financial advice.")
        
        return " ".join(response_parts)
    
    def _get_fallback_response(self, query: str) -> str:
        """Provide a fallback response when processing fails"""
        return (
            f"I apologize, but I'm having trouble processing your query about '{query}'. "
            "This might be due to a temporary issue. Please try rephrasing your question "
            "or contact support if the problem persists."
        )
    
    async def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent"""
        return {
            "name": self.name,
            "specialization": self.specialization,
            "version": self.version,
            "capabilities": [
                "Investment analysis and advice",
                "Retirement planning guidance",
                "Budgeting and financial planning",
                "General financial education"
            ],
            "last_updated": datetime.now().isoformat()
        }
