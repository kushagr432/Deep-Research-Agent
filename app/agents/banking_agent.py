"""
Banking Agent for handling banking and loan-related queries
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
import ollama

logger = logging.getLogger(__name__)

class BankingAgent:
    """Agent specialized in banking, loans, and credit-related queries"""
    
    def __init__(self):
        self.name = "BankingAgent"
        self.specialization = "Banking and Credit Services"
        self.version = "1.0.0"
        
        # Mock banking knowledge base
        self.banking_knowledge = {
            "loans": [
                "Compare loan terms and interest rates from multiple lenders",
                "Understand the difference between fixed and variable interest rates",
                "Consider the total cost of the loan, not just the monthly payment"
            ],
            "credit_cards": [
                "Pay your credit card balance in full each month to avoid interest",
                "Keep your credit utilization below 30% of your available credit",
                "Monitor your credit score regularly and report any errors"
            ],
            "mortgages": [
                "Save for a down payment of at least 20% to avoid PMI",
                "Get pre-approved for a mortgage before house hunting",
                "Consider both the interest rate and closing costs when comparing offers"
            ],
            "savings": [
                "High-yield savings accounts offer better interest rates than traditional banks",
                "Consider setting up automatic transfers to build your savings habit",
                "Emergency funds should cover 3-6 months of essential expenses"
            ]
        }
    
    async def process_query(self, query: str, vector_results: List[Dict[str, Any]]) -> str:
        """Process a banking-related query using knowledge and vector search"""
        try:
            logger.info(f"BankingAgent processing query: {query[:100]}...")
            domain = self._classify_banking_domain(query)
            knowledge = self._get_relevant_knowledge(query, domain, vector_results)
            response = await self._generate_response(query, knowledge, domain)
            logger.info("BankingAgent completed processing query")
            return response
        except Exception as e:
            logger.error(f"Error in BankingAgent: {e}")
            return self._get_fallback_response(query)
    
    def _classify_banking_domain(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ["loan", "borrow", "lending", "debt"]):
            return "loans"
        elif any(word in query_lower for word in ["credit card", "credit limit", "credit score"]):
            return "credit_cards"
        elif any(word in query_lower for word in ["mortgage", "home loan", "house loan", "refinance"]):
            return "mortgages"
        elif any(word in query_lower for word in ["savings", "checking", "account", "deposit"]):
            return "savings"
        elif any(word in query_lower for word in ["bank", "branch", "atm", "online banking"]):
            return "banking_services"
        else:
            return "general_banking"
    
    def _get_relevant_knowledge(self, query: str, domain: str, vector_results: List[Dict[str, Any]]) -> List[str]:
        knowledge = []
        if domain in self.banking_knowledge:
            knowledge.extend(self.banking_knowledge[domain])
        if vector_results:
            for result in vector_results[:3]:
                if "content" in result:
                    knowledge.append(result["content"])
                elif "text" in result:
                    knowledge.append(result["text"])
        knowledge.extend([
            "Always read the fine print and understand all terms before signing",
            "Shop around and compare offers from multiple financial institutions",
            "Maintain good financial habits like paying bills on time"
        ])
        return knowledge
    
    async def _generate_response(self, query: str, knowledge: List[str], domain: str) -> str:
        """Non-streaming response with LangChain Ollama"""
        try:
            from langchain_community.llms import Ollama
            llm = Ollama(model="llama3:8b", temperature=0.7, top_p=0.9)
            prompt = f"""
            You are a knowledgeable banking and credit advisor. Answer the following question clearly and helpfully.

            Question: {query}
            Banking Domain: {domain}

            Context Information:
            {chr(10).join([f"- {point}" for point in knowledge[:3]])}

            Instructions:
            1. Provide a clear, educational answer about {query}
            2. Use the context information to enhance your response
            3. Include practical examples when relevant
            4. Keep the tone professional but accessible
            5. Focus on banking, loans, credit, and financial services
            6. End with a brief disclaimer about consulting your bank

            Answer:
            """
            response = await llm.agenerate([prompt])
            generated_text = response.generations[0][0].text.strip()
            if generated_text:
                if domain not in generated_text.lower():
                    generated_text = f"Regarding your {domain.replace('_', ' ')} question:\n\n{generated_text}"
                return generated_text
            else:
                return self._get_fallback_response(query)
        except Exception as e:
            logger.error(f"Error calling Llama3:8b: {e}")
            return self._get_mock_response(query, knowledge, domain)
    
    async def generate_response_streaming(self, query: str, knowledge: List[str], domain: str):
        """Live streaming response using Ollama async client"""
        try:
            knowledge_context = "\n".join(knowledge[:5]) if knowledge else "General banking knowledge"
            prompt = f"""You are a helpful banking and credit advisor.
Answer the following question step by step:

Question: {query}
Domain: {domain}
Relevant Context: {knowledge_context}

Keep responses concise, structured, and user-friendly.
"""
            client = ollama.AsyncClient()
            
            # Try different model names in case llama3:8b is not available
            models_to_try = ["llama3:8b", "llama3", "llama2", "llama2:13b", "llama2:7b"]
            stream = None
            
            for model_name in models_to_try:
                try:
                    logger.info(f"Attempting to use model: {model_name}")
                    stream = await client.chat(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True
                    )
                    logger.info(f"Successfully connected to model: {model_name}")
                    break
                except Exception as model_error:
                    logger.warning(f"Failed to use model {model_name}: {model_error}")
                    continue
            
            if stream is None:
                logger.error("All models failed, falling back to mock response")
                yield self._get_mock_response(query, knowledge, domain)
                return
            
            logger.info(f"Starting streaming response for query: {query[:50]}...")
            
            async for chunk in stream:
                logger.debug(f"Received chunk: {type(chunk)} - {chunk}")
                
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    # Ensure we're yielding a string, not a dict
                    if isinstance(content, str):
                        yield content
                    else:
                        # Convert to string if it's not already
                        logger.warning(f"Received non-string content: {type(content)} - {content}")
                        yield str(content)
                else:
                    logger.debug(f"Unexpected chunk format: {chunk}")
                    
        except Exception as e:
            logger.error(f"Streaming error: {e} | Query: {query[:50]}...")
            # Fallback to mock response instead of error message
            fallback_response = self._get_mock_response(query, knowledge, domain)
            # Split into sentences for streaming
            sentences = fallback_response.split('. ')
            for sentence in sentences:
                if sentence.strip():
                    yield sentence.strip() + '.'
    
    def _get_mock_response(self, query: str, knowledge: List[str], domain: str) -> str:
        response_parts = [f"Regarding your {domain.replace('_', ' ')} question..."]
        if knowledge:
            response_parts.append("\nHere are some important considerations:")
            for i, point in enumerate(knowledge[:3], 1):
                response_parts.append(f"{i}. {point}")
        if domain == "loans":
            response_parts.append("\n🏦 Loan Tip: Always calculate the total cost of borrowing, including fees and interest.")
        elif domain == "credit_cards":
            response_parts.append("\n💳 Credit Card Tip: Set up automatic payments to avoid late fees and protect your credit score.")
        elif domain == "mortgages":
            response_parts.append("\n🏠 Mortgage Tip: Consider the long-term costs and ensure the payment fits comfortably in your budget.")
        elif domain == "savings":
            response_parts.append("\n💰 Savings Tip: Automate your savings to make it a habit rather than an afterthought.")
        response_parts.append("\n⚠️ Disclaimer: This information is educational. Consult your bank for personalized advice.")
        return " ".join(response_parts)
    
    def _get_fallback_response(self, query: str) -> str:
        return (
            f"I apologize, but I'm having trouble processing your banking query about '{query}'. "
            "This might be due to a temporary issue. Please try rephrasing your question "
            "or contact your bank directly for specific account-related inquiries."
        )
    
    async def get_agent_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "specialization": self.specialization,
            "version": self.version,
            "capabilities": [
                "Loan and credit guidance",
                "Mortgage advice and education",
                "Credit card best practices",
                "Savings and account management",
                "General banking education"
            ],
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_banking_products(self) -> Dict[str, List[str]]:
        return {
            "loans": ["Personal Loans", "Auto Loans", "Student Loans", "Business Loans"],
            "credit_cards": ["Rewards Cards", "Balance Transfer Cards", "Secured Cards", "Business Cards"],
            "mortgages": ["Conventional Loans", "FHA Loans", "VA Loans", "Jumbo Loans"],
            "accounts": ["Checking Accounts", "Savings Accounts", "CDs", "Money Market Accounts"]
        }
