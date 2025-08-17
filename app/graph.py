"""
LangGraph workflow definition for Financial Research Chatbot
"""
import asyncio
import time
import logging
import os
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.finance_agent import FinanceAgent
from app.agents.banking_agent import BankingAgent
from app.services.vector_db import VectorDBService

logger = logging.getLogger(__name__)

class QueryState(TypedDict):
    """State object for the LangGraph workflow"""
    query: str
    normalized_query: str
    user_id: str
    deep_research: bool
    generate_pdf: bool
    cached_response: Optional[str]
    agent_response: Optional[str]
    final_answer: str
    cached: bool
    processing_time: float
    agent_used: str
    vector_results: List[Dict[str, Any]]
    errors: List[str]
    needs_clarification: bool
    research_type: str
    deep_research_results: Optional[Dict[str, Any]]
    report_path: Optional[str]
    report_filename: Optional[str]

class FinancialResearchGraph:
    """Main LangGraph workflow for financial research queries"""
    
    def __init__(self, cache_service, queue_service):
        self.cache_service = cache_service
        self.queue_service = queue_service
        self.vector_db = VectorDBService()
        self.finance_agent = FinanceAgent()
        self.banking_agent = BankingAgent()
        self.execution_stats = {
            "total_queries": 0,
            "cached_responses": 0,
            "agent_responses": 0,
            "errors": 0
        }
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(QueryState)
        
        # Add nodes
        workflow.add_node("clarify_query", self._clarify_query)
        workflow.add_node("normalize_query", self._normalize_query)
        workflow.add_node("check_cache", self._check_cache)
        workflow.add_node("route_to_agent", self._route_to_agent)
        workflow.add_node("fetch_knowledge", self._fetch_knowledge)
        workflow.add_node("process_with_agent", self._process_with_agent)
        workflow.add_node("deep_research_node", self._deep_research)
        workflow.add_node("generate_report", self._generate_report)
        workflow.add_node("save_to_cache", self._save_to_cache)
        workflow.add_node("finalize_response", self._finalize_response)
        
        # Define the flow
        workflow.set_entry_point("clarify_query")
        
        # Clarify query first, then decide path
        workflow.add_conditional_edges(
            "clarify_query",
            self._should_clarify,
            {
                "needs_clarification": END,
                "clear": "normalize_query"
            }
        )
        
        # Normalize query first
        workflow.add_edge("normalize_query", "check_cache")
        
        # Check cache - if cached, go to finalize, else continue
        workflow.add_conditional_edges(
            "check_cache",
            self._should_use_cache,
            {
                "cached": "finalize_response",
                "not_cached": "route_to_agent"
            }
        )
        
        # Route to appropriate agent
        workflow.add_edge("route_to_agent", "fetch_knowledge")
        workflow.add_edge("fetch_knowledge", "process_with_agent")
        
        # Add conditional edge for deep research
        workflow.add_conditional_edges(
            "process_with_agent",
            self._should_do_deep_research,
            {
                "deep_research": "deep_research_node",
                "standard": "save_to_cache"
            }
        )
        
        # Deep research path
        workflow.add_edge("deep_research_node", "generate_report")
        workflow.add_edge("generate_report", "save_to_cache")
        
        # Standard path
        workflow.add_edge("save_to_cache", "finalize_response")
        
        # Finalize and end
        workflow.add_edge("finalize_response", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    async def execute_query_streaming(self, query: str, user_id: str, deep_research: bool = False, generate_pdf: bool = False):
        """Execute the complete workflow with live streaming updates"""
        start_time = time.time()
        
        try:
            # Send initial status
            yield {'status': 'starting', 'message': 'Initializing research workflow...'}
            
            # Initialize state
            initial_state = {
                "query": query,
                "normalized_query": "",
                "user_id": user_id,
                "deep_research": deep_research,
                "generate_pdf": generate_pdf,
                "cached_response": None,
                "agent_response": None,
                "final_answer": "",
                "cached": False,
                "processing_time": 0.0,
                "agent_used": "",
                "vector_results": [],
                "errors": [],
                "needs_clarification": False,
                "research_type": "deep" if deep_research else "standard",
                "deep_research_results": None,
                "report_path": None,
                "report_filename": None
            }
            
            # Execute the graph with streaming updates
            config = {"configurable": {"thread_id": f"{user_id}_{int(start_time)}"}}
            
            # Stream through each node with progress updates
            async for progress_update in self._execute_graph_streaming(initial_state, config):
                yield progress_update
            
            # Calculate final processing time
            processing_time = time.time() - start_time
            
            # Send completion
            yield {'status': 'complete', 'processing_time': processing_time}
            
        except Exception as e:
            self.execution_stats["errors"] += 1
            logger.error(f"Error in graph execution: {e}")
            yield {'status': 'error', 'message': f'Graph execution failed: {str(e)}'}
    
    async def _execute_graph_streaming(self, initial_state: Dict[str, Any], config: Dict[str, Any]):
        """Execute graph nodes with streaming updates"""
        try:
            # Execute each node and yield progress
            current_state = initial_state.copy()
            
            # Clarify query
            current_state = await self._clarify_query(current_state)
            yield {'status': 'progress', 'step': 'clarify_query', 'message': 'Query clarity assessed', 'data': {'needs_clarification': current_state.get('needs_clarification')}}
            
            if current_state.get("needs_clarification"):
                return
            
            # Normalize query
            current_state = await self._normalize_query(current_state)
            yield {'status': 'progress', 'step': 'normalize_query', 'message': 'Query normalized', 'data': {'normalized_query': current_state.get('normalized_query')}}
            
            # Check cache
            current_state = await self._check_cache(current_state)
            yield {'status': 'progress', 'step': 'check_cache', 'message': 'Cache checked', 'data': {'cached': current_state.get('cached'), 'cached_response': current_state.get('cached_response')}}
            
            if current_state.get("cached_response"):
                current_state = await self._finalize_response(current_state)
                yield {'status': 'progress', 'step': 'finalize_response', 'message': 'Cached response finalized', 'data': {'final_answer': current_state.get('final_answer')}}
                return
            
            # Route to agent
            current_state = await self._route_to_agent(current_state)
            yield {'status': 'progress', 'step': 'route_to_agent', 'message': 'Agent selected', 'data': {'agent_used': current_state.get('agent_used')}}
            
            # Fetch knowledge
            current_state = await self._fetch_knowledge(current_state)
            yield {'status': 'progress', 'step': 'fetch_knowledge', 'message': 'Knowledge retrieved', 'data': {'vector_results_count': len(current_state.get('vector_results', []))}}
            
            # Process with agent - STREAM THE RESPONSE
            yield {'status': 'progress', 'step': 'process_with_agent', 'message': 'Generating AI response...', 'data': {'agent': current_state.get('agent_used')}}
            
            if current_state["agent_used"] == "banking":
                # Stream banking agent response in real-time
                response_chunks = []
                
                # Extract knowledge from vector results (convert dicts to strings)
                knowledge = []
                for result in current_state["vector_results"]:
                    if isinstance(result, dict):
                        if "content" in result:
                            knowledge.append(result["content"])
                        elif "text" in result:
                            knowledge.append(result["text"])
                        elif "snippet" in result:
                            knowledge.append(result["snippet"])
                
                # Determine banking domain from the query
                query_lower = current_state["normalized_query"].lower()
                if any(word in query_lower for word in ["loan", "borrow", "lending", "debt"]):
                    domain = "loans"
                elif any(word in query_lower for word in ["credit card", "credit limit", "credit score"]):
                    domain = "credit_cards"
                elif any(word in query_lower for word in ["mortgage", "home loan", "house loan", "refinance"]):
                    domain = "mortgages"
                elif any(word in query_lower for word in ["savings", "checking", "account", "deposit"]):
                    domain = "savings"
                elif any(word in query_lower for word in ["bank", "branch", "atm", "online banking"]):
                    domain = "banking_services"
                else:
                    domain = "general_banking"
                
                async for chunk in self.banking_agent.generate_response_streaming(
                    current_state["normalized_query"], 
                    knowledge, 
                    domain
                ):
                    response_chunks.append(chunk)
                    # Stream each chunk as it's generated
                    yield {'status': 'response_chunk', 'step': 'ai_generation', 'message': 'AI response chunk generated', 'data': {'chunk': chunk, 'agent': 'banking'}}
                
                # Combine chunks into final response
                current_state["agent_response"] = " ".join(response_chunks)
                
            else:
                # Stream finance agent response in real-time
                response_chunks = []
                
                # Extract knowledge from vector results (convert dicts to strings)
                knowledge = []
                for result in current_state["vector_results"]:
                    if isinstance(result, dict):
                        if "content" in result:
                            knowledge.append(result["content"])
                        elif "text" in result:
                            knowledge.append(result["text"])
                        elif "snippet" in result:
                            knowledge.append(result["snippet"])
                
                async for chunk in self.finance_agent.generate_response_streaming(
                    current_state["normalized_query"], 
                    knowledge, 
                    current_state["agent_used"]
                ):
                    response_chunks.append(chunk)
                    # Stream each chunk as it's generated
                    yield {'status': 'response_chunk', 'step': 'ai_generation', 'message': 'AI response chunk generated', 'data': {'chunk': chunk, 'agent': 'finance'}}
                
                # Combine chunks into final response
                current_state["agent_response"] = " ".join(response_chunks)
            
            yield {'status': 'progress', 'step': 'process_with_agent', 'message': 'AI response completed', 'data': {'agent_response_length': len(current_state.get('agent_response', ''))}}
            
            # Handle deep research if requested
            if current_state.get("deep_research", False):
                yield {'status': 'progress', 'step': 'deep_research_node', 'message': 'Starting deep research...', 'data': {'research_type': 'comprehensive'}}
                
                # Import and initialize deep research agent
                from app.agents.deep_research_agent import DeepResearchAgent
                deep_agent = DeepResearchAgent()
                deep_agent._generate_pdf_requested = current_state.get("generate_pdf", False)
                
                # Stream deep research in real-time
                research_results = {}
                async for chunk in deep_agent.perform_deep_research_streaming(
                    current_state["query"],
                    "comprehensive",
                    current_state.get("generate_pdf", False)
                ):
                    # Stream each research step as it happens
                    if chunk.get('status') == 'progress':
                        yield {'status': 'research_progress', 'step': chunk.get('step'), 'message': chunk.get('message'), 'data': chunk.get('data', {})}
                    elif chunk.get('status') == 'results':
                        research_results = chunk.get('data', {})
                        # Stream the final research results
                        yield {'status': 'research_results', 'step': 'deep_research_complete', 'message': 'Deep research completed', 'data': research_results}
                        break
                
                # Store research results in state
                current_state["deep_research_results"] = research_results
                current_state["research_type"] = "deep"
                
                if current_state.get("generate_pdf", False):
                    yield {'status': 'progress', 'step': 'generate_report', 'message': 'Generating PDF report...', 'data': {'generate_pdf': True}}
                    current_state = await self._generate_report(current_state)
                    yield {'status': 'progress', 'step': 'generate_report', 'message': 'PDF report generated', 'data': {'report_path': current_state.get('report_path')}}
            
            # Save to cache
            current_state = await self._save_to_cache(current_state)
            yield {'status': 'progress', 'step': 'save_to_cache', 'message': 'Response saved to cache', 'data': {'cached': True}}
            
            # Finalize response
            current_state = await self._finalize_response(current_state)
            yield {'status': 'progress', 'step': 'finalize_response', 'message': 'Response finalized', 'data': {'final_answer_length': len(current_state.get('final_answer', ''))}}
            
            # Send final results
            if current_state.get("needs_clarification", False):
                yield {
                    'status': 'clarification_needed',
                    'message': 'Query needs clarification',
                    'data': {
                        'answer': current_state.get("final_answer", "Please provide more details for your query."),
                        'cached': False,
                        'needs_clarification': True,
                        'clarification_questions': current_state.get("final_answer", ""),
                        'agent_used': "clarifier"
                    }
                }
            else:
                yield {
                    'status': 'results',
                    'message': 'Query completed successfully',
                    'data': {
                        'answer': current_state.get("final_answer", "No response generated"),
                        'cached': current_state.get("cached", False),
                        'agent_used': current_state.get("agent_used", ""),
                        'vector_results_count': len(current_state.get("vector_results", [])),
                        'needs_clarification': False
                    }
                }
            
            # Update stats for successful research
            self.execution_stats["total_queries"] += 1
            if current_state.get("cached", False):
                self.execution_stats["cached_responses"] += 1
            else:
                self.execution_stats["agent_responses"] += 1
            
        except Exception as e:
            logger.error(f"Error in streaming execution: {e}")
            raise
    
    async def execute_query(self, query: str, user_id: str, deep_research: bool = False, generate_pdf: bool = False) -> Dict[str, Any]:
        """Execute the complete workflow for a query (non-streaming fallback)"""
        start_time = time.time()
        
        try:
            # Initialize state
            initial_state = {
                "query": query,
                "normalized_query": "",
                "user_id": user_id,
                "deep_research": deep_research,
                "generate_pdf": generate_pdf,
                "cached_response": None,
                "agent_response": None,
                "final_answer": "",
                "cached": False,
                "processing_time": 0.0,
                "agent_used": "",
                "vector_results": [],
                "errors": [],
                "needs_clarification": False,
                "research_type": "deep" if deep_research else "standard",
                "deep_research_results": None,
                "report_path": None,
                "report_filename": None
            }
            
            # Execute the graph
            config = {"configurable": {"thread_id": f"{user_id}_{int(start_time)}"}}
            result = await self.graph.ainvoke(initial_state, config)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Check if clarification was needed
            if result.get("needs_clarification", False):
                return {
                    "answer": result.get("final_answer", "Please provide more details for your query."),
                    "cached": False,
                    "processing_time": processing_time,
                    "needs_clarification": True,
                    "clarification_questions": result.get("final_answer", ""),
                    "agent_used": "clarifier"
                }
            
            # Update stats for successful research
            self.execution_stats["total_queries"] += 1
            if result.get("cached", False):
                self.execution_stats["cached_responses"] += 1
            else:
                self.execution_stats["agent_responses"] += 1
            
            return {
                "answer": result.get("final_answer", "No response generated"),
                "cached": result.get("cached", False),
                "processing_time": processing_time,
                "agent_used": result.get("agent_used", ""),
                "vector_results_count": len(result.get("vector_results", [])),
                "needs_clarification": False
            }
            
        except Exception as e:
            self.execution_stats["errors"] += 1
            logger.error(f"Error in graph execution: {e}")
            raise
    
    async def _clarify_query(self, state: QueryState) -> QueryState:
        """Clarify if the query needs further clarification before research."""
        try:
            query = state["query"].lower()
            
            # Check if query is clear enough for research
            clarity_score = self._assess_query_clarity(query)
            
            if clarity_score < 0.5:  # Threshold for clarity
                # Generate clarifying questions based on domain
                clarifying_questions = self._generate_clarifying_questions(state["query"])
                
                state["final_answer"] = f"""
                I need more details to provide you with accurate research. Please clarify:

                {clarifying_questions}

                Please provide more specific information so I can conduct thorough research for you.
                """.strip()
                
                state["needs_clarification"] = True
                logger.info(f"Query needs clarification: {state['query'][:50]}...")
                
            else:
                # Query is clear, proceed with research
                state["needs_clarification"] = False
                logger.info(f"Query is clear, proceeding with research: {state['query'][:50]}...")
            
        except Exception as e:
            state["errors"].append(f"Query clarification failed: {e}")
            logger.error(f"Query clarification error: {e}")
            state["needs_clarification"] = False
        
        return state
    
    def _assess_query_clarity(self, query: str) -> float:
        """Assess how clear and specific a query is."""
        clarity_score = 0.0
        
        # Check for specific financial terms
        financial_terms = [
            "stock", "earnings", "revenue", "profit", "market cap", "pe ratio",
            "dividend", "portfolio", "investment", "retirement", "loan", "mortgage",
            "credit", "banking", "financial", "analysis", "report", "performance",
            "interest", "compound", "savings", "budget", "debt", "tax"
        ]
        
        # Check for specific companies/entities
        if any(word in query for word in ["tesla", "apple", "microsoft", "google", "amazon"]):
            clarity_score += 0.3
        
        # Check for specific financial metrics
        if any(term in query for term in financial_terms):
            clarity_score += 0.3  # Increased from 0.2
        
        # Check for time specificity
        if any(word in query for word in ["2024", "2025", "quarter", "annual", "monthly"]):
            clarity_score += 0.2
        
        # Check for action specificity
        if any(word in query for word in ["analyze", "compare", "evaluate", "assess", "research"]):
            clarity_score += 0.2
        
        # Check for scope specificity
        if any(word in query for word in ["detailed", "summary", "overview", "deep dive"]):
            clarity_score += 0.1
        
        # Bonus for basic financial education questions
        basic_questions = ["what is", "how does", "explain", "define", "tell me about"]
        if any(phrase in query.lower() for phrase in basic_questions):
            clarity_score += 0.2
        
        # Penalize vague queries
        vague_terms = ["good", "bad", "best", "worst", "help", "advice"]
        if any(term in query for term in vague_terms):
            clarity_score -= 0.1
        
        return min(max(clarity_score, 0.0), 1.0)
    
    def _generate_clarifying_questions(self, query: str) -> str:
        """Generate domain-specific clarifying questions."""
        query_lower = query.lower()
        
        # Determine domain
        if any(word in query_lower for word in ["stock", "earnings", "company", "tesla", "apple"]):
            domain = "company_analysis"
        elif any(word in query_lower for word in ["investment", "portfolio", "retirement"]):
            domain = "investment_planning"
        elif any(word in query_lower for word in ["loan", "mortgage", "credit", "bank"]):
            domain = "banking_services"
        elif any(word in query_lower for word in ["market", "trend", "economy"]):
            domain = "market_analysis"
        else:
            domain = "general_finance"
        
        # Generate domain-specific questions
        if domain == "company_analysis":
            questions = [
                "• Which specific company or stock are you interested in?",
                "• What time period do you want to analyze? (e.g., Q1 2024, last 5 years)",
                "• What specific aspects interest you? (earnings, stock performance, financial health, etc.)",
                "• Do you want a quick overview or detailed analysis?",
                "• Are you looking for investment advice or just information?"
            ]
        elif domain == "investment_planning":
            questions = [
                "• What's your investment timeline? (short-term, long-term, retirement)",
                "• What's your risk tolerance? (conservative, moderate, aggressive)",
                "• How much capital are you planning to invest?",
                "• What specific investment types interest you? (stocks, bonds, ETFs, real estate)",
                "• Do you have any specific financial goals?"
            ]
        elif domain == "banking_services":
            questions = [
                "• What specific banking service do you need? (loan, mortgage, credit card, account)",
                "• What's your current financial situation? (income, credit score, existing debt)",
                "• What's your timeline for this financial decision?",
                "• Are you comparing options or need specific advice?",
                "• What's your primary goal? (lower rates, better terms, building credit)"
            ]
        elif domain == "market_analysis":
            questions = [
                "• Which market or sector are you interested in? (US stocks, crypto, real estate, etc.)",
                "• What time horizon are you analyzing? (daily, weekly, monthly, yearly)",
                "• What specific market indicators interest you? (trends, volatility, correlations)",
                "• Are you looking for current analysis or historical patterns?",
                "• Do you need actionable insights or just market information?"
            ]
        else:
            questions = [
                "• What specific financial topic do you want to research?",
                "• What's your timeframe for this research?",
                "• What level of detail do you need? (summary, overview, deep analysis)",
                "• Are you looking for information, analysis, or recommendations?",
                "• What will you use this research for?"
            ]
        
        return "\n".join(questions)
    
    async def _normalize_query(self, state: QueryState) -> QueryState:
        """Normalize and clean the input query"""
        try:
            # Basic normalization (in production, use more sophisticated NLP)
            normalized = state["query"].lower().strip()
            normalized = " ".join(normalized.split())  # Remove extra whitespace
            
            state["normalized_query"] = normalized
            logger.info(f"Normalized query: {normalized}")
            
        except Exception as e:
            state["errors"].append(f"Query normalization failed: {e}")
            logger.error(f"Query normalization error: {e}")
        
        return state
    
    async def _check_cache(self, state: QueryState) -> QueryState:
        """Check if response exists in cache"""
        try:
            cache_key = f"query:{hash(state['normalized_query'])}"
            cached_response = await self.cache_service.get(cache_key)
            
            if cached_response:
                state["cached_response"] = cached_response
                state["cached"] = True
                logger.info(f"Cache hit for query: {state['normalized_query'][:50]}...")
            else:
                logger.info(f"Cache miss for query: {state['normalized_query'][:50]}...")
                
        except Exception as e:
            state["errors"].append(f"Cache check failed: {e}")
            logger.error(f"Cache check error: {e}")
        
        return state
    
    def _should_use_cache(self, state: QueryState) -> str:
        """Determine if we should use cached response"""
        return "cached" if state.get("cached_response") else "not_cached"
    
    def _should_clarify(self, state: QueryState) -> str:
        """Determine if the query needs clarification."""
        return "needs_clarification" if state.get("needs_clarification") else "clear"
    
    def _should_do_deep_research(self, state: QueryState) -> str:
        """Determine if deep research should be performed based on user preference."""
        # Use the user's explicit choice
        if state.get("deep_research", False):
            return "deep_research"
        else:
            return "standard"
    
    async def _route_to_agent(self, state: QueryState) -> QueryState:
        """Route query to appropriate agent based on content"""
        try:
            # Simple routing logic (in production, use ML classification)
            query_lower = state["normalized_query"].lower()
            
            if any(word in query_lower for word in ["bank", "banking", "loan", "credit", "mortgage"]):
                state["agent_used"] = "banking"
            else:
                state["agent_used"] = "finance"
            
            logger.info(f"Routed to {state['agent_used']} agent")
            
            # Push to Kafka for async processing (optional)
            await self.queue_service.publish_message(
                "financial_queries",
                {
                    "query": state["normalized_query"],
                    "user_id": state["user_id"],
                    "agent": state["agent_used"]
                }
            )
            
        except Exception as e:
            state["errors"].append(f"Agent routing failed: {e}")
            logger.error(f"Agent routing error: {e}")
        
        return state
    
    async def _fetch_knowledge(self, state: QueryState) -> QueryState:
        """Fetch relevant knowledge from vector database"""
        try:
            # Get relevant documents from vector DB
            vector_results = await self.vector_db.search(
                state["normalized_query"],
                top_k=5
            )
            
            state["vector_results"] = vector_results
            logger.info(f"Retrieved {len(vector_results)} vector results")
            
        except Exception as e:
            state["errors"].append(f"Vector search failed: {e}")
            logger.error(f"Vector search error: {e}")
        
        return state
    
    async def _process_with_agent(self, state: QueryState) -> QueryState:
        """Process query with the selected agent using streaming"""
        try:
            if state["agent_used"] == "banking":
                # Use streaming response from banking agent
                response_chunks = []
                
                # Extract knowledge from vector results (convert dicts to strings)
                knowledge = []
                for result in state["vector_results"]:
                    if isinstance(result, dict):
                        if "content" in result:
                            knowledge.append(result["content"])
                        elif "text" in result:
                            knowledge.append(result["text"])
                        elif "snippet" in result:
                            knowledge.append(result["snippet"])
                
                # Determine banking domain from the query
                query_lower = state["normalized_query"].lower()
                if any(word in query_lower for word in ["loan", "borrow", "lending", "debt"]):
                    domain = "loans"
                elif any(word in query_lower for word in ["credit card", "credit limit", "credit score"]):
                    domain = "credit_cards"
                elif any(word in query_lower for word in ["mortgage", "home loan", "house loan", "refinance"]):
                    domain = "mortgages"
                elif any(word in query_lower for word in ["savings", "checking", "account", "deposit"]):
                    domain = "savings"
                elif any(word in query_lower for word in ["bank", "branch", "atm", "online banking"]):
                    domain = "banking_services"
                else:
                    domain = "general_banking"
                
                async for chunk in self.banking_agent.generate_response_streaming(
                    state["normalized_query"], 
                    knowledge, 
                    domain
                ):
                    response_chunks.append(chunk)
                
                # Combine chunks into final response
                state["agent_response"] = " ".join(response_chunks)
                
            else:
                # Use streaming response from finance agent
                response_chunks = []
                
                # Extract knowledge from vector results (convert dicts to strings)
                knowledge = []
                for result in state["vector_results"]:
                    if isinstance(result, dict):
                        if "content" in result:
                            knowledge.append(result["content"])
                        elif "text" in result:
                            knowledge.append(result["text"])
                        elif "snippet" in result:
                            knowledge.append(result["snippet"])
                
                async for chunk in self.finance_agent.generate_response_streaming(
                    state["normalized_query"], 
                    knowledge, 
                    state["agent_used"]
                ):
                    response_chunks.append(chunk)
                
                # Combine chunks into final response
                state["agent_response"] = " ".join(response_chunks)
            
            logger.info(f"Agent {state['agent_used']} processed query successfully")
            
        except Exception as e:
            state["errors"].append(f"Agent processing failed: {e}")
            logger.error(f"Agent processing error: {e}")
        
        return state
    
    async def _deep_research(self, state: QueryState) -> QueryState:
        """Perform comprehensive research using the DeepResearchAgent with streaming."""
        try:
            logger.info("Starting deep research...")

            # Import and use the DeepResearchAgent
            from app.agents.deep_research_agent import DeepResearchAgent
            deep_agent = DeepResearchAgent()
            
            # Set the PDF generation flag based on user preference
            deep_agent._generate_pdf_requested = state.get("generate_pdf", False)

            # Use streaming deep research
            research_results = {}
            async for chunk in deep_agent.perform_deep_research_streaming(
                state["query"],
                "comprehensive",
                state.get("generate_pdf", False)
            ):
                if chunk.get('status') == 'results':
                    research_results = chunk.get('data', {})
                    break

            # Store research results in state
            state["deep_research_results"] = research_results
            state["research_type"] = "deep"

            logger.info("Deep research completed successfully")

        except Exception as e:
            state["errors"].append(f"Deep research failed: {e}")
            logger.error(f"Deep research error: {e}")
        
        return state
    
    async def _generate_report(self, state: QueryState) -> QueryState:
        """Generate a comprehensive report based on deep research."""
        try:
            # Only generate report if user requested it AND we're doing deep research
            if (state.get("deep_research", False) and 
                state.get("generate_pdf", False) and 
                state.get("deep_research_results")):
                
                logger.info("Generating comprehensive report...")
                
                # Import and use the DeepResearchAgent for report generation
                from app.agents.deep_research_agent import DeepResearchAgent
                deep_agent = DeepResearchAgent()
                
                # Generate PDF report
                pdf_path = await deep_agent.generate_pdf_report(
                    state["deep_research_results"],
                    "comprehensive"
                )
                
                if pdf_path:
                    # Store report path in state
                    state["report_path"] = pdf_path
                    state["report_filename"] = os.path.basename(pdf_path)
                    logger.info(f"Report generated: {pdf_path}")
                else:
                    logger.warning("Failed to generate report")
            else:
                logger.info("Skipping report generation - not requested or not deep research")
            
        except Exception as e:
            state["errors"].append(f"Report generation failed: {e}")
            logger.error(f"Report generation error: {e}")
        
        return state
    
    async def _save_to_cache(self, state: QueryState) -> QueryState:
        """Save response to cache for future use"""
        try:
            if state.get("agent_response"):
                cache_key = f"query:{state['user_id']}:{hash(state['normalized_query'])}"
                await self.cache_service.set(
                    cache_key,
                    state["agent_response"],
                    expire=3600  # 1 hour
                )
                logger.info("Response saved to cache")
                
        except Exception as e:
            state["errors"].append(f"Cache save failed: {e}")
            logger.error(f"Cache save error: {e}")
        
        return state
    
    async def _finalize_response(self, state: QueryState) -> QueryState:
        """Finalize the response for the user"""
        try:
            if state.get("cached_response"):
                state["final_answer"] = state["cached_response"]
            elif state.get("agent_response"):
                if state.get("research_type") == "deep" and state.get("report_filename"):
                    # Deep research with report
                    report_url = f"/static/{state['report_filename']}"
                    state["final_answer"] = f"""
{state['agent_response']}

📊 **Deep Research Report Generated**
Your comprehensive research report is ready for download:
🔗 [Download Report]({report_url})

The report contains detailed analysis, sources, recommendations, and risk assessment.
                    """.strip()
                else:
                    # Standard response
                    state["final_answer"] = state["agent_response"]
            else:
                state["final_answer"] = "I apologize, but I encountered an error processing your query. Please try again."
            
            logger.info("Response finalized successfully")
            
        except Exception as e:
            state["errors"].append(f"Response finalization failed: {e}")
            state["final_answer"] = "An error occurred while processing your request."
            logger.error(f"Response finalization error: {e}")
        
        return state
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return self.execution_stats.copy()
