# graph.py
# Performance Note: This graph can take 1-3 minutes due to:
# 1. LLM calls (Ollama) - each takes 10-30 seconds
# 2. Web search (DuckDuckGo) - takes 5-15 seconds  
# 3. Sequential execution - operations run one after another
# 
# To improve performance, consider:
# - Using a faster LLM model (OpenAI, Anthropic, HuggingFace)
# - Implementing parallel processing where possible
# - Adding response caching
# - Limiting web search results

import asyncio
from typing import Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from app.agents import BankingAgent, FinanceAgent, DeepResearchAgent

# Define the state schema as a TypedDict
class GraphState(TypedDict):
    query: str
    intent: Optional[str]
    response: Optional[str]
    web_data: Optional[str]
    deep_research: bool
    generate_report: bool
    pdf_report_path: Optional[str]
    report_generated: bool

class FinancialResearchGraph:
    def __init__(self):
        # Initialize Ollama client
        self.llm_client = Ollama(model="mistral")
        
        # Initialize specialized agents
        self.banking_agent = BankingAgent(self.llm_client)
        self.finance_agent = FinanceAgent(self.llm_client)
        self.deep_research_agent = DeepResearchAgent(self.llm_client)
        
        print(f"🚀 [GRAPH] Initialized agents:")
        print(f"   🏦 Banking Agent: {self.banking_agent.get_agent_info()}")
        print(f"   📈 Finance Agent: {self.finance_agent.get_agent_info()}")
        print(f"   🔬 Deep Research Agent: {self.deep_research_agent.get_agent_info()}")
        
        # Initialize StateGraph with proper schema
        self.graph = StateGraph(GraphState)

        # Add nodes
        self.graph.add_node("query_check", self.check_query_sufficiency)
        self.graph.add_node("intent_detection", self.detect_intent)
        self.graph.add_node("intent_branch", self.intent_branch)
        self.graph.add_node("deep_research_node", self.deep_research)
        self.graph.add_node("banking", self.banking)
        self.graph.add_node("finance", self.finance)
        self.graph.add_node("end", self.end)

        # Add edges
        self.graph.add_edge("query_check", "intent_detection")
        self.graph.add_edge("intent_detection", "intent_branch")

        # Conditional routing based on user deep_research or detected intent
        self.graph.add_conditional_edges(
            "intent_branch",
            self.intent_branch_routing,
            {
                "deep_research": "deep_research_node",
                "banking": "banking",
                "finance": "finance",
                "end": "end"
            }
        )

        # End nodes
        self.graph.add_edge("deep_research_node", END)
        self.graph.add_edge("banking", END)
        self.graph.add_edge("finance", END)
        self.graph.add_edge("end", END)

        # Set entry point
        self.graph.set_entry_point("query_check")

        # Compile the graph
        self.compiled_graph = self.graph.compile()

    # ------------------ Node functions ------------------
    async def check_query_sufficiency(self, state: GraphState) -> GraphState:
        print(f"🔍 [query_check] Checking query: '{state['query']}'")
        if len(state["query"].split()) < 3:
            state["response"] = "Query insufficient. Please add more details."
            print(f"❌ [query_check] Query insufficient - words: {len(state['query'].split())}")
        else:
            state["response"] = "SUFFICIENT"
            print(f"✅ [query_check] Query sufficient - words: {len(state['query'].split())}")
        return state

    async def detect_intent(self, state: GraphState) -> GraphState:
        print(f"🧠 [intent_detection] Starting intent detection...")
        # Only detect intent if not deep research
        if not state.get("deep_research"):
            prompt = f"Determine intent: {state['query']}. Possible: banking, finance."
            print(f"🤖 [intent_detection] Calling LLM with prompt: {prompt[:100]}...")
            start_time = asyncio.get_event_loop().time()
            response = self.llm_client(prompt)
            llm_time = asyncio.get_event_loop().time() - start_time
            state["intent"] = response.strip().lower()
            print(f"🎯 [intent_detection] LLM response: '{state['intent']}' (took {llm_time:.2f}s)")
        else:
            print(f"⏭️ [intent_detection] Skipping intent detection (deep research mode)")
        return state

    async def intent_branch(self, state: GraphState) -> GraphState:
        print(f"🔄 [intent_branch] Routing decision...")
        return state

    def intent_branch_routing(self, state: GraphState):
        # Route to deep research if user requested it
        if state.get("deep_research"):
            print(f"🔬 [routing] Routing to deep_research_node")
            return "deep_research"

        # Otherwise use detected intent
        intent = state.get("intent", "")
        if intent in ["banking", "finance"]:
            print(f"🎯 [routing] Routing to {intent}")
            return intent

        # Fallback
        print(f"⚠️ [routing] No clear intent, routing to end")
        return "end"

    async def deep_research(self, state: GraphState) -> GraphState:
        """Delegate to deep research agent"""
        # Check if report generation is requested
        generate_report = state.get("generate_report", False)
        
        result = await self.deep_research_agent.process(
            state['query'], 
            generate_report=generate_report
        )
        
        state["response"] = result["response"]
        state["web_data"] = result.get("web_data", "")
        state["pdf_report_path"] = result.get("pdf_report_path", "")
        state["report_generated"] = result.get("report_generated", False)
        
        return state

    async def banking(self, state: GraphState) -> GraphState:
        """Delegate to banking agent"""
        result = await self.banking_agent.process(state['query'])
        state["response"] = result["response"]
        return state

    async def finance(self, state: GraphState) -> GraphState:
        """Delegate to finance agent"""
        result = await self.finance_agent.process(state['query'])
        state["response"] = result["response"]
        return state

    async def end(self, state: GraphState) -> GraphState:
        print(f"🏁 [end] Finalizing response...")
        if not state.get("response"):
            state["response"] = "END"
        print(f"✅ [end] Final response: {state['response'][:100]}...")
        return state
    
    # ------------------ Streaming methods ------------------
    async def banking_stream(self, state: GraphState) -> GraphState:
        """Delegate to banking agent with streaming"""
        async for result in self.banking_agent.process_stream(state['query']):
            # Update state with streaming results
            if result.get("type") == "final_result":
                state["response"] = result["response"]
        return state

    async def finance_stream(self, state: GraphState) -> GraphState:
        """Delegate to finance agent with streaming"""
        async for result in self.finance_agent.process_stream(state['query']):
            # Update state with streaming results
            if result.get("type") == "final_result":
                state["response"] = result["response"]
        return state

    async def deep_research_stream(self, state: GraphState) -> GraphState:
        """Delegate to deep research agent with streaming"""
        # Check if report generation is requested
        generate_report = state.get("generate_report", False)
        
        async for result in self.deep_research_agent.process_stream(
            state['query'], 
            generate_report=generate_report
        ):
            # Update state with streaming results
            if result.get("type") == "final_result":
                state["response"] = result["response"]
                state["web_data"] = result.get("web_data", "")
                state["pdf_report_path"] = result.get("pdf_report_path", "")
                state["report_generated"] = result.get("report_generated", False)
        
        return state

    # ------------------ Graph execution method ------------------
    async def process(self, query: str, session_id: str = None, state: dict = None) -> GraphState:
        """Process a query through the graph"""
        print(f"\n🚀 [PROCESS] Starting graph execution for query: '{query[:100]}...'")
        print(f"🔧 [PROCESS] Deep research mode: {state.get('deep_research', False) if state else False}")
        
        start_time = asyncio.get_event_loop().time()
        
        # Initialize the state
        initial_state = GraphState(
            query=query,
            intent=None,
            response=None,
            web_data=None,
            deep_research=state.get("deep_research", False) if state else False,
            generate_report=state.get("generate_report", False) if state else False,
            pdf_report_path=None,
            report_generated=False
        )
        
        print(f"📋 [PROCESS] Initial state created, starting graph execution...")
        
        # Run the compiled graph
        result = await self.compiled_graph.ainvoke(initial_state)
        
        total_time = asyncio.get_event_loop().time() - start_time
        print(f"⏱️ [PROCESS] Total execution time: {total_time:.2f}s")
        print(f"🎯 [PROCESS] Final intent: {result.get('intent', 'N/A')}")
        print(f"💬 [PROCESS] Response length: {len(result.get('response', ''))} characters")
        
        return result
    
    async def process_stream(self, query: str, session_id: str = None, state: dict = None) -> GraphState:
        """Process a query through the graph with streaming updates"""
        print(f"\n🚀 [PROCESS_STREAM] Starting streaming graph execution for query: '{query[:100]}...'")
        
        # Initialize the state
        initial_state = GraphState(
            query=query,
            intent=None,
            response=None,
            web_data=None,
            deep_research=state.get("deep_research", False) if state else False,
            generate_report=state.get("generate_report", False) if state else False,
            pdf_report_path=None,
            report_generated=False
        )
        
        # For streaming, we need to manually route and call streaming agents
        # First, check query sufficiency
        if len(query.split()) < 3:
            initial_state["response"] = "Query insufficient. Please add more details."
            return initial_state
        
        # Detect intent if not deep research
        if not initial_state.get("deep_research"):
            # Simple intent detection for streaming
            if any(word in query.lower() for word in ["bank", "account", "loan", "credit", "mortgage"]):
                initial_state["intent"] = "banking"
            elif any(word in query.lower() for word in ["invest", "stock", "portfolio", "retirement", "tax"]):
                initial_state["intent"] = "finance"
            else:
                initial_state["intent"] = "finance"  # Default to finance
        
        # Route to appropriate streaming agent
        if initial_state.get("deep_research"):
            await self.deep_research_stream(initial_state)
        elif initial_state.get("intent") == "banking":
            await self.banking_stream(initial_state)
        else:
            await self.finance_stream(initial_state)
        
        return initial_state
