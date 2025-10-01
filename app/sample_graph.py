# app/sample_graph.py
# Simplified version of FinancialResearchGraph for visualization and testing
# This follows the same structure as your original graph but with minimal dependencies

import os
from typing import Optional, TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# Define the state schema as a TypedDict
class GraphState(TypedDict):
    query: str
    intent: Optional[str]
    response: Optional[str]
    web_data: Optional[str]
    yahoo_finance_data: Optional[dict]
    deep_research: bool
    generate_report: bool
    generate_dashboard: bool
    pdf_report_path: Optional[str]
    dashboard_path: Optional[str]
    report_generated: bool
    dashboard_generated: bool

class FinancialResearchGraph:
    def __init__(self):
        print(f"🚀 [GRAPH] Initializing FinancialResearchGraph...")
        
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
        self.graph.add_conditional_edges(
            "query_check",
            self.query_check_routing,
            {
                "continue": "intent_detection",
                "end": "end"
            }
        )
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
        
        print(f"✅ [GRAPH] Graph compiled successfully!")

    # ------------------ Node functions ------------------
    async def check_query_sufficiency(self, state: GraphState) -> GraphState:
        print(f"🔍 [query_check] Checking query: '{state['query']}'")
        if len(state["query"].split()) < 3:
            state["response"] = "Query insufficient. Please provide more specific details (at least 3 words)."
            print(f"❌ [query_check] Query insufficient - words: {len(state['query'].split())}")
        else:
            state["response"] = "SUFFICIENT"
            print(f"✅ [query_check] Query sufficient - words: {len(state['query'].split())}")
        return state

    async def detect_intent(self, state: GraphState) -> GraphState:
        print(f"🎯 [intent_detection] Starting intent detection...")
        if not state.get("deep_research"):
            # Simple keyword-based intent detection
            query_lower = state['query'].lower()
            if any(word in query_lower for word in ["bank", "account", "loan", "credit", "mortgage"]):
                state["intent"] = "banking"
            elif any(word in query_lower for word in ["invest", "stock", "portfolio", "retirement", "tax"]):
                state["intent"] = "finance"
            else:
                state["intent"] = "finance"  # Default to finance
            
            print(f"🎯 [intent_detection] Detected intent: {state['intent']}")
        else:
            print(f"⏭️ [intent_detection] Skipping intent detection (deep research mode)")
        return state

    async def intent_branch(self, state: GraphState) -> GraphState:
        print(f"🔄 [intent_branch] Routing decision...")
        return state

    def query_check_routing(self, state: GraphState):
        """Route after query sufficiency check"""
        if state.get("response") and "Query insufficient" in state["response"]:
            print(f"🚫 [routing] Query insufficient, routing to end")
            return "end"
        else:
            print(f"✅ [routing] Query sufficient, continuing to intent detection")
            return "continue"

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
        """Deep research functionality"""
        print(f"🔬 [deep_research] Processing deep research query...")
        state["response"] = f"Deep research completed for: {state['query']}"
        state["web_data"] = "Sample web research data"
        state["yahoo_finance_data"] = {"symbol": "AAPL", "price": 150.00}
        state["report_generated"] = state.get("generate_report", False)
        state["dashboard_generated"] = state.get("generate_dashboard", False)
        return state

    async def banking(self, state: GraphState) -> GraphState:
        """Banking functionality"""
        print(f"🏦 [banking] Processing banking query...")
        state["response"] = f"Banking advice for: {state['query']}"
        return state

    async def finance(self, state: GraphState) -> GraphState:
        """Finance functionality"""
        print(f"📈 [finance] Processing finance query...")
        state["response"] = f"Finance advice for: {state['query']}"
        return state

    async def end(self, state: GraphState) -> GraphState:
        print(f"🏁 [end] Finalizing response...")
        if not state.get("response"):
            state["response"] = "END"
        print(f"✅ [end] Final response: {state['response'][:100]}...")
        return state
    
    # ------------------ Graph execution method ------------------
    async def process(self, query: str, session_id: str = None, state: dict = None) -> GraphState:
        """Process a query through the graph"""
        print(f"\n🚀 [PROCESS] Starting graph execution for query: '{query[:100]}...'")
        print(f"🔧 [PROCESS] Deep research mode: {state.get('deep_research', False) if state else False}")
        
        # Initialize the state
        initial_state: GraphState = {
            "query": query,
            "intent": None,
            "response": None,
            "web_data": None,
            "yahoo_finance_data": None,
            "deep_research": state.get("deep_research", False) if state else False,
            "generate_report": state.get("generate_report", False) if state else False,
            "generate_dashboard": state.get("generate_dashboard", False) if state else False,
            "pdf_report_path": None,
            "dashboard_path": None,
            "report_generated": False,
            "dashboard_generated": False
        }
        
        print(f"📋 [PROCESS] Initial state created, starting graph execution...")
        
        # Run the compiled graph
        result = await self.compiled_graph.ainvoke(initial_state)
        
        print(f"🎯 [PROCESS] Final intent: {result.get('intent', 'N/A')}")
        print(f"💬 [PROCESS] Response length: {len(result.get('response', ''))} characters")
        
        return result

graph = FinancialResearchGraph().compiled_graph


# Test function to demonstrate the graph
async def test_graph():
    """Test the graph with sample queries"""
    print("🧪 Testing FinancialResearchGraph...")
    
    # Create graph instance
    graph = FinancialResearchGraph()
    
    # Test 1: Insufficient query
    print("\n" + "="*50)
    print("TEST 1: Insufficient query")
    print("="*50)
    result1 = await graph.process("hi")
    print(f"Result: {result1['response']}")
    
    # Test 2: Banking query
    print("\n" + "="*50)
    print("TEST 2: Banking query")
    print("="*50)
    result2 = await graph.process("How do I get a mortgage loan?")
    print(f"Intent: {result2['intent']}")
    print(f"Response: {result2['response']}")
    
    # Test 3: Finance query
    print("\n" + "="*50)
    print("TEST 3: Finance query")
    print("="*50)
    result3 = await graph.process("What should I invest in for retirement?")
    print(f"Intent: {result3['intent']}")
    print(f"Response: {result3['response']}")
    
    # Test 4: Deep research query
    print("\n" + "="*50)
    print("TEST 4: Deep research query")
    print("="*50)
    result4 = await graph.process("Analyze the current market trends for tech stocks", 
                                 state={"deep_research": True, "generate_report": True})
    print(f"Intent: {result4['intent']}")
    print(f"Response: {result4['response']}")
    print(f"Report generated: {result4['report_generated']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_graph())
