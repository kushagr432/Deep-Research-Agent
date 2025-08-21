#!/usr/bin/env python3
"""
LangGraph Visualization Script
This script visualizes your FinancialResearchGraph structure
"""

import matplotlib.pyplot as plt
import networkx as nx
from app.graph import FinancialResearchGraph

def visualize_langgraph():
    """Visualize the LangGraph structure"""
    
    # Create a new graph instance to access the graph structure
    graph_instance = FinancialResearchGraph()
    
    # Create a NetworkX graph for visualization
    G = nx.DiGraph()
    
    # Define nodes and their positions
    nodes = [
        "query_check",
        "intent_detection", 
        "intent_branch",
        "deep_research_node",
        "banking",
        "finance",
        "end"
    ]
    
    # Add nodes
    for node in nodes:
        G.add_node(node)
    
    # Define edges based on your graph structure
    edges = [
        ("query_check", "intent_detection"),
        ("intent_detection", "intent_branch"),
        ("intent_branch", "deep_research_node"),
        ("intent_branch", "banking"),
        ("intent_branch", "finance"),
        ("intent_branch", "end"),
        ("deep_research_node", "end"),
        ("banking", "end"),
        ("finance", "end")
    ]
    
    # Add edges
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    
    # Use hierarchical layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=3000,
                          alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          arrowstyle='->',
                          width=2)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, 
                           font_size=10,
                           font_weight='bold')
    
    # Add title
    plt.title("Financial Research LangGraph Structure", 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add description
    plt.figtext(0.5, 0.02, 
                "Flow: query_check → intent_detection → intent_branch → [deep_research/banking/finance] → end",
                ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.axis('off')
    
    # Save the visualization
    plt.savefig('langgraph_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ Graph visualization saved as 'langgraph_visualization.png'")
    
    # Show the plot
    plt.show()

def print_graph_info():
    """Print detailed information about the graph structure"""
    print("\n" + "="*60)
    print("📊 LANGGRAPH STRUCTURE INFORMATION")
    print("="*60)
    
    graph_instance = FinancialResearchGraph()
    
    print("\n🏗️  Graph Nodes:")
    for node in graph_instance.graph.nodes:
        print(f"   • {node}")
    
    print("\n🔗 Graph Edges:")
    for edge in graph_instance.graph.edges:
        print(f"   • {edge[0]} → {edge[1]}")
    
    print("\n🎯 Entry Point:")
    print(f"   • {graph_instance.graph.entry_point}")
    
    print("\n🔄 Conditional Routing:")
    print("   • intent_branch → deep_research (when deep_research=True)")
    print("   • intent_branch → banking (when intent=banking)")
    print("   • intent_branch → finance (when intent=finance)")
    print("   • intent_branch → end (fallback)")
    
    print("\n📝 State Schema:")
    state_fields = [
        "query", "intent", "response", "web_data",
        "deep_research", "generate_report", 
        "pdf_report_path", "report_generated"
    ]
    for field in state_fields:
        print(f"   • {field}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        print("🚀 Starting LangGraph visualization...")
        print_graph_info()
        visualize_langgraph()
        print("\n✅ Visualization complete!")
        
    except Exception as e:
        print(f"❌ Error during visualization: {str(e)}")
        print("\n💡 Make sure you have matplotlib and networkx installed:")
        print("   pip install matplotlib networkx")
