#!/usr/bin/env python3
"""
Graph Visualization Script for Financial Research AI
Uses LangGraph CLI to visualize and export the graph structure
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def run_langgraph_command(command):
    """Run a LangGraph CLI command and return the result"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        if result.returncode == 0:
            print(f"✅ {command}")
            return result.stdout
        else:
            print(f"❌ {command}")
            print(f"Error: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ {command} - Exception: {e}")
        return None

def visualize_graph():
    """Main function to visualize the graph"""
    print("🚀 LangGraph Visualization Tool")
    print("=" * 50)
    
    # Check if we're in the virtual environment
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️  Please activate your virtual environment first:")
        print("   myenv\\Scripts\\activate")
        return
    
    # Check if langgraph CLI is available
    print("🔍 Checking LangGraph CLI...")
    if not run_langgraph_command("langgraph --version"):
        print("❌ LangGraph CLI not found. Please install it first.")
        return
    
    print("\n📊 Graph Structure Information:")
    print("-" * 30)
    
    # Read the existing graph structure
    graph_file = Path("graph_exports/graph_structure.json")
    if graph_file.exists():
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)
        
        print(f"Nodes: {len(graph_data['nodes'])}")
        print(f"Edges: {len(graph_data['edges'])}")
        print(f"Entry Point: {graph_data['entry_point']}")
        print(f"Conditional Edges: {len(graph_data['conditional_edges'])}")
        
        print("\n📋 Nodes:")
        for node in graph_data['nodes']:
            print(f"  • {node}")
        
        print("\n🔗 Edges:")
        for edge in graph_data['edges']:
            print(f"  • {edge[0]} → {edge[1]}")
        
        if graph_data['conditional_edges']:
            print("\n🔄 Conditional Edges:")
            for node, conditions in graph_data['conditional_edges'].items():
                for condition, target in conditions.items():
                    print(f"  • {node} [{condition}] → {target}")
    
    print("\n🎯 Available LangGraph CLI Commands:")
    print("-" * 40)
    print("1. langgraph dev          - Run development server")
    print("2. langgraph up           - Launch production server")
    print("3. langgraph build        - Build Docker image")
    print("4. langgraph dockerfile   - Generate Dockerfile")
    
    print("\n💡 To start the development server:")
    print("   langgraph dev")
    
    print("\n💡 To view the graph in the browser:")
    print("   Open: http://localhost:8000")

def export_graph_visualization():
    """Export the graph in various formats"""
    print("\n📤 Exporting Graph Visualizations...")
    
    # Create graph_exports directory if it doesn't exist
    exports_dir = Path("graph_exports")
    exports_dir.mkdir(exist_ok=True)
    
    # Generate DOT format for Graphviz
    dot_content = """digraph FinancialResearchGraph {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    edge [color=gray50];
    
    // Nodes
    query_check [label="Query Check", fillcolor=lightgreen];
    intent_detection [label="Intent Detection", fillcolor=lightyellow];
    intent_branch [label="Intent Branch", fillcolor=lightcoral];
    deep_research_node [label="Deep Research", fillcolor=lightsteelblue];
    banking [label="Banking Agent", fillcolor=lightpink];
    finance [label="Finance Agent", fillcolor=lightcyan];
    end [label="End", fillcolor=lightgray];
    
    // Edges
    query_check -> intent_detection;
    intent_detection -> intent_branch;
    intent_branch -> deep_research_node [label="deep_research"];
    intent_branch -> banking [label="banking"];
    intent_branch -> finance [label="finance"];
    intent_branch -> end [label="end"];
    deep_research_node -> end;
    banking -> end;
    finance -> end;
}"""
    
    with open(exports_dir / "graph_structure.dot", "w") as f:
        f.write(dot_content)
    
    # Generate Mermaid format
    mermaid_content = """graph TB
    query_check[Query Check]
    intent_detection[Intent Detection]
    intent_branch[Intent Branch]
    deep_research_node[Deep Research]
    banking[Banking Agent]
    finance[Finance Agent]
    end[End]
    
    query_check --> intent_detection
    intent_detection --> intent_branch
    intent_branch -->|deep_research| deep_research_node
    intent_branch -->|banking| banking
    intent_branch -->|finance| finance
    intent_branch -->|end| end
    deep_research_node --> end
    banking --> end
    finance --> end"""
    
    with open(exports_dir / "graph_structure.mmd", "w") as f:
        f.write(mermaid_content)
    
    print("✅ Graph visualizations exported to graph_exports/")

if __name__ == "__main__":
    visualize_graph()
    export_graph_visualization()
    
    print("\n🎉 Graph visualization complete!")
    print("\n📁 Check the graph_exports/ directory for visualization files:")
    print("   • graph_structure.dot  - Graphviz format")
    print("   • graph_structure.mmd  - Mermaid format")
    print("   • graph_structure.json - JSON structure")
