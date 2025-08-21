# LangGraph Visualization Guide

## Generated Files

This script generates several visualization formats in the `graph_exports/` directory:

### 1. JSON Format (`graph_structure.json`)
- Raw graph data structure
- Can be used for programmatic analysis
- Easy to parse and manipulate

### 2. DOT Format (`graph_structure.dot`)
- Use with Graphviz for high-quality visualizations
- Install Graphviz: https://graphviz.org/download/
- Command: `dot -Tpng graph_structure.dot -o graph.png`

### 3. Mermaid Format (`graph_structure.mmd`)
- Use with Mermaid Live Editor: https://mermaid.live/
- Or with GitHub/GitLab markdown
- Supports interactive diagrams

### 4. PlantUML Format (`graph_structure.puml`)
- Use with PlantUML: http://www.plantuml.com/
- Great for documentation
- Supports multiple output formats

## Online Visualization Tools

### Mermaid Live Editor
1. Go to https://mermaid.live/
2. Paste the content of `graph_structure.mmd`
3. View and customize your diagram

### PlantUML Online
1. Go to http://www.plantuml.com/plantuml/
2. Paste the content of `graph_structure.puml`
3. Generate PNG, SVG, or other formats

### Graphviz Online
1. Go to https://dreampuf.github.io/GraphvizOnline/
2. Paste the content of `graph_structure.dot`
3. Download the generated image

## Customization

You can modify the visualization scripts to:
- Change colors and styles
- Add more detailed labels
- Include state information
- Show execution flow
