# Financial & Deep Research AI Chatbot

A FastAPI-based AI chatbot that uses LangGraph and LangChain to provide intelligent responses to financial and banking queries, with optional deep research capabilities using web search.

## Features

- **Intent Detection**: Automatically detects whether queries are related to banking or finance
- **Deep Research**: Option to perform web research using DuckDuckGo search
- **LangGraph Workflow**: Uses LangGraph for structured conversation flow
- **Ollama Integration**: Local LLM support using Ollama
- **FastAPI Backend**: Modern, fast API with automatic documentation

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- A Llama2 model downloaded (e.g., `llama2:7b`)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FastAPI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install and start Ollama:
```bash
# Download Ollama from https://ollama.ai/
# Then pull the Llama2 model:
ollama pull llama2:7b
```

## Usage

### Starting the Server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### POST /query
Submit a query to the AI chatbot.

**Request Body:**
```json
{
  "user_query": "How do I open a savings account?",
  "deep_research": false,
  "generate_report": false,
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "response": "AI generated response...",
  "intent": "banking",
  "deep_research": false,
  "generate_report": false,
  "report_generated": false,
  "pdf_report_path": null,
  "session_id": "optional-session-id"
}
```

#### POST /query/stream
**🚀 NEW: Streaming endpoint for real-time chat-like responses!**

Submit a query and receive live streaming updates as the AI processes your request.

**Request Body:** Same as `/query`

**Response:** Server-Sent Events (SSE) stream with real-time updates:

```javascript
// Frontend usage example
const eventSource = new EventSource('/query/stream');
eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'start':
            console.log('Starting analysis...');
            break;
        case 'status':
            console.log('Status:', data.message);
            break;
        case 'response_chunk':
            console.log('Response chunk:', data.content);
            // Append to chat interface
            break;
        case 'complete':
            console.log('Analysis complete!');
            eventSource.close();
            break;
    }
};
```

**Streaming Features:**
- **Real-time status updates** during processing
- **Live response chunks** for chat-like experience
- **Progress indicators** for each step
- **Automatic completion** when done

**Note:** `generate_report` only works when `deep_research` is `true`. PDF reports are only generated for comprehensive research queries.

#### GET /download-report/{filename}
Download a generated PDF report by filename.

#### GET /reports
List all available PDF reports with metadata.

#### GET /health
Health check endpoint.

### Testing

Run the test script to verify the graph functionality:

```bash
python test_graph.py
```

## Architecture

The application uses a **modular agent-based architecture** with LangGraph workflow:

### 🤖 **Agent System:**
- **BaseAgent**: Common functionality and logging for all agents
- **BankingAgent**: Specialized in banking queries (accounts, loans, credit, etc.)
- **FinanceAgent**: Specialized in investment and financial planning
- **DeepResearchAgent**: Performs web research and comprehensive analysis

### 🔄 **Graph Workflow:**
1. **query_check**: Validates query sufficiency
2. **intent_detection**: Detects user intent (banking/finance)
3. **intent_branch**: Routes to appropriate agent
4. **Agent Processing**: Delegates to specialized agent for processing
5. **end**: Final response generation

### 🏗️ **Benefits of Agent Architecture:**
- **Modular**: Each agent handles specific domain expertise
- **Maintainable**: Easy to add new agents or modify existing ones
- **Testable**: Individual agents can be tested independently
- **Scalable**: New capabilities can be added as new agents

## Configuration

- **Ollama Model**: Change the model in `app/graph.py` line 25
- **Search Results**: Modify the `max_results` parameter in the `duckduckgo_search` method
- **Port**: Change the port in `app/main.py` or use environment variables

## Error Handling

The application includes comprehensive error handling for:
- Search failures
- LLM errors
- Invalid queries
- Graph execution errors

## Development

To add new capabilities:
1. Add new nodes to the graph in `app/graph.py`
2. Define the node function
3. Add appropriate edges and routing logic
4. Update the state schema if needed

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running and the model is downloaded
- **Search Failures**: Check internet connectivity and DuckDuckGo availability
- **Graph Compilation Errors**: Verify all nodes and edges are properly defined

## License

This project is licensed under the MIT License.
