# Financial Research Chatbot

A production-ready AI-powered financial research assistant built with FastAPI, LangGraph, Redis, Kafka, and vector search capabilities.

## 🚀 Features

- **Intelligent Query Processing**: Uses LangGraph to orchestrate complex financial research workflows
- **Multi-Agent Architecture**: Specialized agents for finance and banking queries
- **Vector Knowledge Retrieval**: Integrates with Pinecone/FAISS for semantic search
- **Caching Layer**: Redis-based caching for improved performance and cost reduction
- **Async Message Processing**: Kafka integration for handling high-volume requests
- **Production-Ready**: Comprehensive error handling, logging, and monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App  │    │   LangGraph     │    │   Vector DB     │
│                 │    │   Workflow      │    │   (Pinecone/    │
│  - /query      │◄──►│                 │◄──►│   FAISS)        │
│  - /health     │    │  - Query Flow   │    │                 │
│  - /stats      │    │  - Agent Routing│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Cache   │    │   Kafka Queue   │    │   LLM Agents    │
│                 │    │                 │    │                 │
│  - Query Cache │    │  - Async Msgs   │    │  - FinanceAgent │
│  - Response    │    │  - Event Stream │    │  - BankingAgent │
│  - User Data   │    │  - Load Balance │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
financial-research-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entrypoint
│   ├── graph.py             # LangGraph workflow definition
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── finance_agent.py # General financial queries
│   │   └── banking_agent.py # Banking-specific queries
│   └── services/
│       ├── __init__.py
│       ├── cache.py         # Redis integration
│       ├── queue.py         # Kafka producer/consumer
│       └── vector_db.py     # Vector DB wrapper
├── config.py                # Configuration management
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Redis server
- Kafka cluster (optional for development)
- Docker (recommended)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd financial-research-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file based on the configuration in `config.py`:

```bash
# Copy and modify the example configuration
cp config.py .env
```

### 5. Start Infrastructure Services

#### Using Docker Compose (Recommended)

```bash
# Start Redis and Kafka
docker-compose up -d redis kafka
```

#### Manual Setup

```bash
# Start Redis
redis-server

# Start Kafka (requires separate setup)
# Follow Kafka documentation for your platform
```

### 6. Run the Application

```bash
# Development mode
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🚀 Usage

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Process Financial Query
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How should I diversify my investment portfolio?",
       "user_id": "user123"
     }'
```

#### 3. Get System Statistics
```bash
curl http://localhost:8000/stats
```

### Example Queries

#### Finance Queries
- "What are the best investment strategies for beginners?"
- "How much should I save for retirement?"
- "What is compound interest and how does it work?"

#### Banking Queries
- "How do I improve my credit score?"
- "What should I look for when comparing mortgage rates?"
- "What are the benefits of high-yield savings accounts?"

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka brokers | `localhost:9092` |
| `VECTOR_DB_TYPE` | Vector database type | `mock` |
| `PINECONE_API_KEY` | Pinecone API key | `None` |
| `FASTAPI_PORT` | Application port | `8000` |

### Vector Database Options

1. **Mock Mode** (Default): In-memory mock database for development
2. **Pinecone**: Cloud-based vector database
3. **FAISS**: Facebook's similarity search library

## 🧪 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest
```

### Adding New Agents

1. Create a new agent class in `app/agents/`
2. Implement the `process_query` method
3. Add routing logic in `app/graph.py`
4. Update the agent initialization

### Adding New Services

1. Create a new service class in `app/services/`
2. Implement required methods
3. Add to the main application initialization
4. Update health checks and statistics

## 📊 Monitoring & Observability

### Health Checks

- `/health` - Overall system health
- Service-specific health checks for Redis, Kafka, and Vector DB

### Statistics

- `/stats` - System performance metrics
- Cache hit/miss ratios
- Query processing times
- Agent usage statistics

### Logging

- Structured logging with configurable levels
- Request/response logging
- Error tracking and reporting

## 🚀 Production Deployment

### Docker Deployment

```bash
# Build image
docker build -t financial-chatbot .

# Run container
docker run -p 8000:8000 \
  -e REDIS_URL=redis://redis:6379 \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  financial-chatbot
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financial-chatbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: financial-chatbot
  template:
    metadata:
      labels:
        app: financial-chatbot
    spec:
      containers:
      - name: chatbot
        image: financial-chatbot:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

## 🔒 Security Considerations

- API key management for external services
- Rate limiting and request validation
- Input sanitization and validation
- Secure communication protocols
- Access control and authentication (to be implemented)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the code examples

## 🔮 Future Enhancements

- [ ] Real LLM integration (OpenAI, Anthropic)
- [ ] Advanced query classification
- [ ] Multi-language support
- [ ] User authentication and authorization
- [ ] Advanced analytics and reporting
- [ ] WebSocket support for real-time updates
- [ ] Integration with financial data APIs
- [ ] Machine learning model training pipeline
