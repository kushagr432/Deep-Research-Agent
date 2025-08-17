"""
Vector Database Service for Financial Research Chatbot
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class VectorDBService:
    """Vector database service wrapper supporting Pinecone and FAISS"""
    
    def __init__(self, vector_db_type: str = "mock", **kwargs):
        """
        Initialize vector database service
        
        Args:
            vector_db_type: Type of vector database ("pinecone", "faiss", or "mock")
            **kwargs: Additional configuration parameters
        """
        self.vector_db_type = vector_db_type
        self.connected = False
        self.stats = {
            "searches": 0,
            "embeddings_created": 0,
            "errors": 0
        }
        
        # Configuration
        self.config = kwargs
        
        # Mock data for demonstration
        self._mock_documents = self._create_mock_documents()
        self._mock_embeddings = self._create_mock_embeddings()
        
        # Initialize based on type
        if vector_db_type == "pinecone":
            self._init_pinecone()
        elif vector_db_type == "faiss":
            self._init_faiss()
        else:
            self._init_mock()
    
    def _init_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            # In production, this would initialize Pinecone client
            # import pinecone
            # pinecone.init(api_key=self.config.get("api_key"), environment=self.config.get("environment"))
            # self.index = pinecone.Index(self.config.get("index_name"))
            logger.info("Pinecone initialization requested (mock mode)")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self._init_mock()
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        try:
            # In production, this would create/load FAISS index
            # import faiss
            # self.index = faiss.IndexFlatL2(self.config.get("dimension", 768))
            logger.info("FAISS initialization requested (mock mode)")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self._init_mock()
    
    def _init_mock(self):
        """Initialize mock vector database"""
        self.connected = True
        logger.info("Using mock vector database for demonstration")
    
    def _create_mock_documents(self) -> List[Dict[str, Any]]:
        """Create mock financial documents for demonstration"""
        return [
            {
                "id": "doc_001",
                "title": "Investment Portfolio Diversification",
                "content": "Diversification is a risk management strategy that mixes a wide variety of investments within a portfolio. A diversified portfolio contains a mix of distinct asset types and investment vehicles in an attempt at limiting exposure to any single asset or risk.",
                "category": "investment",
                "tags": ["diversification", "risk management", "portfolio"],
                "created_at": "2024-01-15T10:00:00Z"
            },
            {
                "id": "doc_002",
                "title": "Retirement Planning Strategies",
                "content": "Retirement planning involves determining retirement income goals and the actions necessary to achieve those goals. This includes identifying sources of income, estimating expenses, implementing a savings program, and managing assets and risk.",
                "category": "retirement",
                "tags": ["retirement", "planning", "savings"],
                "created_at": "2024-01-16T11:00:00Z"
            },
            {
                "id": "doc_003",
                "title": "Credit Score Management",
                "content": "Your credit score is a three-digit number that lenders use to assess your creditworthiness. Factors that affect your credit score include payment history, credit utilization, length of credit history, and types of credit accounts.",
                "category": "credit",
                "tags": ["credit score", "lending", "financial health"],
                "created_at": "2024-01-17T12:00:00Z"
            },
            {
                "id": "doc_004",
                "title": "Mortgage Rate Comparison",
                "content": "When comparing mortgage rates, consider both the interest rate and the annual percentage rate (APR). The APR includes the interest rate plus other loan costs such as broker fees, discount points, and some closing costs.",
                "category": "mortgage",
                "tags": ["mortgage", "rates", "APR", "closing costs"],
                "created_at": "2024-01-18T13:00:00Z"
            },
            {
                "id": "doc_005",
                "title": "Emergency Fund Planning",
                "content": "An emergency fund is a financial safety net designed to cover unexpected expenses. Financial experts recommend saving three to six months' worth of living expenses in an easily accessible account.",
                "category": "savings",
                "tags": ["emergency fund", "savings", "financial security"],
                "created_at": "2024-01-19T14:00:00Z"
            }
        ]
    
    def _create_mock_embeddings(self) -> Dict[str, List[float]]:
        """Create mock embeddings for demonstration"""
        # Generate random embeddings (in production, these would be real embeddings)
        np.random.seed(42)  # For reproducible results
        embeddings = {}
        
        for doc in self._mock_documents:
            # Generate 768-dimensional embedding (typical for many models)
            embedding = np.random.normal(0, 1, 768).tolist()
            embeddings[doc["id"]] = embedding
        
        return embeddings
    
    async def connect(self) -> bool:
        """Connect to vector database"""
        try:
            if self.vector_db_type == "mock":
                self.connected = True
                return True
            else:
                # In production, implement actual connection logic
                logger.info(f"Connecting to {self.vector_db_type}...")
                await asyncio.sleep(0.1)  # Simulate connection time
                self.connected = True
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to vector database: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from vector database"""
        self.connected = False
        logger.info("Disconnected from vector database")
    
    async def search(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity
        
        Args:
            query: Search query
            top_k: Number of top results to return
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of similar documents with similarity scores
        """
        try:
            if not self.connected:
                logger.warning("Vector database not connected, returning empty results")
                return []
            
            # Simulate async processing
            await asyncio.sleep(0.05)
            
            # Mock similarity search
            results = self._mock_similarity_search(query, top_k, threshold)
            
            self.stats["searches"] += 1
            logger.info(f"Vector search completed for query: {query[:50]}...")
            
            return results
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in vector search: {e}")
            return []
    
    def _mock_similarity_search(self, query: str, top_k: int, threshold: float) -> List[Dict[str, Any]]:
        """Mock similarity search implementation"""
        query_lower = query.lower()
        scored_docs = []
        
        for doc in self._mock_documents:
            # Simple keyword-based scoring (in production, use actual vector similarity)
            score = 0.0
            
            # Check title relevance
            if any(word in doc["title"].lower() for word in query_lower.split()):
                score += 0.4
            
            # Check content relevance
            if any(word in doc["content"].lower() for word in query_lower.split()):
                score += 0.3
            
            # Check tag relevance
            if any(word in " ".join(doc["tags"]).lower() for word in query_lower.split()):
                score += 0.2
            
            # Check category relevance
            if any(word in doc["category"].lower() for word in query_lower.split()):
                score += 0.1
            
            # Add some randomness to simulate real similarity scores
            score += np.random.uniform(0, 0.1)
            
            if score >= threshold:
                scored_docs.append({
                    "document": doc,
                    "score": min(score, 1.0),
                    "embedding_id": doc["id"]
                })
        
        # Sort by score and return top_k results
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]
    
    async def add_document(self, document: Dict[str, Any], embedding: Optional[List[float]] = None) -> bool:
        """
        Add a document to the vector database
        
        Args:
            document: Document to add
            embedding: Optional pre-computed embedding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.connected:
                return False
            
            # Generate embedding if not provided
            if embedding is None:
                embedding = await self._generate_embedding(document["content"])
            
            # In production, add to actual vector database
            # For mock, just add to local storage
            doc_id = f"doc_{len(self._mock_documents) + 1:03d}"
            document["id"] = doc_id
            document["created_at"] = datetime.now().isoformat()
            
            self._mock_documents.append(document)
            self._mock_embeddings[doc_id] = embedding
            
            self.stats["embeddings_created"] += 1
            logger.info(f"Added document to vector database: {doc_id}")
            
            return True
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error adding document: {e}")
            return False
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (mock implementation)"""
        # In production, this would call an embedding model
        # For now, generate random embedding
        np.random.seed(hash(text) % 2**32)  # Deterministic based on text
        return np.random.normal(0, 1, 768).tolist()
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector database
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.connected:
                return False
            
            # Remove from mock storage
            self._mock_documents = [doc for doc in self._mock_documents if doc["id"] != doc_id]
            if doc_id in self._mock_embeddings:
                del self._mock_embeddings[doc_id]
            
            logger.info(f"Deleted document from vector database: {doc_id}")
            return True
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error deleting document: {e}")
            return False
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document or None if not found
        """
        try:
            if not self.connected:
                return None
            
            for doc in self._mock_documents:
                if doc["id"] == doc_id:
                    return doc
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check vector database health"""
        try:
            if not self.connected:
                return {"status": "disconnected", "error": "Not connected to vector database"}
            
            # Test search functionality
            test_results = await self.search("test", top_k=1)
            
            return {
                "status": "healthy",
                "connected": True,
                "vector_db_type": self.vector_db_type,
                "total_documents": len(self._mock_documents),
                "test_search_successful": len(test_results) > 0
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        return {
            "connected": self.connected,
            "vector_db_type": self.vector_db_type,
            "stats": self.stats.copy(),
            "total_documents": len(self._mock_documents),
            "last_updated": datetime.now().isoformat()
        }
    
    async def clear_stats(self):
        """Clear vector database statistics"""
        self.stats = {
            "searches": 0,
            "embeddings_created": 0,
            "errors": 0
        }
        logger.info("Vector database statistics cleared")
