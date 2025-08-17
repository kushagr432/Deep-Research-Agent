"""
FastAPI entrypoint for Financial Research Chatbot
"""
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from app.graph import FinancialResearchGraph
from app.services.cache import CacheService
from app.services.queue import QueueService
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Research Chatbot",
    description="AI-powered financial research assistant using LangGraph and vector search - Streaming Only",
    version="1.0.0"
)

# Create reports directory if it doesn't exist
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=reports_dir), name="static")

# Initialize services
cache_service = CacheService()
queue_service = QueueService()
graph = FinancialResearchGraph(cache_service, queue_service)

class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    deep_research: bool = False
    generate_pdf: bool = False

class QueryResponse(BaseModel):
    answer: str
    cached: bool
    processing_time: float
    needs_clarification: Optional[bool] = False
    clarification_questions: Optional[str] = None
    agent_used: Optional[str] = None
    vector_results_count: Optional[int] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        await cache_service.connect()
        await queue_service.connect()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on shutdown"""
    try:
        await cache_service.disconnect()
        await queue_service.disconnect()
        logger.info("Services cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Financial Research Chatbot API", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    cache_status = await cache_service.health_check()
    queue_status = await queue_service.health_check()
    
    return {
        "status": "healthy",
        "cache": cache_status,
        "queue": queue_status,
        "timestamp": asyncio.get_event_loop().time()
    }

@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Process financial research queries with streaming responses only
    
    Args:
        request: QueryRequest containing the user's question
        
    Returns:
        StreamingResponse with real-time updates
    """
    try:
        logger.info(f"Processing streaming query: {request.query[:100]}...")
        
        # Let the graph handle all query processing
        return StreamingResponse(
            stream_graph_results(request.query, request.user_id, request.deep_research, request.generate_pdf),
            media_type="text/plain"
        )
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )

async def stream_graph_results(query: str, user_id: str, deep_research: bool, generate_pdf: bool):
    """Stream results directly from the LangGraph workflow with live updates"""
    try:
        # Use the new streaming method from the graph
        async for chunk in graph.execute_query_streaming(query, user_id, deep_research, generate_pdf):
            yield f"data: {json.dumps(chunk)}\n\n"
        
    except Exception as e:
        error_msg = {'status': 'error', 'message': str(e)}
        yield f"data: {json.dumps(error_msg)}\n\n"

@app.get("/download-report/{filename}")
async def download_report(filename: str):
    """
    Download a generated research report
    
    Args:
        filename: Name of the report file to download
        
    Returns:
        FileResponse with the report file
    """
    try:
        file_path = f"reports/{filename}"
        if os.path.exists(file_path):
            print(f"Downloading report from {file_path}")
            print(f"Filename: {filename}")
            return FileResponse(
                path=file_path,
                filename=filename,
                media_type='application/octet-stream'
            )
        else:
            raise HTTPException(status_code=404, detail="Report not found")
            
    except Exception as e:
        logger.error(f"Error downloading report {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to download report")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        cache_stats = await cache_service.get_stats()
        queue_stats = await queue_service.get_stats()
        
        return {
            "cache": cache_stats,
            "queue": queue_stats,
            "graph_executions": graph.get_execution_stats()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
