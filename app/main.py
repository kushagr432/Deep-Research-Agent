# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.graph import FinancialResearchGraph
import asyncio
import os
import json

app = FastAPI(title="Financial & Deep Research AI Chatbot")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
# Initialize LangGraph / LangChain graph
graph = FinancialResearchGraph()

# Request model
class QueryRequest(BaseModel):
    user_query: str
    deep_research: bool = False   # user specifies if deep research is needed
    generate_report: bool = False # user specifies if PDF report should be generated
    generate_dashboard: bool = False # user specifies if HTML dashboard should be generated
    session_id: str = None        # optional for session tracking

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        # Process the query through the graph with user-provided flags
        final_state = await graph.process(
            query=request.user_query,
            session_id=request.session_id,
            state={
                "deep_research": request.deep_research,
                "generate_report": request.generate_report,
                "generate_dashboard": request.generate_dashboard
            }
        )
        
        # Extract the response from the final state
        response_text = final_state.get("response", "No response generated")
        
        # Ensure all data is JSON-serializable
        def ensure_json_safe(obj):
            import pandas as pd
            import numpy as np
            
            if isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d')
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(key): ensure_json_safe(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [ensure_json_safe(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj) if obj is not None else None
        
        # Create safe response data
        safe_yahoo_data = ensure_json_safe(final_state.get("yahoo_finance_data"))
        
        return JSONResponse({
            "response": response_text,
            "intent": final_state.get("intent"),
            "deep_research": final_state.get("deep_research"),
            "generate_report": final_state.get("generate_report"),
            "generate_dashboard": final_state.get("generate_dashboard"),
            "report_generated": final_state.get("report_generated"),
            "dashboard_generated": final_state.get("dashboard_generated"),
            "pdf_report_path": final_state.get("pdf_report_path"),
            "dashboard_path": final_state.get("dashboard_path"),
            "yahoo_finance_data": safe_yahoo_data,
            "session_id": request.session_id #this is just for reference no implementation currently
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """Streaming endpoint for real-time chat-like responses"""
    try:
        async def generate_stream():
            # Start processing
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting analysis...'})}\n\n"
            
            # Route to appropriate streaming agent based on request
            if request.deep_research:
                # Deep research with streaming
                async for result in graph.deep_research_agent.process_stream(
                    request.user_query,
                    generate_report=request.generate_report,
                    generate_dashboard=request.generate_dashboard
                ):
                    yield f"data: {json.dumps(result)}\n\n"
            else:
                # Simple intent detection for banking/finance
                if any(word in request.user_query.lower() for word in ["bank", "account", "loan", "credit", "mortgage"]):
                    # Banking query
                    async for result in graph.banking_agent.process_stream(request.user_query):
                        yield f"data: {json.dumps(result)}\n\n"
                else:
                    # Finance query
                    async for result in graph.finance_agent.process_stream(request.user_query):
                        yield f"data: {json.dumps(result)}\n\n"
            
            # Send completion message
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Analysis complete!'})}\n\n"
            
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing streaming query: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/download-report/{filename}")
async def download_report(filename: str):
    """Download a generated PDF report"""
    try:
        # Security: Only allow PDF files from reports directory
        if not filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        file_path = os.path.join("reports", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Return the file for download
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/pdf'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading report: {str(e)}")

@app.get("/download-dashboard/{filename}")
async def download_dashboard(filename: str):
    """Download a generated HTML dashboard"""
    try:
        # Security: Only allow HTML files from reports directory
        if not filename.endswith('.html'):
            raise HTTPException(status_code=400, detail="Only HTML files are allowed")
        
        file_path = os.path.join("reports", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        # Return the file for download with proper HTML media type
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='text/html'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading dashboard: {str(e)}")

@app.get("/reports")
async def list_reports():
    """List all available reports"""
    try:
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            return {"reports": []}
        
        reports = []
        for filename in os.listdir(reports_dir):
            if filename.endswith('.pdf'):
                file_path = os.path.join(reports_dir, filename)
                file_size = os.path.getsize(file_path)
                file_time = os.path.getmtime(file_path)
                
                reports.append({
                    "filename": filename,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "created": file_time,
                    "download_url": f"/download-report/{filename}"
                })
        
        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x["created"], reverse=True)
        
        return {"reports": reports}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing reports: {str(e)}")

@app.get("/dashboards")
async def list_dashboards():
    """List all available HTML dashboards"""
    try:
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            return {"dashboards": []}
        
        dashboards = []
        for filename in os.listdir(reports_dir):
            if filename.endswith('.html'):
                file_path = os.path.join(reports_dir, filename)
                file_size = os.path.getsize(file_path)
                file_time = os.path.getmtime(file_path)
                
                dashboards.append({
                    "filename": filename,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "created": file_time,
                    "download_url": f"/download-dashboard/{filename}"
                })
        
        # Sort by creation time (newest first)
        dashboards.sort(key=lambda x: x["created"], reverse=True)
        
        return {"dashboards": dashboards}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing dashboards: {str(e)}")

@app.get("/view-dashboard/{filename}")
async def view_dashboard(filename: str):
    """View a generated HTML dashboard in the browser"""
    try:
        # Security: Only allow HTML files from reports directory
        if not filename.endswith('.html'):
            raise HTTPException(status_code=400, detail="Only HTML files are allowed")
        
        file_path = os.path.join("reports", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        # Return the HTML file for viewing in browser
        return FileResponse(
            path=file_path,
            media_type='text/html'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error viewing dashboard: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
