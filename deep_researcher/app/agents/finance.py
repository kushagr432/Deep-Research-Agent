from fastapi import APIRouter
from app.services.cache_service import get_cached_response, save_to_cache
from app.services.queue_service import send_to_queue

router = APIRouter()

@router.post("/query")
async def finance_query(query: str):
    # Step 1: Check cache
    cached = get_cached_response(query)
    if cached:
        return {"response": cached, "source": "cache"}

    # Step 2: Send to Kafka (async processing)
    send_to_queue("finance-queue", query)

    # Step 3: Dummy response for now
    response = f"Finance analysis for: {query}"
    save_to_cache(query, response)

    return {"response": response, "source": "llm"}
