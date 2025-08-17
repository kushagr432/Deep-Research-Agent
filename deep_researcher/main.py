from fastapi import FastAPI
from app.agents.finance import router as finance_router
from app.agents.banking import router as banking_router

app = FastAPI(title="Financial Research Chatbot")

# Register routers (each agent)
app.include_router(finance_router, prefix="/finance", tags=["Finance Agent"])
app.include_router(banking_router, prefix="/banking", tags=["Banking Agent"])

@app.get("/")
def root():
    return {"msg": "Financial Chatbot is running 🚀"}
