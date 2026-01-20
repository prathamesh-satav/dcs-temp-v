from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ai_service import SegmentationAgent
import uvicorn
import os

# Definition of the Request Body
class QueryRequest(BaseModel):
    api_key: str
    cluster_summary: str
    user_query: str

app = FastAPI(title="Customer Segmentation AI Agent")

@app.get("/")
def home():
    return {"status": "active", "service": "Segmentation Agent V2"}

@app.post("/chat")
def chat_with_data(request: QueryRequest):
    """
    Endpoint that Streamlit calls to get AI insights.
    """
    try:
        # Initialize Agent with the key provided in the request
        agent = SegmentationAgent(api_key=request.api_key)
        
        # Run Analysis
        response = agent.analyze_clusters(
            cluster_summary=request.cluster_summary,
            user_query=request.user_query
        )
        
        return {"response": response}
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)