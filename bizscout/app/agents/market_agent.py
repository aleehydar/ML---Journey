from typing import Dict, Any, List
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from app.tools.search import pakistan_search

class MarketData(BaseModel):
    demand_score: int = Field(description="Demand score 1-10")
    market_size_estimate: str = Field(description="Rough market size estimate in PKR")
    key_trends: List[str] = Field(description="List of key trends in this sector")
    sources: List[str] = Field(description="List of sources or URLs")

def market_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    business_idea = state.get("business_idea", "")
    city = state.get("city", "")
    
    search_results = pakistan_search(f"{business_idea} market demand {city}")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(MarketData)
    
    prompt = f"""
    You are a market demand analyst in Pakistan. Analyze demand for:
    Idea: {business_idea}
    City: {city}
    
    Search Results:
    {search_results}
    
    Extract the demand score (1-10), estimate the market size in PKR, list key trends, and provide sources.
    """
    
    response = structured_llm.invoke(prompt)
    return {"market_data": response.dict()}
