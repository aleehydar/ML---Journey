from typing import Dict, Any, List
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from app.tools.search import pakistan_search

class CompetitorData(BaseModel):
    competitor_count: int = Field(description="Estimated number of competitors")
    top_competitors: List[str] = Field(description="List of 3-5 top competitors with their strengths/weaknesses")
    market_gaps: List[str] = Field(description="Identified gaps in the market")
    saturation_level: str = Field(description="Market saturation level: Low, Medium, or High")

def competitor_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    business_idea = state.get("business_idea", "")
    city = state.get("city", "")
    
    search_results = pakistan_search(f"competitors for {business_idea} in {city}")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(CompetitorData)
    
    prompt = f"""
    You are a competitor research analyst in Pakistan. Analyze existing businesses for:
    Idea: {business_idea}
    City: {city}
    
    Search Results:
    {search_results}
    
    Identify the number of competitors, top competitors (with strengths/weaknesses), market gaps, and overall saturation level.
    """
    
    response = structured_llm.invoke(prompt)
    return {"competitor_data": response.dict()}
