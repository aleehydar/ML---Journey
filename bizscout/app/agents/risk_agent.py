from typing import Dict, Any, List
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

class RiskData(BaseModel):
    viability_score: int = Field(description="Overall viability score 0-100")
    recommendation: str = Field(description="GO, PROCEED WITH CAUTION, or NO GO")
    top_risks: List[str] = Field(description="Top 3 risks specific to Pakistan")
    executive_summary: str = Field(description="A 3-sentence executive summary")
    action_items: List[str] = Field(description="Numbered list of action items")

def risk_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    business_idea = state.get("business_idea", "")
    city = state.get("city", "")
    budget_pkr = state.get("budget_pkr", 0)
    
    market_data = state.get("market_data", {})
    competitor_data = state.get("competitor_data", {})
    regulatory_data = state.get("regulatory_data", {})
    financial_data = state.get("financial_data", {})
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    structured_llm = llm.with_structured_output(RiskData)
    
    prompt = f"""
    You are the final Risk and Viability Analyst for a new business in Pakistan.
    Synthesize the following data and provide a final recommendation.
    
    Business: {business_idea} in {city} with budget {budget_pkr} PKR
    
    Market Data: {market_data}
    Competitor Data: {competitor_data}
    Regulatory Data: {regulatory_data}
    Financial Data: {financial_data}
    
    Identify top 3 risks specific to Pakistan (e.g. power outages, currency, inflation).
    Calculate a viability score (0-100).
    Give a recommendation (exactly one of: GO, PROCEED WITH CAUTION, NO GO).
    Write a 3-sentence executive summary.
    Provide actionable steps.
    """
    
    response = structured_llm.invoke(prompt)
    
    final_report = {
        "business_idea": business_idea,
        "city": city,
        "budget_pkr": budget_pkr,
        "market_data": market_data,
        "competitor_data": competitor_data,
        "regulatory_data": regulatory_data,
        "financial_data": financial_data,
        "risk_data": response.dict()
    }
    
    return {"risk_data": response.dict(), "final_report": final_report}
