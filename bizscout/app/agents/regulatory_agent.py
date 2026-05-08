from typing import Dict, Any, List
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from app.tools.search import pakistan_search

class RegulatoryData(BaseModel):
    required_licenses: List[str] = Field(description="List of required local/municipal/sector licenses")
    secp_requirements: str = Field(description="SECP and FBR registration requirements")
    regulatory_complexity: str = Field(description="Complexity: Easy, Medium, or Hard")
    estimated_compliance_time_weeks: int = Field(description="Estimated time in weeks to get compliant")

def regulatory_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    business_idea = state.get("business_idea", "")
    
    search_results = pakistan_search(f"SECP FBR regulations licenses for {business_idea}")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(RegulatoryData)
    
    prompt = f"""
    You are a legal and regulatory expert in Pakistan. Analyze the regulations for:
    Idea: {business_idea}
    
    Search Results:
    {search_results}
    
    Identify required licenses (FBR NTN, local, etc.), SECP requirements, regulatory complexity (Easy/Medium/Hard), and compliance time in weeks.
    """
    
    response = structured_llm.invoke(prompt)
    return {"regulatory_data": response.dict()}
