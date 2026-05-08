from typing import Dict, Any, List
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from app.tools.search import pakistan_search
from app.tools.calculator import calculate_breakeven_months, check_budget_sufficient

class FinancialData(BaseModel):
    setup_cost_pkr: int = Field(description="Estimated setup cost in PKR")
    monthly_cost_pkr: int = Field(description="Estimated monthly operational cost in PKR")
    breakeven_months: int = Field(description="Estimated breakeven timeline in months")
    budget_sufficient: bool = Field(description="True if the user's budget is sufficient, False otherwise")
    funding_options: List[str] = Field(description="Potential funding options if needed")
    estimated_monthly_revenue_pkr: int = Field(description="Estimated monthly revenue in PKR")

def financial_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    business_idea = state.get("business_idea", "")
    city = state.get("city", "")
    budget_pkr = state.get("budget_pkr", 0)
    
    # We can also read from previous agents, but let's do a search specific for costs
    search_results = pakistan_search(f"startup costs for {business_idea} in {city} PKR")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(FinancialData)
    
    prompt = f"""
    You are a financial advisor in Pakistan. Estimate the costs for:
    Idea: {business_idea}
    City: {city}
    User Budget: {budget_pkr} PKR
    
    Search Results:
    {search_results}
    
    Provide reasonable estimates for:
    - setup_cost_pkr
    - monthly_cost_pkr
    - estimated_monthly_revenue_pkr (realistic guess based on market)
    - funding_options
    
    Note: Your returned JSON must contain breakeven_months and budget_sufficient, but we will override them with calculation tools.
    """
    
    response = structured_llm.invoke(prompt)
    
    # Use our tools to calculate breakeven and check budget
    breakeven = calculate_breakeven_months(
        response.setup_cost_pkr, 
        response.monthly_cost_pkr, 
        response.estimated_monthly_revenue_pkr
    )
    
    is_sufficient = check_budget_sufficient(
        budget_pkr, 
        response.setup_cost_pkr, 
        response.monthly_cost_pkr
    )
    
    final_data = response.dict()
    final_data["breakeven_months"] = breakeven
    final_data["budget_sufficient"] = is_sufficient
    
    return {"financial_data": final_data}
