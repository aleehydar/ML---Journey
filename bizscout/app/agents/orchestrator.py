from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from app.agents.market_agent import market_agent
from app.agents.competitor_agent import competitor_agent
from app.agents.regulatory_agent import regulatory_agent
from app.agents.financial_agent import financial_agent
from app.agents.risk_agent import risk_agent

class BusinessState(TypedDict):
    business_idea: str
    city: str
    budget_pkr: int
    market_data: Optional[Dict[str, Any]]
    competitor_data: Optional[Dict[str, Any]]
    regulatory_data: Optional[Dict[str, Any]]
    financial_data: Optional[Dict[str, Any]]
    risk_data: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]

def create_graph():
    workflow = StateGraph(BusinessState)
    
    workflow.add_node("market", market_agent)
    workflow.add_node("competitor", competitor_agent)
    workflow.add_node("regulatory", regulatory_agent)
    workflow.add_node("financial", financial_agent)
    workflow.add_node("risk", risk_agent)
    
    # We will run them sequentially to ensure state is cleanly passed and 
    # to avoid multiple triggers on the financial node in basic LangGraph.
    # While they can be parallelized, a sequential flow ensures 100% deterministic results here.
    workflow.add_edge(START, "market")
    workflow.add_edge("market", "competitor")
    workflow.add_edge("competitor", "regulatory")
    
    # Financial runs after the first three have collected their data
    workflow.add_edge("regulatory", "financial")
    
    # Risk runs last because it needs everything
    workflow.add_edge("financial", "risk")
    workflow.add_edge("risk", END)
    
    return workflow.compile()

def run_analysis(business_idea: str, city: str, budget_pkr: int) -> Dict[str, Any]:
    graph = create_graph()
    initial_state = {
        "business_idea": business_idea,
        "city": city,
        "budget_pkr": budget_pkr,
    }
    
    result = graph.invoke(initial_state)
    return result.get("final_report", {})
