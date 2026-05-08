import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8002")

st.set_page_config(page_title="BizScout Pakistan", page_icon="🚀", layout="centered")

st.title("BizScout Pakistan — AI Business Intelligence")
st.subheader("Validate your business idea in 2 minutes")

VALID_CITIES = [
    "Karachi", "Lahore", "Islamabad", "Peshawar", 
    "Quetta", "Multan", "Faisalabad", "Rawalpindi"
]

business_idea = st.text_area("Describe your business idea", 
                             placeholder="e.g. I want to open a cloud kitchen in Karachi serving healthy meal preps")
city = st.selectbox("Select your city", VALID_CITIES)
budget_pkr = st.slider("Your budget (PKR)", min_value=100000, max_value=10000000, step=100000, value=1000000)

if st.button("Analyze My Business Idea", use_container_width=True, type="primary"):
    if not business_idea:
        st.error("Please describe your business idea first.")
    else:
        with st.spinner("5 AI agents are researching your idea..."):
            try:
                response = requests.post(
                    f"{API_URL}/analyze", 
                    json={"business_idea": business_idea, "city": city, "budget_pkr": budget_pkr}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.divider()
                    
                    risk = data.get("risk_data", {})
                    score = risk.get("viability_score", 0)
                    rec = risk.get("recommendation", "UNKNOWN")
                    
                    if score >= 70:
                        score_color = "#28a745"
                    elif score >= 40:
                        score_color = "#ffc107"
                    else:
                        score_color = "#dc3545"
                        
                    st.markdown(f"<h1 style='text-align: center; color: {score_color}; font-size: 80px; margin: 0;'>{score}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center; margin-top: 0;'>{rec}</h3>", unsafe_allow_html=True)
                    
                    st.info(f"**Executive Summary:**\n\n{risk.get('executive_summary', '')}")
                    
                    with st.expander("📊 Market Analysis"):
                        st.json(data.get("market_data", {}))
                    with st.expander("🏆 Competitor Analysis"):
                        st.json(data.get("competitor_data", {}))
                    with st.expander("⚖️ Regulatory Analysis"):
                        st.json(data.get("regulatory_data", {}))
                    with st.expander("💰 Financial Analysis"):
                        st.json(data.get("financial_data", {}))
                    with st.expander("⚠️ Risk Analysis"):
                        st.json(risk)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Top Risks")
                        for r in risk.get("top_risks", []):
                            st.markdown(f"- {r}")
                    with col2:
                        st.subheader("Action Items")
                        for i, action in enumerate(risk.get("action_items", [])):
                            st.markdown(f"{i+1}. {action}")
                            
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend. Is it running? Error: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: small;'>Powered by LangGraph + Groq | Built by Ali Haidar<br>For research purposes only. Consult professionals before investing.</p>", unsafe_allow_html=True)
