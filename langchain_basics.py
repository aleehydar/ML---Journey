from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

response = llm.invoke([HumanMessage(content="What is machine learning in one sentence?")])
print(response.content)
from langchain_core.messages import SystemMessage

response2 = llm.invoke([
    SystemMessage(content="You are a helpful ML tutor who explains concepts simply."),
    HumanMessage(content="Explain overfitting in simple words.")
])
print(response2.content)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert data scientist. Answer questions clearly and concisely."),
    ("human", "{question}")
])

chain = prompt | llm

response3 = chain.invoke({"question": "What is the difference between Random Forest and XGBoost?"})
print(response3.content)