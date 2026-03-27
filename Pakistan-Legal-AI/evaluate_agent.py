"""
Pakistan Legal Intelligence - Agentic Pipeline Evaluator
--------------------------------------------------------
This script evaluates the production agent, ensuring it routes 
to the correct tools (Tax vs. Legal Search) and provides faithful 
answers backed by the injected PyMuPDF FAISS Index.
"""

import asyncio
import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Import the main agent generator
from pakistan_legal_assistant import answer_legal_question

load_dotenv()

# We test both Tax math (Tool 1) and Legal Semantic Retrieval (Tool 2)
TEST_CASES = [
    {
        "type": "math",
        "question": "My monthly salary is 200,000 PKR. What is my monthly income tax?",
        "ground_truth": "Annual income is 2,400,000. Tax is 15,000 + 12.5% of (2,400,000 - 1,200,000) = 165,000 annual. Monthly tax is 13,750 PKR."
    },
    {
        "type": "math",
        "question": "Calculate the annual tax for a person earning 800,000 PKR per month.",
        "ground_truth": "Annual income is 9,600,000. Tax is 592,500 + 35% of (9,600,000 - 4,100,000) = 2,517,500 PKR annual."
    },
    {
        "type": "law",
        "question": "Is an accomplice competent to testify against an accused person?",
        "ground_truth": "Yes, according to the Qanun-e-Shahadat, an accomplice is a competent witness against an accused person."
    },
    {
        "type": "law",
        "question": "Can the President or Prime Minister of Pakistan be sued or prosecuted for their official duties while in office?",
        "ground_truth": "According to the Constitution of Pakistan (Article 248), they have immunity. No criminal proceedings shall be instituted against the President or a Governor during their term of office."
    },
    {
        "type": "law",
        "question": "As a doctor, can I be forced to disclose confidential communications made by my patient?",
        "ground_truth": "No, under the Qanun-e-Shahadat, professional communications are generally privileged unless the client consents or it involves an illegal purpose."
    }
]

judge_llm = ChatGroq(model="llama-3.1-8b-instant")

async def get_agent_answer(question: str) -> str:
    """Capture the streamed output from the main agent."""
    generator = answer_legal_question(question, history=[])
    
    full_answer = ""
    async for event_str in generator:
        if event_str.startswith("data: "):
            try:
                payload = json.loads(event_str[6:].strip())
                if payload["type"] == "token":
                    full_answer += payload["data"]
            except:
                pass
    return full_answer

async def judge_faithfulness(question: str, generated_answer: str, ground_truth: str) -> dict:
    judge_prompt = f"""You are an expert AI evaluator grading an AI Legal Assistant.

GROUND TRUTH FACT/LAW: {ground_truth}

AI MODEL'S ANSWER: {generated_answer}

QUESTION: {question}

TASK: Judge whether the AI's answer fundamentally aligns with the Ground Truth Fact.
- Score 1.0 if the core legal or mathematical conclusion matches the ground truth. It is okay if the model provides extra helpful context, as long as the core fact is right.
- Score 0.5 if the answer is partially right but misses a detail.
- Score 0.0 if the answer contradicts the core truth, hallucinates, or gets the math calculation wrong.

Respond with ONLY a single JSON object. Example: {{"score": 1.0, "reason": "The answer accurately states..."}}

JSON Response:"""
    
    response = await judge_llm.ainvoke([HumanMessage(content=judge_prompt)])
    
    try:
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return {"score": float(result.get("score", 0.0)), "verdict": result.get("reason", "N/A")}
    except Exception:
        pass
    
    return {"score": 0.0, "verdict": "Judge error"}

async def run_evaluation():
    print("=" * 80)
    print("  Pakistan Legal Intelligence — Comprehensive Agentic Evaluation")
    print("=" * 80)
    
    total_score = 0.0
    
    for i, test in enumerate(TEST_CASES):
        print(f"\n[Test {i+1}/5] Category: {test['type'].upper()}")
        print(f"❓ Question: {test['question']}")
        
        # 1. Ask the Agent (triggers routing, tools, and streaming)
        answer = await get_agent_answer(test["question"])
        
        # 2. Print brief answer snippet
        snippet = (answer[:150] + "...") if len(answer) > 150 else answer
        print(f"🤖 Agent Response: {snippet}")
        
        # 3. Judge the answer
        result = await judge_faithfulness(test["question"], answer, test["ground_truth"])
        total_score += result["score"]
        
        score_bar = "█" * int(result["score"] * 10) + "░" * (10 - int(result["score"] * 10))
        print(f"📊 Score: [{score_bar}] {result['score']}/1.0 ({result['verdict']})")
        
    avg_score = round(total_score / len(TEST_CASES), 2)
    print("\n" + "=" * 80)
    print(f"  FINAL SYSTEM ACCURACY: {avg_score}/1.0  ({int(avg_score * 100)}% Pass Rate)")
    if avg_score >= 0.8:
        print("  ✅ PASS: The Agentic Pipeline is production-ready!")
    else:
        print("  ❌ FAIL: The Agent made reasoning or factual errors.")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_evaluation())
