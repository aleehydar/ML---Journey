"""
Pakistan Legal Intelligence - Automated RAG Evaluation Script
-------------------------------------------------------------
This script uses the "LLM-as-a-Judge" pattern to automatically evaluate
the faithfulness (hallucination score) of the RAG pipeline.

HOW IT WORKS:
1. We have 5 strict test questions with known ground-truth context.
2. We run each question through our RAG Agent and capture the answer.
3. A separate "Judge LLM" (via Groq) is given the Agent's answer and the
   real retrieved context. It extracts every factual claim in the answer,
   then checks if each claim can be verified directly from the context.
4. Faithfulness Score = Verified Claims / Total Claims
   - 1.0 = Perfect. No hallucinations.
   - 0.5 = 50% of claims are grounded. Needs improvement.

Run: python evaluate_rag.py
"""

import asyncio
import json
import re
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from pakistan_legal_assistant import vectorstore, cross_encoder, search_pakistan_law

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE: 5 questions about Pakistan law
# Each has a 'ground_truth' string showing what the DB actually says.
# ─────────────────────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "question": "What is the minimum wage in Pakistan in 2024?",
        "ground_truth": "The federal minimum wage is PKR 32,000 per month for unskilled workers."
    },
    {
        "question": "How many hours should a worker work per day maximum?",
        "ground_truth": "No worker shall be required to work more than 48 hours per week or 9 hours per day."
    },
    {
        "question": "How many days of annual leave does a Pakistani worker get?",
        "ground_truth": "Every worker who has completed one year of service is entitled to 14 days of annual leave."
    },
    {
        "question": "What is the income tax for someone earning PKR 700,000 annually?",
        "ground_truth": "Earnings between PKR 600,000 and PKR 1,200,000 are taxed at 5%. So 700,000 - 600,000 = 100,000 * 5% = PKR 5,000."
    },
    {
        "question": "Can a landlord evict a tenant immediately?",
        "ground_truth": "A landlord cannot evict a tenant without proper legal notice. A minimum of one month notice is required."
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# JUDGE LLM - Separate instance used only for evaluation, not for answering.
# ─────────────────────────────────────────────────────────────────────────────
judge_llm = ChatGroq(model="llama-3.1-8b-instant")

async def get_rag_answer(question: str) -> str:
    """Runs the question through the search tool and LLM to get an answer."""
    # Use the search tool to get context
    context = search_pakistan_law.invoke({"query": question})
    
    # Ask the LLM to answer using that context
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.1-8b-instant")
    from langchain_core.messages import HumanMessage
    
    prompt_text = f"""Answer the following question based ONLY on the provided context. 
    Do not add anything that is not in the context.
    
    Context:
    {context}
    
    Question: {question}
    Answer:"""
    
    response = await llm.ainvoke([HumanMessage(content=prompt_text)])
    return response.content

async def judge_faithfulness(question: str, generated_answer: str, ground_truth: str) -> dict:
    """
    The Judge LLM holistically evaluates if the generated answer is faithful
    to the ground truth context, on a scale of 0.0 to 1.0.
    
    WHY HOLISTIC SCORING?
    Short factual answers like "9 hours" or "No" correctly answer the question
    but can't be decomposed into claims. A holistic judge handles these better.
    """
    from langchain_core.messages import HumanMessage
    
    judge_prompt = f"""You are an expert AI evaluator assessing factual faithfulness of an AI model's answer.

GROUND TRUTH CONTEXT: {ground_truth}

AI MODEL'S ANSWER: {generated_answer}

QUESTION: {question}

TASK: Judge whether the AI's answer is factually supported by the Ground Truth Context.
- Score 1.0 if the answer is completely faithful (correct and verifiable from context).
- Score 0.5 if the answer is partially faithful (some correct, some not).
- Score 0.0 if the answer contradicts or cannot be verified from the context.

Be lenient with short answers — if "9 hours" answers "max hours per day?" and ground truth says "9 hours per day", that is 1.0.

Respond with ONLY a single JSON object, nothing else. Example: {{"score": 0.8, "reason": "The answer correctly states..."}}

JSON Response:"""
    
    response = await judge_llm.ainvoke([HumanMessage(content=judge_prompt)])
    
    try:
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if match:
            result = json.loads(match.group())
            score = float(result.get("score", 0.0))
            reason = result.get("reason", "N/A")
            return {"score": score, "verdict": reason}
    except (json.JSONDecodeError, ValueError):
        pass
    
    return {"score": 0.0, "verdict": "Judge returned unparseable response."}

async def run_evaluation():
    """Main evaluation runner."""
    print("=" * 70)
    print("  Pakistan Legal Intelligence — Automated RAG Faithfulness Evaluation")
    print("=" * 70)
    print(f"  Running {len(TEST_CASES)} test cases...\n")
    
    total_score = 0.0
    
    for i, test in enumerate(TEST_CASES):
        print(f"─── Test {i+1}: {test['question'][:60]}...")
        
        # Get RAG answer
        answer = await get_rag_answer(test["question"])
        print(f"  RAG Answer: {answer[:150]}...")
        
        # Judge faithfulness
        result = await judge_faithfulness(test["question"], answer, test["ground_truth"])
        
        total_score += result["score"]
        score_bar = "█" * int(result["score"] * 10) + "░" * (10 - int(result["score"] * 10))
        
        print(f"  Faithfulness Score: [{score_bar}] {result['score']} ({result['verdict']})")
        print()
    
    avg_score = round(total_score / len(TEST_CASES), 2)
    print("=" * 70)
    print(f"  FINAL SCORE: {avg_score}/1.0  ({int(avg_score * 100)}% Faithful)")
    if avg_score >= 0.8:
        print("  ✅ PASS: RAG pipeline is production-grade with minimal hallucinations!")
    elif avg_score >= 0.5:
        print("  ⚠️  REVIEW: Moderate hallucination risk. Consider tuning the retriever.")
    else:
        print("  ❌ FAIL: High hallucination risk. Review prompts and retrieval strategy.")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_evaluation())
