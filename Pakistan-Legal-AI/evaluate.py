import asyncio
import json
from datetime import datetime, UTC
from generation_service import generation_service

EVAL_QUESTIONS = [
    # Legal questions
    "What are the fundamental rights of a citizen regarding arrest and detention?",
    "Can the federal government borrow money? Under what limits?",
    "What is the penalty for cyber terrorism under PECA 2016?",
    "Does the Qanun-e-Shahadat order allow accomplices as competent witnesses?",
    "What does the labor law say about termination of employment?",
    "What are the constitutional immunities of the President of Pakistan?",
    # Tax questions
    "I earn 1,500,000 PKR annually, what is my tax?",
    "What is the income tax on a salary of 3,500,000 PKR?",
    "My monthly salary is 200,000 PKR. How much tax do I pay in a year?",
    # General queries
    "Hello! How are you?",
    "What can you do?",
]

async def main():
    results = []
    total_faithfulness = 0.0
    total_relevance = 0.0
    total_recall = 0.0
    valid_evals = 0

    print("🚀 Starting Automated Evaluation Pipeline...")
    
    for i, q in enumerate(EVAL_QUESTIONS):
        print(f"\n[{i+1}/{len(EVAL_QUESTIONS)}] Query: '{q}'")
        try:
            res = await generation_service.answer_legal_question_json(q, org_id="public")
            
            if "error" in res:
                print(f"  └ INTERNAL ERROR: {res['error']}")
                
            ans = res.get("answer", "ERROR")
            evals = res.get("evaluation", {})
            sources = res.get("sources", [])
            
            f = evals.get("faithfulness", 0.0) if evals else 0.0
            r = evals.get("answer_relevance", 0.0) if evals else 0.0
            c = evals.get("context_recall", 0.0) if evals else 0.0
            
            if evals:
                total_faithfulness += f
                total_relevance += r
                total_recall += c
                valid_evals += 1
            
            results.append({
                "question": q,
                "answer": ans,
                "sources": sources,
                "metrics": {
                    "faithfulness": f,
                    "answer_relevance": r,
                    "context_recall": c
                }
            })
            print(f"  └ Answer snippet: {ans[:60]}...")
            if evals:
                print(f"  └ Faithfulness: {f:.2f} | Relevance: {r:.2f} | Recall: {c:.2f}")
            else:
                print("  └ Metrics: N/A (General chat or error)")
        except Exception as e:
            print(f"  └ ❌ Error testing: {e}")
            results.append({"question": q, "error": str(e)})

    avg_f = total_faithfulness / valid_evals if valid_evals > 0 else 0
    avg_r = total_relevance / valid_evals if valid_evals > 0 else 0
    avg_c = total_recall / valid_evals if valid_evals > 0 else 0

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_tested": len(EVAL_QUESTIONS),
        "valid_evals": valid_evals,
        "average_metrics": {
            "faithfulness": avg_f,
            "answer_relevance": avg_r,
            "context_recall": avg_c
        },
        "details": results
    }

    with open("final_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n✅ Evaluation Complete!")
    print(f"  - Avg Faithfulness:   {avg_f:.2f}")
    print(f"  - Avg Relevance:      {avg_r:.2f}")
    print(f"  - Avg Context Recall: {avg_c:.2f}")
    print("Saved report to final_evaluation_report.json")

if __name__ == "__main__":
    asyncio.run(main())
