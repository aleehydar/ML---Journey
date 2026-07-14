import asyncio
import os
import pytest
from evaluate import EVAL_QUESTIONS
from generation_service import generation_service

async def run_eval_batch(use_hyde: bool):
    os.environ["USE_HYDE"] = "true" if use_hyde else "false"
    
    total_recall = 0.0
    valid_evals = 0
    
    print(f"\n🚀 Running Eval with USE_HYDE={use_hyde}...")
    for i, q in enumerate(EVAL_QUESTIONS):
        print(f"[{i+1}/{len(EVAL_QUESTIONS)}] {q}")
        try:
            res = await generation_service.answer_legal_question_json(q, org_id="public")
            evals = res.get("evaluation", {})
            if evals and "context_recall" in evals:
                total_recall += evals["context_recall"]
                valid_evals += 1
        except Exception as e:
            print(f"  └ ❌ Error: {e}")
            
    avg_recall = total_recall / valid_evals if valid_evals > 0 else 0
    print(f"✅ USE_HYDE={use_hyde} | Avg Context Recall: {avg_recall:.2f} (over {valid_evals} valid questions)")
    return avg_recall

@pytest.mark.asyncio
async def test_hyde_improvement():
    recall_without = await run_eval_batch(use_hyde=False)
    recall_with = await run_eval_batch(use_hyde=True)
    
    print(f"\n📊 RESULTS")
    print(f"Without HyDE: {recall_without:.2f}")
    print(f"With HyDE:    {recall_with:.2f}")
    print(f"Difference:   {recall_with - recall_without:+.2f}")
    
if __name__ == "__main__":
    asyncio.run(test_hyde_improvement())
