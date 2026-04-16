#!/usr/bin/env python3
"""
Run an extensive 25-question benchmark against the legal assistant.
The script records whether each question received a meaningful answer.
"""

import asyncio
import json
from datetime import UTC, datetime

import pakistan_legal_assistant as pla
from pakistan_legal_assistant import answer_legal_question


TEST_QUESTIONS = [
    "If someone threatens my life, does Pakistani law protect my right to life?",
    "I got arrested and nobody told me why. Is that legal in Pakistan?",
    "My employer makes me work 11 hours daily. Is this allowed?",
    "I have completed one full year at my job. How many paid leave days should I get?",
    "My salary is 700,000 yearly. Roughly how much income tax should I pay?",
    "If I earn 200,000 per month, what is my monthly tax estimate?",
    "Can my landlord throw me out of the house overnight without notice?",
    "I am worried my private chats got leaked. Which Pakistani law can help me?",
    "Can someone share my personal data after divorce without my permission?",
    "As a doctor, can I reveal what my patient told me in confidence?",
    "Can the President or Governor be criminally prosecuted during their term?",
    "Does Pakistani law treat men and women equally under the Constitution?",
    "Is an accomplice still allowed to testify in court in Pakistan?",
    "What does PECA say about unauthorized access to someone's data?",
    "If a company employee steals internal files, what legal trouble can they face?",
    "Is there anything in your documents about Companies Amendment Act 2021?",
    "Do your PDFs include NBFC regulation amendment notifications?",
    "Do your docs mention the 4th and 5th schedule amendments?",
    "What is the sales tax percentage generally mentioned in Pakistan law?",
    "If my privacy is violated, what remedies can I seek in Pakistan?",
]


REFUSAL_HINTS = [
    "cannot provide guidance",
    "i cannot",
    "i can not",
    "not in the retrieved documents",
    "currently only contains",
    "database currently only contains",
    "not found",
    "no legal documents found",
]


async def _fast_eval_stub(**kwargs):
    # Disable expensive RAGAS calls during bulk benchmark runs.
    return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_recall": 0.0}


pla.ragas_evaluator.evaluate_single = _fast_eval_stub
pla.eval_db.log_evaluation = lambda **kwargs: None


async def get_agent_answer(question: str) -> str:
    full_answer = ""
    async for event_str in answer_legal_question(question, history=[]):
        if event_str.startswith("data: "):
            try:
                payload = json.loads(event_str[6:].strip())
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "token":
                full_answer += payload.get("data", "")
    return full_answer.strip()


def is_answered(answer: str) -> bool:
    if not answer:
        return False
    lowered = answer.lower()
    if any(hint in lowered for hint in REFUSAL_HINTS):
        return False
    if len(answer) < 30:
        return False
    return True


async def run():
    print("=" * 80)
    print("Pakistan Legal AI - Humanized 20 Question Benchmark")
    print("=" * 80)

    results = []
    answered_count = 0

    for idx, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n[{idx}/{len(TEST_QUESTIONS)}] Q: {question}")
        answer = await get_agent_answer(question)
        answered = is_answered(answer)
        answered_count += 1 if answered else 0

        preview = answer[:220].replace("\n", " ")
        if len(answer) > 220:
            preview += "..."
        print(f"Status: {'PASS' if answered else 'FAIL'}")
        print(f"Answer preview: {preview if preview else '(empty)'}")

        results.append(
            {
                "question": question,
                "answered": answered,
                "answer": answer,
            }
        )

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_questions": len(TEST_QUESTIONS),
        "answered_count": answered_count,
        "failed_count": len(TEST_QUESTIONS) - answered_count,
        "answer_rate_percent": round((answered_count / len(TEST_QUESTIONS)) * 100, 2),
        "results": results,
    }

    report_file = "humanized_20q_test_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(
        f"Answered: {answered_count}/{len(TEST_QUESTIONS)} "
        f"({summary['answer_rate_percent']}%)"
    )
    print(f"Detailed report saved to: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run())
