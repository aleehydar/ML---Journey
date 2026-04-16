#!/usr/bin/env python3
"""
Fast 25-question coverage test against legal_texts.json.
Determines if each question is answerable from ingested PDF text.
"""

import json
import re
from collections import Counter
from datetime import UTC, datetime


QUESTIONS = [
    "What does Article 9 of the Constitution of Pakistan protect?",
    "What does Article 10 say about arrest safeguards?",
    "What rights are guaranteed under Article 25?",
    "What immunity does Article 248 provide to the President?",
    "Is an accomplice a competent witness?",
    "When is professional communication privileged under Qanun-e-Shahadat?",
    "Can a landlord evict a tenant immediately?",
    "What is the minimum wage in Pakistan in 2024?",
    "What are maximum daily and weekly working hours?",
    "How many annual leave days are granted after one year of service?",
    "What is the sales tax rate in Pakistan?",
    "How is tax computed for annual income of PKR 700,000?",
    "What penalties apply for unauthorized access under PECA 2016?",
    "Is sharing ex-spouse personal data without consent lawful?",
    "What confidentiality protections exist in Family Courts Act Section 7?",
    "What does the Companies Amendment Act 2021 change?",
    "What is covered in 4th and 5th Schedule amendments?",
    "What does the Financial Institutions Act 2016 regulate?",
    "Does Pakistan Penal Code address cyber-related crimes?",
    "What does PECA Section 16 criminalize?",
    "What are legal remedies for privacy breaches in Pakistan?",
    "What does Constitution Article 14 say about privacy?",
    "Is unauthorized copying of corporate data punishable?",
    "Do the uploaded PDFs include NBFC Regulation amendments?",
    "Do the uploaded PDFs include sugar companies notification changes?",
]


STOP_WORDS = {
    "what", "does", "the", "and", "are", "is", "in", "of", "to", "for",
    "a", "an", "under", "with", "do", "be", "can", "how", "on", "it",
    "pakistan", "law", "legal", "uploaded",
}


def tokenize(text: str):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def run():
    with open("legal_texts.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    results = []
    answered = 0

    for idx, q in enumerate(QUESTIONS, start=1):
        q_tokens = tokenize(q)
        q_counts = Counter(q_tokens)

        scored = []
        for item in corpus:
            text = item.get("text", "")
            text_lower = text.lower()
            score = sum(weight for token, weight in q_counts.items() if token in text_lower)
            if score > 0:
                scored.append((score, item.get("source", "unknown")))

        scored.sort(reverse=True)
        top_hits = scored[:3]
        can_answer = len(top_hits) > 0 and top_hits[0][0] >= 2
        if can_answer:
            answered += 1

        results.append(
            {
                "id": idx,
                "question": q,
                "answerable": can_answer,
                "top_sources": [src for _, src in top_hits],
                "match_score": top_hits[0][0] if top_hits else 0,
            }
        )

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_questions": len(QUESTIONS),
        "answerable_count": answered,
        "not_answerable_count": len(QUESTIONS) - answered,
        "coverage_percent": round((answered / len(QUESTIONS)) * 100, 2),
        "results": results,
    }

    with open("pdf_coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Coverage: {answered}/{len(QUESTIONS)} ({report['coverage_percent']}%)")
    print("Saved: pdf_coverage_report.json")


if __name__ == "__main__":
    run()
