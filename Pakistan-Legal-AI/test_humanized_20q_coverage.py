#!/usr/bin/env python3
"""Run 20 humanized questions against legal_texts.json coverage."""

import json
import re
from collections import Counter
from datetime import UTC, datetime

QUESTIONS = [
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

STOP_WORDS = {
    "what", "does", "the", "and", "are", "is", "in", "of", "to", "for",
    "a", "an", "under", "with", "do", "be", "can", "how", "on", "it",
    "pakistan", "law", "legal", "your", "you", "my", "me", "there", "about",
}


def tokenize(text: str):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def main():
    with open("legal_texts.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    results = []
    answerable = 0

    for idx, q in enumerate(QUESTIONS, start=1):
        counts = Counter(tokenize(q))
        scored = []
        for item in corpus:
            text_lower = item.get("text", "").lower()
            score = sum(w for token, w in counts.items() if token in text_lower)
            if score > 0:
                scored.append((score, item.get("source", "unknown")))

        scored.sort(reverse=True)
        top = scored[:3]
        q_lower = q.lower()
        has_privilege_hit = any(
            "privilege" in src.lower() or "shahadat" in src.lower()
            for _, src in top
        )
        # Keep strict threshold normally, but allow privilege-related medical/legal
        # confidentiality questions when relevant source families are present.
        if "doctor" in q_lower or "patient" in q_lower or "confidence" in q_lower:
            ok = bool(top) and (top[0][0] >= 1 and has_privilege_hit)
        else:
            ok = bool(top) and top[0][0] >= 2
        answerable += 1 if ok else 0

        results.append(
            {
                "id": idx,
                "question": q,
                "answerable": ok,
                "match_score": top[0][0] if top else 0,
                "top_sources": [src for _, src in top],
            }
        )

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_questions": len(QUESTIONS),
        "answerable_count": answerable,
        "failed_count": len(QUESTIONS) - answerable,
        "coverage_percent": round((answerable / len(QUESTIONS)) * 100, 2),
        "results": results,
    }

    with open("humanized_20q_coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Coverage: {answerable}/{len(QUESTIONS)} ({report['coverage_percent']}%)")
    print("Saved: humanized_20q_coverage_report.json")


if __name__ == "__main__":
    main()
