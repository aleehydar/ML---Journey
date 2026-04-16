#!/usr/bin/env python3
"""Per-PDF retrieval probes: 2 human questions per PDF."""

import json
import re
from datetime import UTC, datetime

from pakistan_legal_assistant import search_pakistan_law


PDF_PROBES = {
    "4th and 5th Schedule- Amended.pdf": [
        "What updates are mentioned in the 4th and 5th schedules amendment document?",
        "Does the amended 4th/5th schedule document discuss tax or corporate schedule changes?",
    ],
    "Alterations in the fourth and fifth schedules of the Companies Act, 2017.pdf": [
        "What alterations are described for the fourth and fifth schedules of Companies Act 2017?",
        "Do your documents mention changes to Companies Act 2017 schedules?",
    ],
    "Companies-Amendment-Act-2021-gazette-copy.pdf": [
        "What key changes are included in Companies Amendment Act 2021 gazette?",
        "Is there any mention of Companies Amendment Act 2021 in your legal PDFs?",
    ],
    "Constitution of Pakistan.pdf": [
        "What rights are protected by Article 9 and Article 25 of the Constitution?",
        "Does the Constitution mention equality before law and personal liberty?",
    ],
    "Draft-Notification-amendments-in-NBFC-Regulations-2008.pdf": [
        "What amendments are proposed in NBFC Regulations 2008 draft notification?",
        "Do your files include draft NBFC regulation amendment details?",
    ],
    "Financial-institutions-Act-2016-updated.pdf": [
        "What is regulated under the Financial Institutions Act 2016?",
        "Do your documents discuss financial institutions legal framework in 2016?",
    ],
    "Notification-PIC-addition-of-sugar-companies-in-3-sch_.pdf": [
        "Is there a notification about adding sugar companies in third schedule?",
        "What does the PIC sugar companies schedule notification say?",
    ],
    "PEC2016.pdf": [
        "What does PECA 2016 say about unauthorized access to personal data?",
        "Are there criminal penalties in PECA for cyber offenses?",
    ],
    "Pakistan_Penal_Code_1860_incorporating_amendments_to_16_February_2017.pdf": [
        "Does Pakistan Penal Code include punishments for criminal intimidation or cyber-linked crimes?",
        "What kinds of offenses are covered in Pakistan Penal Code amendments up to 2017?",
    ],
    "Pakistan_ang_010117.pdf": [
        "What constitutional immunity is available to President or Governor during term?",
        "What does Pakistan constitutional text say about arrest safeguards?",
    ],
    "qanun-e-shahadat-order-1984.pdf": [
        "Is an accomplice competent as a witness under Qanun-e-Shahadat?",
        "What does Qanun-e-Shahadat say about privileged communications?",
    ],
}


def extract_sources(tool_output: str):
    return set(re.findall(r"\[Source:\s*(.*?)\]", str(tool_output)))


def main():
    results = []
    for pdf_name, questions in PDF_PROBES.items():
        pdf_hits = 0
        probe_rows = []
        for q in questions:
            output = search_pakistan_law.invoke({"query": q})
            sources = extract_sources(output)
            hit = any(pdf_name.lower() in s.lower() for s in sources)
            if hit:
                pdf_hits += 1
            probe_rows.append(
                {
                    "question": q,
                    "hit": hit,
                    "retrieved_sources": sorted(sources),
                }
            )

        results.append(
            {
                "pdf": pdf_name,
                "hits": pdf_hits,
                "total_probes": len(questions),
                "pass": pdf_hits >= 1,
                "details": probe_rows,
            }
        )

    passed = sum(1 for r in results if r["pass"])
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "pdf_count": len(results),
        "pdfs_retrieved_at_least_once": passed,
        "retrieval_coverage_percent": round((passed / len(results)) * 100, 2),
        "results": results,
    }

    with open("pdf_probe_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"PDF retrieval coverage: {passed}/{len(results)} "
        f"({summary['retrieval_coverage_percent']}%)"
    )
    print("Saved: pdf_probe_report.json")


if __name__ == "__main__":
    main()
