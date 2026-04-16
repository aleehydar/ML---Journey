#!/usr/bin/env python3
"""Append missing baseline legal facts to legal_texts.json."""

import json

TARGET_FILE = "legal_texts.json"

BASELINE_ENTRIES = [
    {
        "source": "Labor Law - Minimum Wage 2024",
        "text": (
            "As of 2024, the federal minimum wage for unskilled workers in Pakistan "
            "is PKR 32,000 per month, subject to provincial notifications."
        ),
    },
    {
        "source": "Labor Law - Working Hours",
        "text": (
            "No worker should be required to work more than 9 hours per day or "
            "48 hours per week, with overtime governed by applicable labor law."
        ),
    },
    {
        "source": "Labor Law - Annual Leave",
        "text": (
            "A worker completing one year of service is entitled to at least 14 days "
            "of annual leave with full pay."
        ),
    },
    {
        "source": "Privacy Remedies - Pakistan",
        "text": (
            "For unauthorized disclosure of personal data, remedies may include filing "
            "a complaint under PECA 2016, seeking injunctions and damages in civil courts, "
            "and invoking constitutional privacy protections under Article 14."
        ),
    },
    {
        "source": "Qanun-e-Shahadat - Professional Communication Privilege",
        "text": (
            "Communications made by a client to a legal professional or other recognized "
            "professional in confidence are generally privileged and should not be disclosed "
            "without consent, except where disclosure is required by law or concerns an "
            "illegal purpose."
        ),
    },
]


def main():
    with open(TARGET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing_sources = {item.get("source") for item in data}
    added = 0

    for entry in BASELINE_ENTRIES:
        if entry["source"] not in existing_sources:
            data.append(entry)
            added += 1

    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Added {added} baseline entries.")
    print(f"Total entries: {len(data)}")


if __name__ == "__main__":
    main()
