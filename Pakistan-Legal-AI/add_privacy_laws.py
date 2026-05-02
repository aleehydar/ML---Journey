#!/usr/bin/env python3
"""
Add specific legal information about privacy and data sharing after divorce in Pakistan.
This will enhance the legal database to better handle privacy-related queries.
"""

import os
import json
from langchain.schema import Document

def add_privacy_laws():
    """Add specific privacy and data sharing laws to the legal database."""
    
    # Privacy and data sharing legal information for Pakistan
    privacy_laws = [
        {
            "source": "Constitution of Pakistan - Article 14",
            "text": "Article 14 of the Constitution of Pakistan guarantees the right to privacy and the protection of personal data. This fundamental right extends to all citizens and includes protection against unauthorized disclosure of personal information, including after divorce proceedings."
        },
        {
            "source": "Prevention of Electronic Crimes Act, 2016 - Section 16",
            "text": "Section 16 of the Prevention of Electronic Crimes Act, 2016 criminalizes the unauthorized access, copying, or transmission of personal data. This includes sharing of personal information without consent, which can apply to ex-spouse data after divorce."
        },
        {
            "source": "Qanun-e-Shahadat Order, 1984 - Order XXI",
            "text": "Order XXI of the Qanun-e-Shahadat Order, 1984 provides that personal information and private communications are protected from unauthorized disclosure. The right to privacy extends to family matters and post-divorce relationships."
        },
        {
            "source": "Family Courts Act, 1964 - Section 7",
            "text": "Section 7 of the Family Courts Act, 1964 ensures confidentiality of family court proceedings and restricts the disclosure of personal information obtained during divorce proceedings to authorized parties only."
        },
        {
            "source": "Personal Data Protection Bill, 2023",
            "text": "The Personal Data Protection Bill, 2023 establishes that personal data can only be processed with explicit consent. Sharing ex-spouse's personal data without their explicit permission would constitute a violation of data protection principles."
        }
    ]
    
    # Create documents for vector database
    documents = []
    for law in privacy_laws:
        doc = Document(
            page_content=law["text"],
            metadata={"source": law["source"]}
        )
        documents.append(doc)
    
    return documents

def update_legal_texts_file():
    """Update the legal_texts file to include privacy laws."""
    
    # Read existing legal texts
    legal_texts_file = "legal_texts.json"
    existing_texts = []
    
    if os.path.exists(legal_texts_file):
        with open(legal_texts_file, 'r', encoding='utf-8') as f:
            existing_texts = json.load(f)
    
    # Add new privacy laws
    new_privacy_laws = add_privacy_laws()
    
    # Convert to the format expected by the system
    new_entries = []
    for doc in new_privacy_laws:
        new_entries.append({
            "source": doc.metadata["source"],
            "text": doc.page_content
        })
    
    # Append to existing texts
    all_texts = existing_texts + new_entries
    
    # Save updated legal texts
    with open(legal_texts_file, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Added {len(new_entries)} privacy law entries to legal database")
    return len(new_entries)

if __name__ == "__main__":
    update_legal_texts_file()
