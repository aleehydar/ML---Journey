#!/usr/bin/env python3
"""Extract text from all PDFs in /data and add to legal_texts.json"""

import os
import json
from langchain_community.document_loaders import PyMuPDFLoader

data_dir = "/home/ali-haidar/Desktop/ML-Journey/Pakistan-Legal-AI/data"
output_file = "/home/ali-haidar/Desktop/ML-Journey/Pakistan-Legal-AI/legal_texts.json"

# Load existing legal texts
existing = []
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        existing = json.load(f)

print(f"📚 Existing legal texts: {len(existing)}")

# Find all PDFs
pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
print(f"📄 Found {len(pdf_files)} PDF files")

added = 0
for pdf_file in pdf_files:
    # Check if already added
    if any(item.get('source') == pdf_file for item in existing):
        print(f"⏭️  Skipping {pdf_file} (already exists)")
        continue
    
    pdf_path = os.path.join(data_dir, pdf_file)
    print(f"📖 Extracting: {pdf_file}...")
    
    try:
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        text = ""
        for page in pages:
            try:
                page_text = page.page_content
                if page_text:
                    import re
                    clean_text = re.sub(r'\s+', ' ', page_text).strip()
                    
                    # TOC Heuristic Filter
                    dot_count = clean_text.count('.')
                    words = clean_text.split()
                    numbers = [w for w in words if w.isdigit()]
                    num_ratio = len(numbers) / len(words) if len(words) > 0 else 0
                    
                    is_toc = False
                    if len(words) > 20 and dot_count > 15 and num_ratio > 0.05:
                        is_toc = True
                    elif len(words) > 10 and num_ratio > 0.15 and "contents" in clean_text.lower()[:200]:
                        is_toc = True
                        
                    if not is_toc:
                        text += clean_text + "\n\n"
            except Exception:
                pass
            
        if text.strip():
            existing.append({
                "source": pdf_file,
                "text": text.strip()  # No truncation, load full text for BM25
            })
            added += 1
            print(f"✅ Added: {pdf_file} ({len(text)} chars)")
        else:
            print(f"⚠️  No text extracted from {pdf_file}")
                
    except Exception as e:
        print(f"❌ Error with {pdf_file}: {e}")

# Save updated database
with open(output_file, 'w') as f:
    json.dump(existing, f, indent=2)

print(f"\n🎉 Complete!")
print(f"📚 Total legal documents: {len(existing)}")
print(f"📄 New PDFs added: {added}")
print(f"\n🔄 Restart server: python3 app.py")
