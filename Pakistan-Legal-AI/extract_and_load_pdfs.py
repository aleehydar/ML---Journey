#!/usr/bin/env python3
"""Extract text from all PDFs in /data and add to legal_texts.json"""

import os
import json
import PyPDF2

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
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except:
                    pass
            
            if text.strip():
                existing.append({
                    "source": pdf_file,
                    "text": text[:50000]  # Limit to 50k chars per PDF
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
