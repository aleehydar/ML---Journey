"""
Pakistan Legal Intelligence — PDF Ingestion Pipeline
----------------------------------------------------
This script reads all PDF files from the data/ folder, extracts text,
chunks them with metadata, embeds them using HuggingFace, and saves
the resulting FAISS vector index to disk at vectorstore/.

USAGE:
  1. Put your PDF files inside the data/ folder.
  2. Run: python ingest.py
  3. The vectorstore/ folder is created with the persistent FAISS index.
  4. Start the app: python app.py  (it auto-loads from vectorstore/)

HOW IT WORKS:
  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
  │  PDF in   │ →  │  PyPDF Text  │ →  │  Recursive   │ →  │  FAISS     │
  │  data/    │    │  Extraction  │    │  Chunking    │    │  Persist   │
  └──────────┘    └──────────────┘    └──────────────┘    └────────────┘
"""

import os
import sys
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embedding_provider import build_embeddings

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")

def ingest():
    # Step 1: Find all PDFs
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created empty data/ folder. Put your PDF files there and re-run.")
        sys.exit(0)
    
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}/")
        print("Put your legal PDF files there and re-run this script.")
        sys.exit(0)
    
    print(f"Found {len(pdf_files)} PDF file(s):")
    for f in pdf_files:
        print(f"  📄 {f}")
    print()
    
    # Step 2: Load and extract text from each PDF
    all_docs = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        print(f"📖 Loading: {pdf_file}...")
        
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        
        import hashlib
        import re
        from datetime import datetime

        # Each page gets provenance metadata and normalized text
        for page in pages:
            # 1. Normalize OCR text
            raw_text = page.page_content
            # Replace multiple whitespace/newlines with formatted spacing
            clean_text = re.sub(r'\s+', ' ', raw_text).strip()
            page.page_content = clean_text
            
            # 2. Add Provenance Metadata
            page_text_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
            page.metadata["source"] = f"{pdf_file} - Page {page.metadata.get('page', 0) + 1}"
            page.metadata["checksum"] = page_text_hash
            page.metadata["effective_date"] = datetime.now().isoformat()
            page.metadata["source_url"] = f"local://{pdf_file}"
            page.metadata["parser_version"] = "PyMuPDFLoader-v1.0"
        
        all_docs.extend(pages)
        print(f"   Extracted {len(pages)} pages")
    
    print(f"\n📚 Total pages extracted: {len(all_docs)}")
    
    # Filter out TOC and Index pages before chunking
    filtered_docs = []
    toc_stripped = 0
    for doc in all_docs:
        text = doc.page_content
        
        # Heuristic 1: High density of dot-leaders (e.g. "........ 183")
        # In PyMuPDF, dots often get spaced out or squashed
        dot_count = text.count('.')
        
        # Heuristic 2: High ratio of standalone numbers (page numbers)
        words = text.split()
        numbers = [w for w in words if w.isdigit()]
        num_ratio = len(numbers) / len(words) if len(words) > 0 else 0
        
        # We flag a TOC if it has many dots and many numbers
        is_toc = False
        if len(words) > 20 and dot_count > 15 and num_ratio > 0.05:
            is_toc = True
        # Or if it's explicitly titled as contents and very number heavy
        elif len(words) > 10 and num_ratio > 0.15 and "contents" in text.lower()[:200]:
            is_toc = True
            
        if is_toc:
            toc_stripped += 1
            if toc_stripped <= 3:
                print(f"🗑️ STRIPPED TOC CHUNK [{doc.metadata['source']}]: {text[:100]}...")
        else:
            filtered_docs.append(doc)
            
    print(f"\n🧹 TOC/Index Filtering: Removed {toc_stripped} pages. Kept {len(filtered_docs)} pages.")

    # Step 3: Chunk the documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # Larger chunks for legal text = more context per retrieval
        chunk_overlap=150,    # Overlap prevents losing context at chunk boundaries
        separators=["\nArticle ", "\n\n", "\n", ". ", " "]  # Split at articles and paragraphs
    )
    chunks = splitter.split_documents(filtered_docs)
    print(f"✂️  Split into {len(chunks)} chunks (800 chars each, 150 overlap)")
    
    # Step 4: Embed and create FAISS index
    print("🧠 Embedding chunks with all-MiniLM-L6-v2...")
    embeddings = build_embeddings("all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Step 5: Save to disk
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"\n✅ FAISS index saved to {VECTORSTORE_DIR}/")
    print(f"   Index contains {len(chunks)} vectors")
    print(f"\n🚀 You can now start the app: python app.py")

if __name__ == "__main__":
    ingest()
