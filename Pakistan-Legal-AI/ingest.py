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
from langchain_community.embeddings import HuggingFaceEmbeddings

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
        
        # Each page gets metadata with source filename + page number
        for page in pages:
            page.metadata["source"] = f"{pdf_file} - Page {page.metadata.get('page', 0) + 1}"
        
        all_docs.extend(pages)
        print(f"   Extracted {len(pages)} pages")
    
    print(f"\n📚 Total pages extracted: {len(all_docs)}")
    
    # Step 3: Chunk the documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Larger chunks for legal text = more context per retrieval
        chunk_overlap=100,    # Overlap prevents losing context at chunk boundaries
        separators=["\n\n", "\n", ". ", " "]  # Split at paragraphs first, then sentences
    )
    chunks = splitter.split_documents(all_docs)
    print(f"✂️  Split into {len(chunks)} chunks (500 chars each, 100 overlap)")
    
    # Step 4: Embed and create FAISS index
    print("🧠 Embedding chunks with all-MiniLM-L6-v2...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Step 5: Save to disk
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"\n✅ FAISS index saved to {VECTORSTORE_DIR}/")
    print(f"   Index contains {len(chunks)} vectors")
    print(f"\n🚀 You can now start the app: python app.py")

if __name__ == "__main__":
    ingest()
