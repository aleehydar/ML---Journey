import os
from langchain_community.vectorstores import FAISS
from embedding_provider import build_embeddings

def main():
    vs_dir = os.path.join(os.path.dirname(__file__), "vectorstore")
    if not os.path.exists(vs_dir):
        print(f"Vectorstore not found at {vs_dir}")
        return
        
    print(f"Loading FAISS from {vs_dir}...")
    embeddings = build_embeddings("all-MiniLM-L6-v2")
    vs = FAISS.load_local(vs_dir, embeddings, allow_dangerous_deserialization=True)
    
    docstore = vs.docstore._dict
    print(f"Total documents in vectorstore: {len(docstore)}")
    
    found = 0
    for doc_id, doc in docstore.items():
        text = doc.page_content.lower()
        if "dignity of man" in text or "privacy of home" in text:
            print(f"\n--- FOUND ARTICLE 14 IN VECTORSTORE ---")
            print(f"Doc ID: {doc_id}")
            print(f"Source: {doc.metadata.get('source')}")
            print(f"Text: {doc.page_content}")
            found += 1
            
    print(f"\nTotal instances found: {found}")

if __name__ == "__main__":
    main()
