from retrieval_service import retrieval_service

def verify():
    print("\n--- Diagnostic Check 1: BM25 Corpus ---")
    found_bm25 = any('dignity of man' in doc.page_content.lower() for doc in retrieval_service.chunk_docs)
    print('BM25 corpus has real Article 14 text:', found_bm25)

    print("\n--- Diagnostic Check 2: FAISS Top-30 ---")
    dense = retrieval_service.vectorstore.similarity_search_with_score(
        'What does Article 14 of the Constitution say about privacy?', k=30)
    found_faiss = any('dignity of man' in doc.page_content.lower() for doc, score in dense)
    print('FAISS top-30 contains correct chunk:', found_faiss)

    print("\n--- Diagnostic Check 3: Final Retrieval Pipeline ---")
    result = retrieval_service.retrieve('What does Article 14 of the Constitution say about privacy?', org_id='public', k=6)
    found_full = any('dignity of man' in c.text.lower() for c in result.chunks)
    print('Final retrieval contains correct chunk:', found_full)

if __name__ == "__main__":
    verify()
