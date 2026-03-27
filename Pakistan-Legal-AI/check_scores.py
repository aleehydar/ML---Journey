import asyncio
from pakistan_legal_assistant import search_pakistan_law, vectorstore, cross_encoder

queries = [
    # Highly relevant
    "Who is the competent witness in a criminal case?",
    # Completely irrelevant (out of database scope)
    "I am living in attock but i dont have my house certificate, where can i found it",
    "What is the penalty for stealing a car in Lahore?"
]

for q in queries:
    print(f"\nQuery: {q}")
    docs = vectorstore.similarity_search(q, k=10)
    pairs = [[q, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    
    for score, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:3]:
        print(f"Score: {score:.2f} | Source: {doc.metadata['source']}")
