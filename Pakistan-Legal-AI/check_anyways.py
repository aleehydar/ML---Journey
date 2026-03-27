import asyncio
from pakistan_legal_assistant import search_pakistan_law, vectorstore, cross_encoder

query = "anyways i have a question"

print(f"\nQuery: {query}")
docs = vectorstore.similarity_search(query, k=10)
pairs = [[query, doc.page_content] for doc in docs]
scores = cross_encoder.predict(pairs)

for score, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:3]:
    print(f"Score: {score:.2f} | Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:150]}...\n")
