import asyncio
from pakistan_legal_assistant import search_pakistan_law, vectorstore, cross_encoder

query = "i cant afford my son education he is in prep"

print(f"\nQuery: {query}")
# Force hyde or normal search to find Article 25A
docs = vectorstore.similarity_search("education free compulsory Constitution", k=40)
# Find the exact doc with "25A" or "education"
target_doc = None
for d in docs:
    if "education" in d.page_content.lower() and "compulsory" in d.page_content.lower():
        target_doc = d
        break

if target_doc:
    print(f"Found target doc: {target_doc.page_content[:100]}...")
    score = cross_encoder.predict([query, target_doc.page_content])
    print(f"CrossEncoder Score: {score}")
else:
    print("Target doc not found in DB.")

# Test actual search
print("\nTesting actual tool execution:")
res = search_pakistan_law.invoke({"query": query})
print("Top returned docs length:", len(res))
