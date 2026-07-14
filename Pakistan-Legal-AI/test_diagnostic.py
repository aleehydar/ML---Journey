import asyncio
from dotenv import load_dotenv
load_dotenv()

from retrieval_service import retrieval_service
from generation_service import generation_service

async def main():
    query = "Under the Constitution of Pakistan, what are the protections provided regarding the dignity of man and the privacy of a home?"
    print(f"\n======================================")
    print(f"1. TESTING RAW RETRIEVAL PIPELINE")
    print(f"======================================")
    
    # Trace Intent Extraction
    filters = retrieval_service._extract_intent_filters(query)
    print(f"Extracted Filters: {filters}")
    
    # Trace Retrieval
    retrieval = retrieval_service.retrieve(query, org_id="public", k=6)
    print(f"Total Chunks Retrieved: {len(retrieval.chunks)}")
    for i, c in enumerate(retrieval.chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {c.source_id}")
        print(f"Score:  {c.score:.3f}")
        print(f"Text:   {c.text[:200]}...")

    print(f"\n======================================")
    print(f"2. TESTING GENERATION PIPELINE")
    print(f"======================================")
    
    generator = generation_service.answer_legal_question(
        question=query,
        history=[],
        user_id="test_user",
        org_id="public",
        permissions=["chat:write"]
    )
    
    print("\n--- STREAMING OUTPUT (JSON EVENTS) ---")
    async for chunk in generator:
        print(chunk.strip())

if __name__ == "__main__":
    asyncio.run(main())
