from retrieval_service import retrieval_service
from dotenv import load_dotenv
load_dotenv()

res = retrieval_service.retrieve('fundamental rights of a pakistani man', 'public')
print("TOP SCORE:", res.top_score)
for i, c in enumerate(res.chunks):
    print(f"\n--- Chunk {i} [{c.source_id}] (Score: {c.score:.2f}) ---")
    print(c.text.strip())
