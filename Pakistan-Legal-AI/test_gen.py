import asyncio
from generation_service import GenerationService, _validate_grounding

async def main():
    s = GenerationService()
    # Let's mock a query to seeing what Llama 3 generates
    context = "[Source: doc1]\nSome text about rights"
    ans = await s._generate_legal_answer("fundamental rights of a pakistani man", context)
    print("LLM RAW ANSWER:")
    print(ans)
    print("VALIDATION RESULT: ", _validate_grounding(ans))

if __name__ == "__main__":
    asyncio.run(main())
