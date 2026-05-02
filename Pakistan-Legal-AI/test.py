import asyncio
from langchain_core.messages import SystemMessage, HumanMessage
from generation_service import generation_service

async def main():
    res = await generation_service.answer_legal_question_json("What are the fundamental rights of a citizen regarding arrest and detention?")
    print("FINAL RESULT:")
    import json
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
