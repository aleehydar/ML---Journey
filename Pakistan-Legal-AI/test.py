import asyncio
from dotenv import load_dotenv
load_dotenv()

from generation_service import generation_service

async def main():
    q = "My annual salary is 3,500,000 PKR. Exactly how much income tax do I have to pay?"
    res = await generation_service.answer_legal_question_json(q)
    print("Tax Q:")
    print(res)
    
    q2 = "What documents and laws do you have in your database?"
    res2 = await generation_service.answer_legal_question_json(q2)
    print("DB Info Q:")
    print(res2)
    
    q3 = "Under the Constitution of Pakistan, what are the protections provided regarding the dignity of man and the privacy of a home?"
    res3 = await generation_service.answer_legal_question_json(q3)
    print("Const Q:")
    print(res3)

if __name__ == "__main__":
    asyncio.run(main())
