#!/usr/bin/env python3
"""
Evaluate the Pakistan Legal AI system with 100 distinct test queries.
Queries the running FastAPI backend directly via HTTP SSE.
"""

import json
import time
import requests
import jwt
from datetime import datetime, UTC

API_URL = "http://127.0.0.1:8000/chat"
JWT_SECRET = "your-secret-key-for-testing-12345"

TEST_QUESTIONS = [
    # Constitution (1-20)
    "Does the Constitution of Pakistan guarantee the right to life and liberty?",
    "Can the right to fair trial be denied under the Constitution of Pakistan?",
    "What are the qualifications to become the President of Pakistan?",
    "Under what circumstances can the Prime Minister of Pakistan be removed?",
    "Who appoints the Chief Justice of Pakistan?",
    "What is the retirement age for a Supreme Court Judge in Pakistan?",
    "Does the Constitution allow discrimination on the basis of sex in public employment?",
    "What is the official language of Pakistan according to the Constitution?",
    "What is the maximum duration for an emergency to remain in force without parliamentary approval?",
    "Can a non-Muslim become the President of Pakistan?",
    "What is the procedure to amend the Constitution of Pakistan?",
    "Can the National Assembly be dissolved prematurely? If yes, by whom?",
    "What are the fundamental rights given to citizens in Pakistan?",
    "Is primary education mandatory and free in Pakistan?",
    "Under what Article is freedom of speech guaranteed?",
    "Can double jeopardy apply in criminal prosecution according to the Constitution?",
    "Does the Constitution provide safeguards against illegal arrest?",
    "What is the jurisdiction of the Federal Shariat Court?",
    "How are the seats in the Senate distributed among provinces?",
    "Can fundamental rights be suspended during a state of emergency?",

    # Qanun-e-Shahadat / Evidence (21-40)
    "How many witnesses are required to prove an obligation according to Qanun-e-Shahadat?",
    "Is the testimony of an accomplice admissible in court?",
    "When is a confession made to a police officer admissible?",
    "Does Qanun-e-Shahadat apply to proceedings before an arbitrator?",
    "Can a child testify in a Pakistani court?",
    "What is privileged communication under the Evidence Act?",
    "Can a lawyer disclose a client's confidential information?",
    "What is the evidential value of an expert opinion in Pakistan?",
    "Who bears the burden of proof in a criminal case?",
    "Can a husband or wife be compelled to disclose communication made during marriage?",
    "What constitutes documentary evidence in Qanun-e-Shahadat?",
    "Is electronic evidence admissible in court in Pakistan?",
    "What is the presumption regarding a document older than 30 years?",
    "Can a person be forced to answer a question that might incriminate them?",
    "What is primary evidence versus secondary evidence?",
    "In financial matters, does the law prefer oral or written evidence?",
    "Is a confession by an accused person made under threat valid?",
    "How is the credibility of a witness established or challenged?",
    "Can an idiot or lunatic testify if they can understand the questions?",
    "What happens if part of a confession is exculpatory and part is inculpatory?",

    # Tax Law (41-60)
    "What is the income tax exemption limit for a salaried person in Pakistan for 2023-2024?",
    "How much tax will I pay if my annual salary is 1,500,000 PKR?",
    "What is the difference between tax rates for a filer and a non-filer?",
    "Is income from agriculture taxable by the Federal Government?",
    "What is the standard rate of Sales Tax in Pakistan?",
    "Are exports subject to sales tax?",
    "What is the penalty for late filing of income tax returns?",
    "How is capital gains tax calculated on the sale of property?",
    "Are IT exports exempt from income tax in Pakistan?",
    "What is withholding tax?",
    "Do I have to pay tax on foreign remittances received through standard banking channels?",
    "Can I claim tax credits for charitable donations?",
    "How long do I need to keep financial records for tax purposes?",
    "Is provident fund withdrawal taxable in Pakistan?",
    "What is the tax rate on dividend income?",
    "Are pensions taxable in Pakistan?",
    "What is Advance Tax?",
    "Can the Federal Board of Revenue freeze bank accounts?",
    "What is the sales tax rate on telecom services?",
    "Is Zakat deducted at source allowed as a tax deduction?",

    # Labor / Employment Law (61-80)
    "What is the maximum number of working hours in a week according to Pakistani labor laws?",
    "How much overtime pay is a worker legally entitled to?",
    "How many days of annual paid leave are mandated?",
    "Are female employees entitled to maternity leave?",
    "How many days of maternity leave are provided by law?",
    "Can an employer terminate a worker without giving notice?",
    "What is the mandatory notice period before termination?",
    "Is a company required to pay a gratuity to an employee upon resignation?",
    "What is the minimum wage for unskilled workers in Pakistan currently?",
    "Can children under 14 be legally employed in factories?",
    "Are workers entitled to sick leave with full pay?",
    "How many casual leaves are allowed per year?",
    "Can a worker form or join a trade union?",
    "What happens if an employee is injured on duty?",
    "Is there a legal requirement to provide a written employment contract?",
    "Can female workers be compelled to work night shifts?",
    "What constitutes unfair labor practices by an employer?",
    "Are bonuses mandatory under the law?",
    "Can a fixed-term employee be fired before the contract expires?",
    "Is group insurance mandatory for industrial workers?",

    # Cyber / PECA (81-100)
    "What is the punishment for unauthorized access to an information system under PECA 2016?",
    "How does PECA define cyber terrorism?",
    "Is online defamation a criminal offense in Pakistan?",
    "What are the penalties for hate speech on the internet?",
    "Can a person be punished for sharing someone's private pictures without consent?",
    "What authority is responsible for investigating cybercrimes in Pakistan?",
    "Can the PTA block websites?",
    "What is the punishment for electronic forgery?",
    "How does the law deal with cyber stalking?",
    "Are internet service providers liable for the content accessed through their networks?",
    "Can law enforcement search and seize computer systems without a warrant?",
    "What is the maximum penalty for child pornography offenses under PECA?",
    "Is spoofing an offense under Pakistani cyber law?",
    "Does PECA apply to crimes committed against Pakistani citizens by people outside Pakistan?",
    "How is 'critical information infrastructure' protected?",
    "What remedies exist if someone creates a fake social media profile of me?",
    "Can electronic evidence collected by FIA be used in a normal criminal court?",
    "What is the punishment for transferring funds electronically through fraud?",
    "Do telecom companies have to retain subscriber data?",
    "What are the limits of freedom of expression under the bounds of PECA?"
]

REFUSAL_HINTS = [
    "cannot provide guidance",
    "i cannot",
    "i can not",
    "not in the retrieved documents",
    "currently only contains",
    "database currently only contains",
    "not found",
    "no legal documents found"
]

def generate_token():
    payload = {
        "sub": "tester-100",
        "org_id": "test-org",
        "permissions": ["chat:write", "docs:read", "eval:read"]
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def main():
    print(f"Starting 100-query evaluation at {datetime.now(UTC).isoformat()}")
    token = generate_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    results = []
    answered_count = 0
    total = len(TEST_QUESTIONS)

    # Let's pause a bit between requests to avoid overloading the LLM API provider
    # Adjust this delay based on API limits (1.5 seconds is usually safe for medium-tier)
    delay_between_requests = 1.5 

    for idx, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{idx}/{total}] Q: {question}")
        payload = {"question": question, "history": []}
        
        full_answer = ""
        try:
            # Setting stream=True as the endpoint uses StreamingResponse
            with requests.post(API_URL, json=payload, headers=headers, stream=True, timeout=30) as response:
                if response.status_code != 200:
                    print(f"  -> Error: {response.status_code} {response.text}")
                    full_answer = f"[ERROR: {response.status_code}]"
                else:
                    for line in response.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            if decoded.startswith('data: '):
                                try:
                                    data_json = json.loads(decoded[6:])
                                    if data_json.get('type') == 'token':
                                        full_answer += data_json.get('data', '')
                                except json.JSONDecodeError:
                                    pass
        except Exception as e:
            print(f"  -> Exception: {e}")
            full_answer = f"[EXCEPTION: {e}]"

        full_answer = full_answer.strip()
        
        # Eval logic
        lowered = full_answer.lower()
        answered = True
        if not full_answer or full_answer.startswith("[ERROR") or full_answer.startswith("[EXCEPTION"):
            answered = False
        elif any(hint in lowered for hint in REFUSAL_HINTS):
            answered = False
        elif len(full_answer) < 30:
            answered = False

        if answered:
            answered_count += 1
            print("  -> PASS")
        else:
            print(f"  -> FAIL (Reason snippet: {full_answer[:80]})")
        
        results.append({
            "id": idx,
            "question": question,
            "answered": answered,
            "answer_preview": full_answer[:200] + "..." if len(full_answer) > 200 else full_answer
        })

        time.sleep(delay_between_requests)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_questions": total,
        "answered_count": answered_count,
        "failed_count": total - answered_count,
        "success_rate_percent": round((answered_count / total) * 100, 2),
        "results": results
    }

    report_file = "100_cases_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("-" * 50)
    print("RESULTS:")
    print(f"Total Questions: {total}")
    print(f"Answered Successfully: {answered_count}")
    print(f"Success Rate: {summary['success_rate_percent']}%")
    print(f"Full report saved to {report_file}")
    print("-" * 50)

if __name__ == "__main__":
    main()
