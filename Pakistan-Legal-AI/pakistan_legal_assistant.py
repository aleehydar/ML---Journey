import os
import json
import asyncio
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

from langchain_core.tools import tool
from evaluation_db import eval_db
from ragas_evaluator import ragas_evaluator

load_dotenv()

# Preload CrossEncoder for reranking
print("Loading CrossEncoder...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─── VECTORSTORE: Load from legal_texts.json (enhanced database with PDFs) ────
LEGAL_TEXTS_FILE = os.path.join(os.path.dirname(__file__), "legal_texts.json")

if os.path.exists(LEGAL_TEXTS_FILE):
    print(f"📂 Loading enhanced legal database from {LEGAL_TEXTS_FILE}...")
    with open(LEGAL_TEXTS_FILE, 'r', encoding='utf-8') as f:
        legal_texts = json.load(f)
    print(f"✅ Loaded {len(legal_texts)} legal documents from database!")
    
    # Build vectorstore from legal_texts.json
    docs = [Document(page_content=item["text"], metadata={"source": item["source"]}) for item in legal_texts]
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"🔍 Built search index with {len(chunks)} document chunks")
else:
    print("⚠️  No legal_texts.json found. Using hardcoded fallback data.")
    # Fallback: Hardcoded Pakistani Legal Documents
    legal_texts = [
        {"source": "Constitution of Pakistan - Article 9", "text": "Security of Person: No person shall be deprived of life or liberty save in accordance with law."},
        {"source": "Constitution of Pakistan - Article 10", "text": "Safeguards as to arrest and detention: No person who is arrested shall be detained in custody without being informed of the grounds for such arrest, nor shall he be denied the right to consult and be defended by a legal practitioner of his choice."},
        {"source": "Constitution of Pakistan - Article 25", "text": "Equality of citizens: All citizens are equal before law and are entitled to equal protection of law. There shall be no discrimination on the basis of sex alone."},
        {"source": "Constitution of Pakistan - Article 37", "text": "Promotion of social justice: The State shall secure the well-being of the people, irrespective of sex, caste, creed or race."},
        {"source": "Labor Law - Minimum Wage 2024", "text": "The minimum wage in Pakistan is set by provincial governments. As of 2024, the federal minimum wage is PKR 32,000 per month for unskilled workers."},
        {"source": "Labor Law - Working Hours", "text": "No worker shall be required to work more than 48 hours per week or 9 hours per day. Overtime must be paid at double the regular rate."},
        {"source": "Labor Law - Annual Leave", "text": "Every worker who has completed one year of service is entitled to annual leave with full pay for a period not less than 14 days."},
        {"source": "Tax Law - Income Tax", "text": "Pakistan uses a MARGINAL tax system. Individuals earning up to PKR 600,000 annually are exempt (0% tax). For earnings between PKR 600,000 and PKR 1,200,000, ONLY the amount EXCEEDING PKR 600,000 is taxed at 5%. Example: PKR 700,000 income → tax = 5% × (700,000 − 600,000) = PKR 5,000."},
        {"source": "Tax Law - Sales Tax", "text": "The standard rate of sales tax in Pakistan is 17% on the value of taxable supplies made in Pakistan."},
        {"source": "Tenant Rights", "text": "A landlord cannot evict a tenant without proper legal notice. A minimum of one month notice is required before eviction proceedings can begin."},
        {"source": "Constitution of Pakistan - Article 14", "text": "Article 14 of Constitution of Pakistan guarantees the right to privacy and protection of personal data. This fundamental right extends to all citizens and includes protection against unauthorized disclosure of personal information, including after divorce proceedings. Sharing personal data of an ex-spouse without their explicit consent would violate this constitutional right."},
        {"source": "Prevention of Electronic Crimes Act, 2016 - Section 16", "text": "Section 16 of the Prevention of Electronic Crimes Act, 2016 criminalizes unauthorized access, copying, or transmission of personal data. This includes sharing of personal information without consent, which can apply to ex-spouse data after divorce. Violations can result in imprisonment and fines."},
        {"source": "Qanun-e-Shahadat Order, 1984 - Order XXI", "text": "Order XXI of the Qanun-e-Shahadat Order, 1984 provides that personal information and private communications are protected from unauthorized disclosure. The right to privacy extends to family matters and post-divorce relationships. Any attempt to share personal data without consent may be considered contempt of court."},
        {"source": "Family Courts Act, 1964 - Section 7", "text": "Section 7 of the Family Courts Act, 1964 ensures confidentiality of family court proceedings and restricts disclosure of personal information obtained during divorce proceedings to authorized parties only. Sharing ex-spouse's personal data with third parties without their permission violates this provision."},
        {"source": "Personal Data Protection Bill, 2023", "text": "The Personal Data Protection Bill, 2023 establishes that personal data can only be processed with explicit consent. Sharing ex-spouse's personal data without their explicit permission would constitute a violation of data protection principles and may result in legal penalties."}
    ]
    docs = [Document(page_content=item["text"], metadata={"source": item["source"]}) for item in legal_texts]
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatGroq(model="llama-3.1-8b-instant")

@tool
def search_pakistan_law(query: str) -> str:
    """Always use this tool to search the Pakistan legal database (Constitution, Qanun-e-Shahadat, labor, etc.).
    Pass the user's question directly to this tool."""
    
    # Enhanced search strategy: Try both direct query and HyDE for better coverage
    search_queries = []
    
    # 1. Add original query
    search_queries.append(query.lower())
    
    # 2. Add key terms for privacy/data sharing queries
    if any(term in query.lower() for term in ['share', 'data', 'privacy', 'consent', 'divorce', 'ex-wife', 'personal information']):
        search_queries.extend([
            'privacy data sharing divorce',
            'personal information consent',
            'data protection divorce',
            'family law privacy rights',
            'qanun privacy data',
            'divorce privacy laws'
        ])
    
    # 3. Add specific legal terms
    if 'wife' in query.lower() or 'spouse' in query.lower():
        search_queries.extend([
            'spouse privacy rights',
            'family law data protection',
            'marriage privacy laws',
            'divorce confidentiality'
        ])
    
    # 4. Add key terms for corporate data theft/unauthorized copying
    if any(term in query.lower() for term in ['company', 'corporate', 'copying', 'data theft', 'unauthorized', 'business', 'employer']):
        search_queries.extend([
            'unauthorized access computer data',
            'electronic crimes company data',
            'data theft corporate Pakistan',
            'prevention electronic crimes act',
            'unauthorized copying information',
            'company data protection',
            'corporate espionage',
            'computer misuse',
            'electronic fraud',
            'cyber crime business'
        ])
    
    # 5. Add key terms for criminal/cyber crime queries
    if any(term in query.lower() for term in ['crime', 'criminal', 'punishment', 'imprisonment', 'fine', 'offense', 'hacking']):
        search_queries.extend([
            'prevention electronic crimes act 2016',
            'pakistan penal code cyber crime',
            'unauthorized access punishment',
            'electronic fraud penalties',
            'cyber crime imprisonment'
        ])
    
    # 6. HyDE (Hypothetical Document Embeddings) for semantic expansion
    # Customize prompt based on query type
    if any(term in query.lower() for term in ['company', 'corporate', 'data', 'unauthorized', 'copying', 'theft']):
        hyde_prompt = f"Write a hypothetical, formal excerpt from Pakistani law (Prevention of Electronic Crimes Act 2016 or Pakistan Penal Code) that addresses this query: '{query}'. Use strict legal terminology about unauthorized access, data theft, corporate information, electronic crimes, computer systems, and penalties including imprisonment and fines. Do not include any introductory text, just the hypothetical legal text."
    elif any(term in query.lower() for term in ['crime', 'punishment', 'imprisonment']):
        hyde_prompt = f"Write a hypothetical, formal excerpt from Pakistani criminal law (Pakistan Penal Code or Prevention of Electronic Crimes Act 2016) that addresses this query: '{query}'. Use strict legal terminology about offenses, penalties, imprisonment terms, fines, and legal consequences. Do not include any introductory text, just the hypothetical legal text."
    else:
        hyde_prompt = f"Write a hypothetical, formal excerpt from a Pakistani law code that perfectly answers this query: '{query}'. Use strict legal terminology (e.g., 'communication during marriage', 'disclosure', 'privilege', 'privacy', 'consent', 'data protection'). Do not include any introductory text, just the hypothetical legal text."
    
    hypothetical_doc = llm.invoke(hyde_prompt).content
    search_queries.append(hypothetical_doc)
    
    # 5. Search with multiple queries and combine results
    all_docs = []
    for search_query in search_queries:
        docs = vectorstore.similarity_search(search_query, k=20)
        all_docs.extend(docs)
    
    # Remove duplicates while preserving order
    seen_sources = set()
    unique_docs = []
    for doc in all_docs:
        source = doc.metadata['source']
        if source not in seen_sources:
            seen_sources.add(source)
            unique_docs.append(doc)
    
    # 6. Re-rank with cross-encoder
    pairs = [[query, doc.page_content] for doc in unique_docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)

    # 6.1 Query-aware source balancing to avoid repeated over-dominance
    # from broad documents when user asks about specific corporate/finance PDFs.
    query_lower = query.lower()

    source_keyword_boosts = {
        "companies-amendment-act-2021-gazette-copy.pdf": [
            "companies amendment",
            "companies act",
            "2021",
            "gazette",
            "company law",
        ],
        "alterations in the fourth and fifth schedules of the companies act, 2017.pdf": [
            "fourth schedule",
            "fifth schedule",
            "companies act 2017",
            "alterations",
        ],
        "notification-pic-addition-of-sugar-companies-in-3-sch_.pdf": [
            "sugar companies",
            "pic",
            "third schedule",
            "notification",
        ],
        "draft-notification-amendments-in-nbfc-regulations-2008.pdf": [
            "nbfc",
            "regulations 2008",
            "non-banking",
            "draft notification",
        ],
        "financial-institutions-act-2016-updated.pdf": [
            "financial institutions",
            "fia 2016",
            "banking",
        ],
    }

    over_broad_sources = {
        "qanun-e-shahadat-order-1984.pdf",
        "constitution of pakistan.pdf",
    }

    adjusted_scored_docs = []
    for base_score, doc in scored_docs:
        source = str(doc.metadata.get("source", "")).lower()
        adjusted = float(base_score)

        # Boost specific PDF sources when query intent clearly matches them.
        for target_source, keywords in source_keyword_boosts.items():
            if target_source in source and any(k in query_lower for k in keywords):
                adjusted += 2.0
                break

        # Slightly damp broad sources for corporate/notification style intents.
        if any(
            k in query_lower
            for k in [
                "companies",
                "schedule",
                "nbfc",
                "financial institution",
                "notification",
                "gazette",
                "sugar companies",
            ]
        ) and any(b in source for b in over_broad_sources):
            adjusted -= 1.2

        adjusted_scored_docs.append((adjusted, doc))

    scored_docs = sorted(adjusted_scored_docs, key=lambda x: x[0], reverse=True)
    
    # Relevance Threshold: Filter out anything with a score < -5.0
    relevant_scored = [doc for score, doc in scored_docs if score > -5.0]
    top_docs = relevant_scored[:4]
    
    context_parts = []
    contexts_for_eval = []  # Store plain contexts for evaluation
    
    for doc in top_docs:
        context_part = f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
        context_parts.append(context_part)
        contexts_for_eval.append(doc.page_content)  # Plain text for evaluation
        
    # Store contexts globally for evaluation (will be cleared after each query)
    global _current_contexts
    _current_contexts = contexts_for_eval
        
    return "\n\n".join(context_parts) if context_parts else "No legal documents found. Please reply with a natural, conversational greeting and politely ask the user to state their specific legal question. Do not mention that a search failed or that data was not found."

@tool
def calculate_tax(annual_income: float) -> str:
    """Use this tool to precisely calculate ANNUAL income tax for salaried individuals based on the 2023-2024 Pakistani Tax brackets.
    You MUST pass the ANNUAL income to this tool, not monthly."""
    try:
        income = float(annual_income)
        if income <= 600000:
            tax = 0.0
        elif income <= 1200000:
            tax = (income - 600000) * 0.025
        elif income <= 2400000:
            tax = 15000 + ((income - 1200000) * 0.125)
        elif income <= 3200000:
            tax = 165000 + ((income - 2400000) * 0.225)
        elif income <= 4100000:
            tax = 345000 + ((income - 3200000) * 0.275)
        else:
            tax = 592500 + ((income - 4100000) * 0.35)
            
        return f"For an annual income of {income:,.2f} PKR, the calculated ANNUAL tax is {tax:,.2f} PKR."
    except Exception as e:
        return "Error calculating tax. Please provide a valid numerical annual income."

tools = [search_pakistan_law, calculate_tax]

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an elite, highly professional Pakistani legal advisor. 
    You must intelligently use your tools to load legal documents and analyze them. 
    CRITICAL INSTRUCTIONS: 
    1. STRICT GROUNDING: You MUST answer the user's question using ONLY the facts and laws provided by your tools. Do NOT use your internal training data.
    2. MISSING LAWS: If the user asks about a specific law that is NOT in the retrieved documents, or if the tool returns no results, you MUST politely state that your database currently only contains specific laws (like the Constitution and Qanun-e-Shahadat) and that you cannot provide guidance outside of them.
    3. EXTRAPOLATION: If the retrieved laws are conceptually related, you may confidently extrapolate standard legal advice based on them. Act like a lawyer building a case.
    4. CITATIONS: If you are providing legal advice based on retrieved documents, you MUST cite the EXACT filename/page [Source: ...] naturally in your text. However, if you are politely declining to answer or making a conversational response, DO NOT invent, hallucinate, or append any [Source: ...] tags.
    5. STYLE: Summarize the retrieved laws in your OWN words — do NOT copy-paste or quote entire legal sections verbatim.
    6. Keep your answers concise: 2-4 short paragraphs of advisory guidance, not walls of legal text."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

llm_with_tools = llm.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}

# Global variable to store current contexts for evaluation
_current_contexts = []

async def answer_legal_question(question: str, history: list = None, user_id: str = None, org_id: str = None):
    # Convert dict history to LangChain message objects
    chat_history = []
    if history:
        for msg in history:
            if msg.get("role") == "user":
                chat_history.append(HumanMessage(content=msg.get("content")))
            else:
                chat_history.append(AIMessage(content=msg.get("content")))
                
    yield f"data: {json.dumps({'type': 'trace', 'data': '🤖 Initializing Cognitive Agent Router...'})}\n\n"
    await asyncio.sleep(0.3)

    # Variables to capture answer for evaluation
    full_answer = ""
    used_search_tool = False
    
    # Declare global contexts variable
    global _current_contexts

    try:
        # Step 1: Format the messages
        messages = prompt.format_messages(question=question, chat_history=chat_history)
        
        # Step 2: Agent Tool-Binding Call (with fallback for Groq tool_use_failed errors)
        yield f"data: {json.dumps({'type': 'trace', 'data': '🧭 Agent evaluating tool requirements...'})}\n\n"
        
        try:
            first_response = await llm_with_tools.ainvoke(messages)
        except Exception as tool_bind_error:
            # Groq rejected the tool call format. Instead of calling LLM without tools (which hallucinates),
            # directly invoke the search tool ourselves and feed the results back to the LLM.
            yield f"data: {json.dumps({'type': 'trace', 'data': '⚠️ Tool binding failed, forcing direct RAG search...'})}\n\n"
            yield f"data: {json.dumps({'type': 'trace', 'data': '🔍 Querying FAISS dense vector index...'})}\n\n"
            yield f"data: {json.dumps({'type': 'trace', 'data': '📊 Cross-Encoder relevance filtering...'})}\n\n"
            
            # Directly run the search tool
            tool_output = search_pakistan_law.invoke({"query": question})
            used_search_tool = True
            
            yield f"data: {json.dumps({'type': 'trace', 'data': '✅ Tool search_pakistan_law finished processing.'})}\n\n"
            
            # Extract and emit sources
            sources = set(re.findall(r"\[Source:\s*(.*?)\]", str(tool_output)))
            if sources:
                yield f"data: {json.dumps({'type': 'sources', 'data': list(sources)})}\n\n"
            
            # Build a simple context-stuffed prompt and stream the answer
            fallback_messages = [
                SystemMessage(content="""You are an elite Pakistani legal advisor. Answer the question using ONLY the provided legal context below. Cite the [Source: ...] tags in your answer. Act like a confident lawyer building a case."""),
                HumanMessage(content=f"Legal Context:\n{tool_output}\n\nQuestion: {question}")
            ]
            
            yield f"data: {json.dumps({'type': 'trace', 'data': '⚡ Streaming synthesized agent response...'})}\n\n"
            async for chunk in llm.astream(fallback_messages):
                if chunk.content:
                    full_answer += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk.content})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        
        # Handle potential XML-style tool leakage from Groq's Llama-3
        xml_tool_match = None
        if not first_response.tool_calls and first_response.content:
            xml_tool_match = re.search(r'<function=(\w+)>\s*(\{.*?\})\s*</function>', first_response.content, re.DOTALL)
            if not xml_tool_match:
                # Also try the malformed double-equals format: <function=name={"query":"..."}</function>
                alt_match = re.search(r'<function=(\w+)=(\{.*?\})</function>', first_response.content, re.DOTALL)
                if alt_match:
                    xml_tool_match = alt_match
            
        # Step 3: Handle Tool Calls if the Agent decides to use them
        if first_response.tool_calls or xml_tool_match:
            
            tool_calls_to_execute = first_response.tool_calls or []
            
            if not tool_calls_to_execute and xml_tool_match:
                tool_name = xml_tool_match.group(1)
                try:
                    tool_args = json.loads(xml_tool_match.group(2))
                    tool_calls_to_execute = [{"name": tool_name, "args": tool_args, "id": "call_xml_manual"}]
                    first_response.content = "I need to check the exact text of the law for this."
                except Exception:
                    pass

            messages.append(first_response)

            for tool_call in tool_calls_to_execute:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                yield f"data: {json.dumps({'type': 'trace', 'data': f'🛠️ Agent invoking tool: {tool_name}'})}\n\n"
                if tool_name == "search_pakistan_law":
                    used_search_tool = True
                    yield f"data: {json.dumps({'type': 'trace', 'data': '🔍 Querying FAISS dense vector index...'})}\n\n"
                    yield f"data: {json.dumps({'type': 'trace', 'data': '📊 Cross-Encoder relevance filtering...'})}\n\n"
                elif tool_name == "calculate_tax":
                    yield f"data: {json.dumps({'type': 'trace', 'data': '🧮 Executing strict math calculations...'})}\n\n"
                    
                # Execute the tool
                if tool_name in tool_map:
                    selected_tool = tool_map[tool_name]
                    tool_output = selected_tool.invoke(tool_args)
                else:
                    tool_output = "Error: Invalid tool requested."
                
                yield f"data: {json.dumps({'type': 'trace', 'data': f'✅ Tool {tool_name} finished processing.'})}\n\n"
                
                # If legal search, extract sources dynamically
                if tool_name == "search_pakistan_law":
                    sources = set(re.findall(r"\[Source:\s*(.*?)\]", str(tool_output)))
                    if sources:
                        yield f"data: {json.dumps({'type': 'sources', 'data': list(sources)})}\n\n"
                        
                # Add tool output to conversation
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
                
            # Step 4: Stream Final Answer given the Tool Output
            yield f"data: {json.dumps({'type': 'trace', 'data': '⚡ Streaming synthesized agent response...'})}\n\n"
            
            # Since we manually modified the first_response for XML matches, we should use a fresh LLM call for the answer
            async for chunk in llm.astream(messages):
                if chunk.content:
                    full_answer += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk.content})}\n\n"
                    
        else:
            # Step 3 (No Tools): The Agent decided to just talk directly
            yield f"data: {json.dumps({'type': 'trace', 'data': '🗣️ Direct conversational agent response...'})}\n\n"
            async for chunk in llm.astream(messages):
                if chunk.content:
                    full_answer += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk.content})}\n\n"
                    
    except Exception as e:
         yield f"data: {json.dumps({'type': 'trace', 'data': f'❌ Agent Pipeline Error: {str(e)}'})}\n\n"

    # Run RAGAS evaluation if search tool was used and we have contexts
    # This runs regardless of whether there were exceptions during response generation
    if used_search_tool and full_answer and _current_contexts:
        try:
            yield f"data: {json.dumps({'type': 'trace', 'data': '📊 Running RAGAS evaluation...'})}\n\n"
            
            # Run evaluation asynchronously
            scores = await ragas_evaluator.evaluate_single(
                question=question,
                answer=full_answer,
                contexts=_current_contexts
            )
            
            # Log evaluation to database
            eval_db.log_evaluation(
                question=question,
                answer=full_answer,
                contexts=_current_contexts,
                faithfulness=scores["faithfulness"],
                answer_relevance=scores["answer_relevance"],
                context_recall=scores["context_recall"],
                user_id=user_id,
                org_id=org_id
            )
            
            yield f"data: {json.dumps({'type': 'evaluation', 'data': scores})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'trace', 'data': f'⚠️ Evaluation failed: {str(e)}'})}\n\n"
    
    # Clear global contexts
    _current_contexts = []

    yield f"data: {json.dumps({'type': 'done'})}\n\n"