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

load_dotenv()

# Preload CrossEncoder for reranking
print("Loading CrossEncoder...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─── VECTORSTORE: Load from disk (PDF pipeline) or fallback to hardcoded ────
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")

if os.path.exists(VECTORSTORE_DIR):
    print(f"📂 Loading persisted FAISS index from {VECTORSTORE_DIR}/...")
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    print(f"✅ Loaded vectorstore with PDF-ingested documents!")
    # Legal texts for the document viewer API (not used for retrieval)
    legal_texts = []
else:
    print("⚠️  No vectorstore/ found. Using hardcoded fallback data.")
    print("   Run 'python ingest.py' to build a persistent index from PDFs.")
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
        {"source": "Tenant Rights", "text": "A landlord cannot evict a tenant without proper legal notice. A minimum of one month notice is required before eviction proceedings can begin."}
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
    
    # 1. HyDE (Hypothetical Document Embeddings): Ask the LLM to hallucinate a perfect legal excerpt
    hyde_prompt = f"Write a hypothetical, formal excerpt from a Pakistani law code that perfectly answers this query: '{query}'. Use strict legal terminology (e.g., 'communication during marriage', 'disclosure', 'privilege'). Do not include any introductory text, just the hypothetical legal text."
    hypothetical_doc = llm.invoke(hyde_prompt).content
    
    # 2. Search FAISS using the expanded hypothetical document to bridge the semantic gap
    relevant_docs = vectorstore.similarity_search(hypothetical_doc, k=40)
    
    pairs = [[query, doc.page_content] for doc in relevant_docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(scores, relevant_docs), key=lambda x: x[0], reverse=True)
    
    # Relevance Threshold: Filter out anything with a score < -5.0
    # Conversational text vs formal legal text often yields slightly negative MS MARCO scores (0 to -4), 
    # but completely irrelevant text yields deeply negative scores (-10).
    relevant_scored = [doc for score, doc in scored_docs if score > -5.0]
    top_docs = relevant_scored[:4]
    
    context_parts = []
    for doc in top_docs:
        context_parts.append(f"[Source: {doc.metadata['source']}]\n{doc.page_content}")
        
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

async def answer_legal_question(question: str, history: list = None):
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
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk.content})}\n\n"
                    
        else:
            # Step 3 (No Tools): The Agent decided to just talk directly
            yield f"data: {json.dumps({'type': 'trace', 'data': '🗣️ Direct conversational agent response...'})}\n\n"
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk.content})}\n\n"
                    
    except Exception as e:
         yield f"data: {json.dumps({'type': 'trace', 'data': f'❌ Agent Pipeline Error: {str(e)}'})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"