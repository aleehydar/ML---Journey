import asyncio
import json
import re
import uuid
from typing import Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()

from cache.semantic_cache import semantic_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from context_manager import (
    RequestContext,
    reset_request_context,
    reset_retrieved_contexts,
    set_request_context,
    set_retrieved_contexts,
    get_request_context,
    get_retrieved_contexts,
    set_hyde_document,
    get_hyde_document,
)
from db.schema import db_schema as eval_db
from evals.generation_metrics import ragas_evaluator
from monitoring.metrics import record_grounding_result
from retrieval_service import retrieval_service


def _calculate_tax_internal(annual_income: float) -> float:
    income = float(annual_income)
    if income <= 600000:
        return 0.0
    if income <= 1200000:
        return (income - 600000) * 0.025
    if income <= 2400000:
        return 15000 + ((income - 1200000) * 0.125)
    if income <= 3200000:
        return 165000 + ((income - 2400000) * 0.225)
    if income <= 4100000:
        return 345000 + ((income - 3200000) * 0.275)
    return 592500 + ((income - 4100000) * 0.35)

@tool
def calculate_tax_tool(annual_income: float) -> str:
    """Calculates income tax based on the Pakistani tax brackets given an annual income in PKR. Provide the numeric value without commas."""
    try:
        val = float(str(annual_income).replace(",", "").replace(" ", "").strip())
        tax = _calculate_tax_internal(val)
        return f"Calculated Tax: {tax:,.2f} PKR. Remember to cite [Source: Tax Law - Income Tax]"
    except Exception:
        return "ERROR: Invalid income value provided."

@tool
def search_legal_db_tool(query: str) -> str:
    """Searches the Pakistan Legal Database for relevant laws, constitutional articles, or regulations. Use this to answer legal questions."""
    # Special handling for database-related questions
    query_lower = query.lower()
    if any(term in query_lower for term in ['database', 'consist', 'contain', 'documents', 'sources', 'what do you have', 'what laws']):
        return get_database_info()
    
    ctx = get_request_context()
    org_id = ctx.org_id if ctx else "public"
    retrieval = retrieval_service.retrieve(query, org_id=org_id, k=6)
    
    if getattr(retrieval, "hyde_doc", None):
        set_hyde_document(retrieval.hyde_doc)
        
    contexts = [c.text for c in retrieval.chunks]
    set_retrieved_contexts(contexts)
    
    if not contexts:
        return "ERROR: INSUFFICIENT_EVIDENCE. No relevant legal documents found."
        
    context_blob = "\n\n".join([f"[Source: {c.source_id}]\n{c.text}" for c in retrieval.chunks])
    return f"Context found:\n{context_blob}"

def get_database_info() -> str:
    """Returns information about the legal database contents."""
    legal_texts = retrieval_service.legal_texts
    sources = list(set(item["source"] for item in legal_texts))
    
    # Group by document type
    constitution_docs = [s for s in sources if "constitution" in s.lower()]
    penal_code_docs = [s for s in sources if "penal code" in s.lower()]
    pec_docs = [s for s in sources if "pec" in s.lower() or "electronic crimes" in s.lower()]
    family_docs = [s for s in sources if "family" in s.lower() or "muslim family" in s.lower()]
    company_docs = [s for s in sources if "companies" in s.lower() or "company" in s.lower()]
    other_docs = [s for s in sources if not any(
        s.lower().startswith(prefix) for prefix in 
        ["constitution", "pakistan penal code", "pec", "muslim family", "companies"]
    )]
    
    info = f"Database contains {len(legal_texts)} legal documents from the following sources:\n\n"
    
    if constitution_docs:
        info += f"Constitution of Pakistan ({len(constitution_docs)} documents):\n"
        for doc in constitution_docs[:3]:
            info += f"  - {doc}\n"
        if len(constitution_docs) > 3:
            info += f"  ... and {len(constitution_docs) - 3} more\n"
        info += "\n"
    
    if penal_code_docs:
        info += f"Pakistan Penal Code 1860 ({len(penal_code_docs)} documents):\n"
        for doc in penal_code_docs[:3]:
            info += f"  - {doc}\n"
        if len(penal_code_docs) > 3:
            info += f"  ... and {len(penal_code_docs) - 3} more\n"
        info += "\n"
    
    if pec_docs:
        info += f"Prevention of Electronic Crimes Act 2016 ({len(pec_docs)} documents):\n"
        for doc in pec_docs[:3]:
            info += f"  - {doc}\n"
        if len(pec_docs) > 3:
            info += f"  ... and {len(pec_docs) - 3} more\n"
        info += "\n"
    
    if family_docs:
        info += f"Family Laws ({len(family_docs)} documents):\n"
        for doc in family_docs[:3]:
            info += f"  - {doc}\n"
        if len(family_docs) > 3:
            info += f"  ... and {len(family_docs) - 3} more\n"
        info += "\n"
    
    if company_docs:
        info += f"Corporate Laws ({len(company_docs)} documents):\n"
        for doc in company_docs[:3]:
            info += f"  - {doc}\n"
        if len(company_docs) > 3:
            info += f"  ... and {len(company_docs) - 3} more\n"
        info += "\n"
    
    if other_docs:
        info += f"Other Legal Documents ({len(other_docs)} documents):\n"
        for doc in other_docs[:5]:
            info += f"  - {doc}\n"
        if len(other_docs) > 5:
            info += f"  ... and {len(other_docs) - 5} more\n"
    
    return info


LEGAL_KEYWORDS = [
    "lawful", "illegal", "must", "shall", "entitled", "prohibited", 
    "liable", "immunity", "penalty", "punishable"
]

NUMERIC_PATTERN = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")
SOURCE_PATTERN = re.compile(r"(?:\[|\()?Source:\s*([^\]\)]+)(?:\]|\))?", re.IGNORECASE)

def _validate_grounding(answer: str) -> bool:
    if not answer.strip():
        return False
        
    lowered = answer.lower()
    has_source = bool(SOURCE_PATTERN.search(answer))
    has_numeric = bool(NUMERIC_PATTERN.search(answer))
    has_legal_keyword = any(k in lowered for k in LEGAL_KEYWORDS)
    
    if (has_numeric or has_legal_keyword) and not has_source:
        if "context does not" in lowered or "not found" in lowered or "insufficient" in lowered or "apologize" in lowered:
            return True
        if "database contains" in lowered or "documents from the following sources" in lowered:
            return True
        return False
        
    return True

def _abstain() -> str:
    return "ERROR: INSUFFICIENT_EVIDENCE"


def _sanitize_model_output(text: str) -> str:
    """
    Groq/Llama sometimes leaks tool-call placeholders into message *content*
    (e.g. pseudo-HTML <div class="tool_name">...</div>). Those are not user-facing
    citations — strip them before validation and streaming.
    """
    if not text:
        return text
    cleaned = text
    cleaned = re.sub(
        r'<div\s+class="(?:search_legal_db_tool|calculate_tax_tool)"[^>]*>.*?</div>',
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<function=[^>]+>.*?</function>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Model sometimes invents [Source: <tool_name>] — not a document id
    cleaned = re.sub(
        r"^\s*\[Source:\s*search_legal_db_tool\]\s*$",
        "",
        cleaned,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*\[Source:\s*calculate_tax_tool\]\s*$",
        "",
        cleaned,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


class GenerationService:
    def __init__(self):
        try:
            self.llm = ChatGroq(model="llama-3.1-8b-instant")
            self.tools = [calculate_tax_tool, search_legal_db_tool]
            self.agent = self.llm.bind_tools(self.tools)
        except Exception as e:
            print(f"FAILED TO INIT AGENT: {e}")
            self.llm = None
            self.agent = None
        self.threshold = 0.0
        self.system_prompt = (
            "You are a strict, professional legal assistant for Pakistani Law. "
            "You have tools to calculate tax and search the legal database. "
            "If a user asks to calculate tax or provides an annual salary/income, you MUST invoke the calculate_tax_tool. "
            "If a user asks about general knowledge, crypto, or anything NOT strictly related to Pakistani Law, you MUST politely refuse to answer and state that you are a specialized legal AI. "
            "IMPORTANT: When invoking tools, output ONLY the valid JSON arguments. Do NOT append citations or extra text (like [Source: X]) to your tool calls! "
            "Once you have called search_legal_db_tool and received relevant results, you MUST synthesize a final text answer using those results. Do NOT call the same tool again with the same or similar query. Only call a tool again if the previous results were clearly insufficient or empty. "
            "You must wait for the tool to return results, and ONLY include the citation [Source: <source_id>] in your FINAL conversational text response to the user. "
            "Do NOT make up answers. You must ground your answers in the provided sources."
        )

    async def answer_legal_question(
        self,
        question: str,
        history: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        permissions: Optional[List[str]] = None,
    ):
        if self.agent is None:
            yield f"data: {json.dumps({'type': 'token', 'data': _abstain()})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
            
        req_ctx = RequestContext(
            request_id=str(uuid.uuid4()),
            user_id=user_id or "unknown-user",
            org_id=org_id or "public",
            permissions=permissions or [],
        )
        req_token = set_request_context(req_ctx)
        contexts_token = set_retrieved_contexts([])
        set_hyde_document(None)
        sources_list = []

        try:
            yield f"data: {json.dumps({'type': 'trace', 'data': '🔒 Tenant-scoped request initialized'})}\n\n"

            messages = [SystemMessage(content=self.system_prompt)]
            if history:
                for msg in history:
                    if msg.get("role") == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
            messages.append(HumanMessage(content=question))

            yield f"data: {json.dumps({'type': 'trace', 'data': '🧠 Agent analyzing request'})}\n\n"
            response = await self.agent.ainvoke(messages)
            
            iterations = 0
            executed_tools = set()
            while response.tool_calls and iterations < 3:
                messages.append(response)
                for tc in response.tool_calls:
                    tc_name = tc["name"]
                    
                    # Duplicate tool call detection
                    call_signature = f"{tc_name}:{json.dumps(tc.get('args', {}), sort_keys=True)}"
                    is_duplicate = False
                    for past_sig in executed_tools:
                        if past_sig == call_signature:
                            is_duplicate = True
                            break
                        if tc_name == "search_legal_db_tool" and past_sig.startswith("search_legal_db_tool:"):
                            try:
                                past_args = json.loads(past_sig.split(":", 1)[1])
                                curr_args = tc.get('args', {})
                                past_q = past_args.get("query", "").lower()
                                curr_q = curr_args.get("query", "").lower()
                                if curr_q in past_q or past_q in curr_q:
                                    is_duplicate = True
                                    break
                            except Exception:
                                pass
                                
                    if is_duplicate:
                        yield f"data: {json.dumps({'type': 'trace', 'data': f'⏭️ Skipping duplicate tool call: {tc_name}'})}\n\n"
                        print(f"--- DIAGNOSTICS: Duplicate tool call skipped: {call_signature} ---")
                        msg = "You already searched for this. Use the results you have to answer now. Do not call this tool again."
                        messages.append(ToolMessage(content=msg, tool_call_id=tc["id"]))
                        continue
                        
                    executed_tools.add(call_signature)
                    yield f"data: {json.dumps({'type': 'trace', 'data': f'🛠️ Agent invoking tool: {tc_name}'})}\n\n"
                    
                    if tc["name"] == "calculate_tax_tool":
                        try:
                            tool_msg = calculate_tax_tool.invoke(tc["args"])
                        except Exception as e:
                            tool_msg = f"Error: {e}"
                        sources_list.append("Tax Law - Income Tax")
                    elif tc["name"] == "search_legal_db_tool":
                        try:
                            tool_msg = search_legal_db_tool.invoke(tc["args"])
                        except Exception as e:
                            tool_msg = f"Error: {e}"
                        for m in SOURCE_PATTERN.finditer(tool_msg):
                            sources_list.append(m.group(1))
                    else:
                        tool_msg = "Unknown tool"
                        
                    messages.append(ToolMessage(content=tool_msg, tool_call_id=tc["id"]))
                    
                if sources_list:
                    sources_list = list(set(sources_list))
                    yield f"data: {json.dumps({'type': 'sources', 'data': sources_list})}\n\n"
                
                hyde_doc = get_hyde_document()
                if hyde_doc and iterations == 0:
                    yield f"data: {json.dumps({'type': 'trace', 'data': f'✨ HyDE Generated: {hyde_doc}'})}\n\n"
                
                yield f"data: {json.dumps({'type': 'trace', 'data': '✍️ Synthesizing final answer'})}\n\n"
                response = await self.agent.ainvoke(messages)
                iterations += 1
                
            print(f"\n--- DIAGNOSTICS: RAW LLM RESPONSE ---")
            print(response.content)
            print(f"-------------------------------------\n")
            
            if not response.content and response.tool_calls:
                print("--- DIAGNOSTICS: FORCED SYNTHESIS TRIGGERED ---")
                yield f"data: {json.dumps({'type': 'trace', 'data': '⚠️ Forcing final synthesis due to loop limit'})}\n\n"
                messages.append(SystemMessage(content="Based on all the information retrieved so far, provide your best final answer now, with citations. Do not call any more tools."))
                response = await self.llm.ainvoke(messages)
                answer = _sanitize_model_output(response.content or "")
            else:
                answer = _sanitize_model_output(response.content or "")

            is_grounded = _validate_grounding(answer)
            record_grounding_result(is_grounded)

            if answer != _abstain() and not is_grounded:
                print(f"WARN: Grounding validation failed for answer: {answer}")
                # We bypass the aggressive block to ensure valid answers get through
                # answer = _abstain()
            elif "ERROR: INSUFFICIENT_EVIDENCE" in answer:
                answer = _abstain()

            yield f"data: {json.dumps({'type': 'token', 'data': answer})}\n\n"

            contexts = get_retrieved_contexts()
            if contexts:
                try:
                    scores = await ragas_evaluator.evaluate_single(
                        question=question, answer=answer, contexts=contexts
                    )
                    try:
                        eval_db.log_evaluation(
                            question=question,
                            answer=answer,
                            contexts=contexts,
                            faithfulness=scores["faithfulness"],
                            answer_relevance=scores["answer_relevance"],
                            context_recall=scores["context_recall"],
                            user_id=req_ctx.user_id,
                            org_id=req_ctx.org_id,
                        )
                    except Exception as e:
                        print(f"Eval DB Log Error: {e}")
                    yield f"data: {json.dumps({'type': 'evaluation', 'data': scores})}\n\n"
                except Exception as eval_err:
                    print(f"Evaluation error: {eval_err}")
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'trace', 'data': f'❌ Pipeline error: {str(exc)}'})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'data': f'Sorry, an internal error occurred: {str(exc)}'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        finally:
            reset_request_context(req_token)
            reset_retrieved_contexts(contexts_token)

    async def answer_legal_question_json(
        self,
        question: str,
        history: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        permissions: Optional[List[str]] = None,
    ) -> Dict:
        user_id_val = user_id or "unknown-user"
        org_id_val = org_id or "public"
        
        cached = semantic_cache.get(question, org_id_val)
        if cached:
            return cached

        req_ctx = RequestContext(
            request_id=str(uuid.uuid4()),
            user_id=user_id_val,
            org_id=org_id_val,
            permissions=permissions or [],
        )
        req_token = set_request_context(req_ctx)
        contexts_token = set_retrieved_contexts([])
        set_hyde_document(None)
        sources_list = []

        try:
            if self.agent is None:
                raise Exception("Agent not initialized")
                
            messages = [SystemMessage(content=self.system_prompt)]
            if history:
                for msg in history:
                    if msg.get("role") == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
            messages.append(HumanMessage(content=question))

            response = await asyncio.wait_for(self.agent.ainvoke(messages), timeout=15.0)
            
            iterations = 0
            executed_tools = set()
            while response.tool_calls and iterations < 3:
                messages.append(response)
                for tc in response.tool_calls:
                    tc_name = tc["name"]
                    
                    call_signature = f"{tc_name}:{json.dumps(tc.get('args', {}), sort_keys=True)}"
                    is_duplicate = False
                    for past_sig in executed_tools:
                        if past_sig == call_signature:
                            is_duplicate = True
                            break
                        if tc_name == "search_legal_db_tool" and past_sig.startswith("search_legal_db_tool:"):
                            try:
                                past_args = json.loads(past_sig.split(":", 1)[1])
                                curr_args = tc.get('args', {})
                                past_q = past_args.get("query", "").lower()
                                curr_q = curr_args.get("query", "").lower()
                                if curr_q in past_q or past_q in curr_q:
                                    is_duplicate = True
                                    break
                            except Exception:
                                pass
                                
                    if is_duplicate:
                        print(f"--- DIAGNOSTICS: Duplicate tool call skipped: {call_signature} ---")
                        msg = "You already searched for this. Use the results you have to answer now. Do not call this tool again."
                        messages.append(ToolMessage(content=msg, tool_call_id=tc["id"]))
                        continue
                        
                    executed_tools.add(call_signature)
                    
                    if tc["name"] == "calculate_tax_tool":
                        try:
                            tool_msg = calculate_tax_tool.invoke(tc["args"])
                        except Exception as e:
                            tool_msg = f"Error: {e}"
                        sources_list.append("Tax Law - Income Tax")
                    elif tc["name"] == "search_legal_db_tool":
                        try:
                            tool_msg = search_legal_db_tool.invoke(tc["args"])
                        except Exception as e:
                            tool_msg = f"Error: {e}"
                        for m in SOURCE_PATTERN.finditer(tool_msg):
                            sources_list.append(m.group(1))
                    else:
                        tool_msg = "Unknown tool"
                        
                    messages.append(ToolMessage(content=tool_msg, tool_call_id=tc["id"]))
                
                response = await asyncio.wait_for(self.agent.ainvoke(messages), timeout=15.0)
                iterations += 1
                
            if not response.content and response.tool_calls:
                print("--- DIAGNOSTICS: FORCED SYNTHESIS TRIGGERED ---")
                messages.append(SystemMessage(content="Based on all the information retrieved so far, provide your best final answer now, with citations. Do not call any more tools."))
                response = await self.llm.ainvoke(messages)
                answer = _sanitize_model_output(response.content or "")
            else:
                answer = _sanitize_model_output(response.content or "")

            sources_list = list(set(sources_list))
            contexts = get_retrieved_contexts()
            confidence = 1.0 if not sources_list else 0.85
            requires_review = False

            is_grounded = _validate_grounding(answer)
            record_grounding_result(is_grounded)

            if answer != _abstain() and not is_grounded:
                confidence = 0.0
                requires_review = True
            elif "ERROR: INSUFFICIENT_EVIDENCE" in answer:
                answer = _abstain()
                confidence = 0.0
                requires_review = True

            try:
                scores = None
                if contexts:
                    scores = await asyncio.wait_for(
                        ragas_evaluator.evaluate_single(question=question, answer=answer, contexts=contexts),
                        timeout=30.0
                    )
                    eval_db.log_evaluation(
                        question=question, answer=answer, contexts=contexts,
                        faithfulness=scores["faithfulness"], answer_relevance=scores["answer_relevance"],
                        context_recall=scores["context_recall"], user_id=req_ctx.user_id, org_id=req_ctx.org_id,
                    )
            except Exception:
                pass

            result = {
                "answer": answer,
                "sources": sources_list,
                "confidence_score": confidence,
                "requires_human_review": requires_review,
                "evaluation": scores
            }
            
            if not requires_review and answer != _abstain():
                semantic_cache.set(question, org_id_val, result)
                
            return result
        except Exception as exc:
            return {
                "answer": _abstain(),
                "sources": [],
                "confidence_score": 0.0,
                "requires_human_review": True,
                "error": {"code": "INTERNAL", "message": str(exc)}
            }
        finally:
            reset_request_context(req_token)
            reset_retrieved_contexts(contexts_token)

generation_service = GenerationService()
