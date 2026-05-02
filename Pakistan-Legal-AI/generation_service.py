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
)
from db.schema import db_schema as eval_db
from evals.generation_metrics import ragas_evaluator
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
    """Calculates income tax based on the Pakistani tax brackets given an annual income in PKR."""
    tax = _calculate_tax_internal(annual_income)
    return f"Calculated Tax: {tax:,.2f} PKR. Remember to cite [Source: Tax Law - Income Tax]"

@tool
def search_legal_db_tool(query: str) -> str:
    """Searches the Pakistan Legal Database for relevant laws, constitutional articles, or regulations. Use this to answer legal questions."""
    ctx = get_request_context()
    org_id = ctx.org_id if ctx else "public"
    retrieval = retrieval_service.retrieve(query, org_id=org_id, k=6)
    
    contexts = [c.text for c in retrieval.chunks]
    set_retrieved_contexts(contexts)
    
    if not contexts:
        return "ERROR: INSUFFICIENT_EVIDENCE. No relevant legal documents found."
        
    context_blob = "\n\n".join([f"[Source: {c.source_id}]\n{c.text}" for c in retrieval.chunks])
    return f"Context found:\n{context_blob}"


LEGAL_KEYWORDS = [
    "lawful", "illegal", "must", "shall", "entitled", "prohibited", 
    "liable", "immunity", "penalty", "punishable"
]

NUMERIC_PATTERN = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")
SOURCE_PATTERN = re.compile(r"\[Source:\s*([^\]]+)\]")

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
        return False
        
    return True

def _abstain() -> str:
    return "ERROR: INSUFFICIENT_EVIDENCE"


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
            "You are a helpful legal assistant for Pakistani Law. "
            "You have tools to calculate tax and search the legal database. "
            "If a user greets you or asks general questions, you can respond directly without using tools. "
            "For ANY legal claim, numeric claim, or legal conclusion, you MUST use the search_legal_db_tool "
            "and then include a citation as [Source: <source_id>] in your final answer. "
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
            
            tool_calls = response.tool_calls
            if tool_calls:
                for tc in tool_calls:
                    tc_name = tc["name"]
                    yield f"data: {json.dumps({'type': 'trace', 'data': f'🛠️ Agent invoking tool: {tc_name}'})}\n\n"
                    
                    if tc["name"] == "calculate_tax_tool":
                        tool_msg = calculate_tax_tool.invoke(tc["args"])
                        sources_list.append("Tax Law - Income Tax")
                    elif tc["name"] == "search_legal_db_tool":
                        tool_msg = search_legal_db_tool.invoke(tc["args"])
                        for m in SOURCE_PATTERN.finditer(tool_msg):
                            sources_list.append(m.group(1))
                    else:
                        tool_msg = "Unknown tool"
                        
                    messages.append(response)
                    messages.append(ToolMessage(content=tool_msg, tool_call_id=tc["id"]))
                    
                if sources_list:
                    sources_list = list(set(sources_list))
                    yield f"data: {json.dumps({'type': 'sources', 'data': sources_list})}\n\n"
                
                yield f"data: {json.dumps({'type': 'trace', 'data': '✍️ Synthesizing final answer'})}\n\n"
                final_response = await self.agent.ainvoke(messages)
                answer = final_response.content
            else:
                answer = response.content

            if answer != _abstain() and not _validate_grounding(answer):
                answer = _abstain()
            elif "ERROR: INSUFFICIENT_EVIDENCE" in answer:
                answer = _abstain()

            yield f"data: {json.dumps({'type': 'token', 'data': answer})}\n\n"

            contexts = get_retrieved_contexts()
            if contexts:
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
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'trace', 'data': f'❌ Pipeline error: {str(exc)}'})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'data': _abstain()})}\n\n"
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
            
            tool_calls = response.tool_calls
            if tool_calls:
                for tc in tool_calls:
                    if tc["name"] == "calculate_tax_tool":
                        tool_msg = calculate_tax_tool.invoke(tc["args"])
                        sources_list.append("Tax Law - Income Tax")
                    elif tc["name"] == "search_legal_db_tool":
                        tool_msg = search_legal_db_tool.invoke(tc["args"])
                        for m in SOURCE_PATTERN.finditer(tool_msg):
                            sources_list.append(m.group(1))
                    else:
                        tool_msg = "Unknown tool"
                        
                    messages.append(response)
                    messages.append(ToolMessage(content=tool_msg, tool_call_id=tc["id"]))
                
                final_response = await asyncio.wait_for(self.agent.ainvoke(messages), timeout=15.0)
                answer = final_response.content
            else:
                answer = response.content

            sources_list = list(set(sources_list))
            contexts = get_retrieved_contexts()
            confidence = 1.0 if not sources_list else 0.85
            requires_review = False

            if answer != _abstain() and not _validate_grounding(answer):
                answer = _abstain()
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
