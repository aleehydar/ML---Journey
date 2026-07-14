import contextvars
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RequestContext:
    request_id: str
    user_id: str
    org_id: str
    permissions: List[str] = field(default_factory=list)


request_context_var: contextvars.ContextVar[Optional[RequestContext]] = contextvars.ContextVar(
    "request_context", default=None
)


retrieved_contexts_var: contextvars.ContextVar[List[str]] = contextvars.ContextVar(
    "retrieved_contexts", default=[]
)

hyde_document_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hyde_document", default=None
)


def set_request_context(ctx: RequestContext):
    return request_context_var.set(ctx)


def reset_request_context(token):
    request_context_var.reset(token)


def get_request_context() -> Optional[RequestContext]:
    return request_context_var.get()


def set_retrieved_contexts(contexts: List[str]):
    return retrieved_contexts_var.set(contexts)


def reset_retrieved_contexts(token):
    retrieved_contexts_var.reset(token)


def get_retrieved_contexts() -> List[str]:
    return retrieved_contexts_var.get()


def set_hyde_document(doc: Optional[str]):
    return hyde_document_var.set(doc)


def get_hyde_document() -> Optional[str]:
    return hyde_document_var.get()
