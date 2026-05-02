import time
import json
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from monitoring.tracing import tracer
from monitoring.metrics import REQUEST_LATENCY, GROUNDING_PASS_COUNTER

class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        with tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            
            response = await call_next(request)
            
            process_time = time.time() - start_time
            REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(process_time)
            
            span.set_attribute("http.status_code", response.status_code)
            
            return response
