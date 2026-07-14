from prometheus_client import Counter, Histogram, Gauge

# Latency histograms
REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds', 
    'Application Request Latency',
    ['method', 'endpoint']
)

# Counter for tracking how many queries resulted in an answer vs abstention
GROUNDING_PASS_COUNTER = Counter(
    'app_grounding_pass_total',
    'Total queries that passed grounding restrictions vs abstained',
    ['status'] # passed or abstained
)

GROUNDING_CHECKS_TOTAL = Counter(
    'app_grounding_checks_total', 
    'Total number of grounding validations performed'
)

GROUNDING_FAILURES_TOTAL = Counter(
    'app_grounding_failures_total', 
    'Total number of grounding validations that failed (proxy for hallucination)'
)

# Gauge for hallucination trend
HALLUCINATION_RATE = Gauge(
    'app_hallucination_rate',
    'Rolling hallucination rate percentage (last 100 requests)'
)

from collections import deque
import threading

_rolling_results = deque(maxlen=100)
_rolling_lock = threading.Lock()

def record_grounding_result(passed: bool):
    """
    Called after grounding validation to update the rolling hallucination rate
    and the companion counters. This is a fast, thread-safe memory operation.
    """
    GROUNDING_CHECKS_TOTAL.inc()
    if not passed:
        GROUNDING_FAILURES_TOTAL.inc()
        
    with _rolling_lock:
        _rolling_results.append(passed)
        total = len(_rolling_results)
        if total > 0:
            failures = sum(1 for p in _rolling_results if not p)
            HALLUCINATION_RATE.set((failures / total) * 100.0)
