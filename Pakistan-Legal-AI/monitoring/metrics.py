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

# Gauge for hallucination trend
# This would be updated by a background worker polling the eval db
HALLUCINATION_RATE = Gauge(
    'app_hallucination_rate',
    'Rolling 7-day hallucination rate percentage'
)
