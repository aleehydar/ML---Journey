from prometheus_client import Counter, Histogram, make_asgi_app

# Counters
prediction_counter = Counter(
    'attrition_predictions_total',
    'Total predictions made',
    ['risk_level']
)

error_counter = Counter(
    'attrition_errors_total',
    'Total errors encountered',
    ['error_type']
)

# Histograms (latency tracking)
prediction_latency = Histogram(
    'attrition_prediction_duration_seconds',
    'Time spent processing predictions',
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

metrics_app = make_asgi_app()
