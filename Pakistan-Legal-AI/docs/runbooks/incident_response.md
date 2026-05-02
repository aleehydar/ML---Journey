# Incident Response Runbook

## Scope
Defines the standard operating procedure during system latency spikes or 500 error cascades.

## Alerts Triggered
- `Latency > 2.0s` for p95 over 5 minutes.
- `HTTP 5xx Error Rate > 1%` over 5 minutes.

## Procedure
1. **Acknowledge** the incident in the monitoring channel.
2. **Triangular Debugging**:
   - Verify Groq LLM API Status (https://status.groq.com).
   - Check local Prometheus Dashboard for CPU/Memory saturation.
   - Inspect open traces in Jaeger/Zipkin (via OpenTelemetry).
3. **Mitigation**:
   - If Groq is down, switch to fallback model via ENV var: `FALLBACK_LLM_ENABLED=true`.
   - If DB is locked, restart the container `docker-compose restart legal-ai`.
4. **Communication**:
   - Update status page.
5. **Post-Mortem**: Document root cause.
