# Model Rollback Protocol

## Scope
Outlines steps to rollback the primary generation model if hallucinations exceed the 7-day trailing average threshold (e.g., `< 70% Faithfulness`).

## Procedure
1. Verify metric via `/api/eval/summary`.
2. Locate the last stable Docker Image or Commit hash.
3. If the model API itself changed, update `retrieval_service.py` top-level model constructor (e.g., reverting from `llama-3.1-8b` to `llama-3-8b`).
4. Commit changes: `git commit -m "Rollback generation model"`.
5. Trigger Redeploy through CI/CD via `git push origin main`.
6. Monitor the Hallucination Rate Gauge in Prometheus for the next 24 hours.
