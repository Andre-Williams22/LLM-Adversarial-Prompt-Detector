# Monitoring

A safeguard classifier that nobody is watching is a safeguard classifier that
has already failed silently. This service exposes three layers of telemetry:
Prometheus metrics for operational health, MLflow runs for per-decision
forensics, and MongoDB records for longitudinal analysis.

## Running the stack locally

```bash
docker compose -f monitoring/docker-compose.yml up -d
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin / admin123 |
| Prometheus | http://localhost:9090 | none |

The dashboard in `monitoring/grafana/dashboards/` is provisioned
automatically. Prometheus scrape targets are in `monitoring/prometheus.yml`;
the default target assumes the application is on `host.docker.internal:8080`.

To push a dashboard edit back into the repository:

```bash
./scripts/update-dashboard.sh
```

## Prometheus metrics

Scraped from `GET /metrics`.

### Detection outcomes

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `adversarial_detections_total` | Counter | `model_name` | Prompts blocked |
| `safe_prompts_total` | Counter | `model_name` | Prompts allowed through |
| `chat_requests_total` | Counter | - | Total requests screened |
| `chat_requests_failed_total` | Counter | `error_type` | Requests that errored |

The ratio of `adversarial_detections_total` to `chat_requests_total` is the
block rate. A sudden move in either direction is the signal worth alerting on:
a spike suggests an attack campaign or a regression that made the policy too
strict, a collapse suggests the ensemble has silently stopped firing.

### Latency and throughput

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `model_inference_duration_seconds` | Histogram | `model_name`, `model_type` | Per-stage inference time |
| `model_inference_total` | Counter | `model_name`, `model_type` | Per-stage invocations |
| `model_inference_errors_total` | Counter | `model_name`, `error_type` | Stage failures |
| `concurrent_requests_count` | Gauge | - | Requests in flight |
| `model_queue_size` | Gauge | - | Requests waiting |

Comparing `model_inference_total{model_name="keyword_detector"}` against the
transformer stages gives the early-exit rate: the share of requests resolved by
keyword matching alone, which is the main determinant of average latency and
cost.

### System health

| Metric | Type | Meaning |
|--------|------|---------|
| `active_models_count` | Gauge | Ensemble members currently loaded |
| `cpu_usage_percent` | Gauge | Process CPU, 0-100 |
| `memory_usage_percent` | Gauge | Process memory, 0-100 |
| `system_cpu_usage` | Gauge | Host CPU, 0.0-1.0 |
| `system_memory_usage` | Gauge | Host memory, 0.0-1.0 |

`active_models_count` dropping below 4 means the ensemble is degraded and the
service is classifying with fewer signals than the policy assumes.

## Useful queries

```promql
# Block rate over 5 minutes
sum(rate(adversarial_detections_total[5m]))
  / sum(rate(chat_requests_total[5m]))

# 95th percentile end-to-end detection latency
histogram_quantile(0.95,
  sum(rate(model_inference_duration_seconds_bucket[5m])) by (le))

# Early-exit rate
sum(rate(model_inference_total{model_name="keyword_detector"}[5m]))
  / sum(rate(chat_requests_total[5m]))

# Stage error rate
sum(rate(model_inference_errors_total[5m])) by (model_name)
```

## MLflow

Every decision writes two kinds of run to the `adversarial_detection_system`
experiment: one per ensemble member with its score and inference time, and one
for the ensemble verdict with the voting details attached. That makes it
possible to reconstruct after the fact exactly why a given prompt was blocked,
including which voting rule fired.

Tracking is best effort. If the tracking store is unreachable the detector logs
one warning at startup and then classifies without it, because an observability
outage must not become a safety outage.

See `docs/mlflow-tracking.md` for the run schema and query examples.

## Interaction logging

When `MONGODB_URI` is set, each interaction is written asynchronously on a
background thread: prompt, response, verdict, scores, latency, and session ID.
The write is fire-and-forget by design, so a database outage cannot fail or
delay a request. `GET /stats` aggregates the last seven days.
