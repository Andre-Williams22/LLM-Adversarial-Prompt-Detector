# Grafana + Prometheus Monitoring Setup

This setup provides comprehensive monitoring for your Adversarial Prompt Detector using Grafana and Prometheus.

## Quick Start

1. **Start the monitoring stack:**
   ```bash
   ./start-monitoring.sh
   ```

2. **Start your FastAPI application:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8080
   ```

3. **Access Grafana Dashboard:**
   - URL: http://localhost:3000
   - Username: `admin`
   - Password: `admin123`

## What You'll See

### Dashboard Panels:

1. **Model Latency Percentiles** - Track 95th and 50th percentile response times for each model
2. **Predictions per Model (Pie Chart)** - Distribution of predictions across models in the last hour
3. **Prediction Rate per Model** - Real-time requests per second for each model
4. **95th Percentile Latency by Model** - Current performance stats for each model
5. **Total Predictions by Model** - Cumulative prediction counts over time

### Key Metrics Tracked:

- `model_latency_seconds` - Histogram of model inference times
- `model_prediction_total` - Counter of predictions per model
- Individual model performance (electra_small, tox_bert, offensive_roberta, bart_mnli)
- Overall ensemble performance (detect_adversarial_prompt)

## Customizing Dashboards

1. Access Grafana at http://localhost:3000
2. Navigate to "Dashboards" → "Adversarial Prompt Detector - Model Performance"
3. Click "Edit" on any panel to modify queries or visualization
4. Add new panels using the "+" button

## Advanced Queries

### Useful Prometheus Queries:

```promql
# Average latency per model
rate(model_latency_seconds_sum[5m]) / rate(model_latency_seconds_count[5m])

# Error rate (if you add error metrics)
rate(model_errors_total[5m])

# Throughput
rate(model_prediction_total[5m])

# 99th percentile latency
histogram_quantile(0.99, sum(rate(model_latency_seconds_bucket[5m])) by (le, model_name))
```

## Troubleshooting

### If metrics aren't showing:
1. Check that your FastAPI app is running on port 8080
2. Verify metrics endpoint: http://localhost:8001/metrics
3. Check Prometheus targets: http://localhost:9090/targets

### If Grafana isn't accessible:
```bash
# Check container status (newer Docker)
docker compose -f docker-compose.monitoring.yml ps

# Check container status (older Docker)
docker-compose -f docker-compose.monitoring.yml ps

# View logs (newer Docker)
docker compose -f docker-compose.monitoring.yml logs grafana

# View logs (older Docker)
docker-compose -f docker-compose.monitoring.yml logs grafana
```

## Stopping the Stack

```bash
# For newer Docker versions
docker compose -f docker-compose.monitoring.yml down

# For older Docker versions
docker-compose -f docker-compose.monitoring.yml down
```

## Production Considerations

For production deployment:
1. Change default Grafana admin password
2. Set up persistent volumes for data retention
3. Configure alerting rules in Prometheus
4. Add authentication and SSL/TLS
5. Consider using cloud-managed Prometheus/Grafana services
