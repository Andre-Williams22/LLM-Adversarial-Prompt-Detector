# LLM-Adversarial-Prompt-Detector

"""
PROJECT: LLM Adversarial Prompt Detector

GOAL:
Build a machine learning pipeline to detect adversarial prompts (e.g., jailbreaks, prompt injections) designed to manipulate LLMs like ChatGPT or Claude. This project includes dataset preparation, model training, evaluation, experiment tracking with MLflow, and public deployment using Streamlit or FastAPI on Google Cloud (andrewwilliams.ai).

KEY OBJECTIVES:
1. Detect if a prompt is safe or adversarial (binary classification)
2. Train a model on a labeled dataset of safe vs. adversarial prompts
3. Use MLflow for local and production tracking of metrics, parameters, and models
4. Create a frontend interface (Streamlit initially) for public interaction
5. Deploy app on Google Cloud Compute Engine, accessible via andrewwilliams.ai

FOLDER STRUCTURE:
├── data/
│   ├── raw/            # Contains raw datasets (e.g., jailbreak.jsonl, alpaca_clean.json)
│   ├── processed/      # Preprocessed labeled data (e.g., prompts_labeled.csv)
│
├── notebooks/          # Jupyter notebooks for EDA and experimentation
├── src/
│   ├── data/           # Preprocessing scripts
│   ├── features/       # Embedding/tokenization
│   ├── models/         # Training and evaluation scripts
│   └── inference/      # Inference logic for deployed app
│
├── app/                # Streamlit or FastAPI application
├── scripts/            # Entrypoint scripts (e.g., run_experiment.py)
├── mlruns/             # MLflow log folder
├── requirements.txt
├── config.yaml
├── .env
├── .gitignore
└── README.md

NEXT STEPS:
1. Download jailbreak and clean prompt datasets and move them to `data/raw/`
2. Write preprocessing script (`src/data/preprocess.py`) to clean, label, and save as `data/processed/prompts_labeled.csv`
3. Use `src/models/train.py` to train baseline classifier (Logistic Regression)
4. Track experiments using MLflow with `mlflow.start_run()`
5. Optionally: Create `src/inference/predict.py` to allow inference via CLI or Streamlit

NOTE TO COPILOT:
You will assist in writing clean, modular Python code for preprocessing, model training, evaluation, and inference. All experiments must be logged using MLflow.
"""



## Run the Application Locally

### Quick Start (All Services)

**Option 1: Manual startup (recommended for development)**

1. **Start monitoring stack:**
   ```bash
   ./start-monitoring.sh
   ```

2. **Start MLflow UI (optional):**
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```

3. **Start FastAPI application:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8080
   ```

**Option 2: Single command startup**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

### Access URLs
- **🚀 Main Application**: http://localhost:8080
- **📊 Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **🔍 Prometheus Metrics**: http://localhost:9090
- **📈 MLflow Experiments**: http://localhost:5000
- **⚡ App Metrics Endpoint**: http://localhost:8001/metrics

### What Each Service Does
- **FastAPI App** (port 8080): Your adversarial prompt detector with Gradio UI
- **Grafana** (port 3000): Real-time model performance dashboards
- **Prometheus** (port 9090): Metrics collection and storage
- **MLflow** (port 5000): Experiment tracking and model management
- **Metrics Server** (port 8001): Prometheus metrics from your models


## Monitoring & Observability

This project includes comprehensive monitoring with Grafana and Prometheus:

- **Real-time Performance Dashboards**: Track model latency, throughput, and prediction rates
- **MLflow Integration**: Complete experiment tracking with nested runs for each model
- **Prometheus Metrics**: Custom metrics for each model (electra_small, tox_bert, offensive_roberta, bart_mnli)

See `MONITORING.md` for detailed setup instructions.

### Dashboard Management

Your Grafana dashboard configuration is stored in `grafana/dashboards/adversarial-detector-dashboard.json`. To update the dashboard after making changes to the JSON file:

#### 🚀 Quick Update (Recommended)
```bash
./update-dashboard.sh
```

#### 📋 Alternative Methods

**Manual API Update:**
```bash
jq '{dashboard: ., overwrite: true}' grafana/dashboards/adversarial-detector-dashboard.json | \
curl -X POST -H "Content-Type: application/json" -u admin:admin123 \
  -d @- http://localhost:3000/api/dashboards/db
```

**Restart Grafana Container:**
```bash
docker restart grafana
```

**Grafana UI Import:**
1. Go to http://localhost:3000
2. **+ → Import**
3. **Upload JSON file** or paste JSON content
4. **Load** and **Import**

#### 🎯 Dashboard Update Workflow
1. **Edit JSON**: Make changes to `grafana/dashboards/adversarial-detector-dashboard.json`
2. **Run script**: `./update-dashboard.sh`
3. **View changes**: Go to http://localhost:3000/d/adversarial-prompt-detector
4. **Refresh page** if needed

Your dashboard includes:
- Total predictions by model
- 95th percentile latency by model  
- Model latency percentiles over time
- Average response time by model
- Request rate by model (req/sec)

## Troubleshooting

### Port Issues
If you get "Address already in use" errors:
```bash
# Kill processes on specific ports
lsof -ti:8080 | xargs kill -9  # FastAPI
lsof -ti:8001 | xargs kill -9  # Metrics server
lsof -ti:5000 | xargs kill -9  # MLflow
```

### Stop All Services
```bash
# Stop monitoring stack
docker compose -f docker-compose.monitoring.yml down

# Kill Python processes
pkill -f uvicorn
pkill -f mlflow
```

## Run Application Docker 

### 1. Build the Docker image

```bash
docker build \
  --file Dockerfile \
  --tag prompt-detector-app:latest \
  . ```
 
### 2. Run the Docker Image
```bash
docker run -d \
  --name prompt-detector-server \
  -e MODEL_PATH=outputs/electra/best_model \
  -p 8080:80 \
  prompt-detector-app:latest
```

## References 

Red Teaming Large Language Models to Reduce Harms (`https://arxiv.org/abs/2209.07858`)

Inverse Reinforcement Learning (`https://arxiv.org/abs/2402.01886z`)

Adversarial Defense Heuristics (`https://arxiv.org/abs/2307.15043`)