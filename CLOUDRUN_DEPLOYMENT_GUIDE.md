# 🚀 Google Cloud Run Deployment Guide

## 💰 Why Cloud Run vs BentoML?

| Feature | BentoML Cloud | Google Cloud Run |
|---------|---------------|------------------|
| **Monthly Cost** | $50-150 | $5-25 |
| **Scaling** | Always-on instances | Scale to zero |
| **Cold Start** | ~2-3 seconds | ~3-5 seconds |
| **Free Tier** | Limited | 2M requests/month FREE |
| **Monitoring** | Basic | Full Google Cloud suite |
| **Flexibility** | BentoML specific | Any containerized app |

**Cloud Run saves 80-90% on costs** because you only pay for actual requests!

## 🏗️ Architecture

```
GitHub → Cloud Build → Container Registry → Cloud Run
   ↓
Your App: FastAPI + Gradio + ML Models
   ↓
Auto-scaling: 0-10 instances based on traffic
```

## 📦 What We've Created

### 1. **Dockerfile.cloudrun** - Optimized Container
- Python 3.10 slim base (faster startup)
- Multi-layer caching for fast rebuilds
- Health checks built-in
- Production environment variables

### 2. **cloudbuild.yaml** - Automated CI/CD
- Builds Docker image
- Pushes to Container Registry  
- Deploys to Cloud Run
- Zero-downtime deployments

### 3. **deploy_cloudrun.sh** - Deployment Script
- Interactive setup
- Automated API enablement
- Multiple deployment options
- Monitoring and logging tools

## 🚀 Quick Start

### Prerequisites
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login to Google Cloud
gcloud auth login

# Install Docker (if deploying manually)
# Follow: https://docs.docker.com/get-docker/
```

### Option 1: Automated Deployment (Recommended)
```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"

# Complete setup and deployment
./deploy_cloudrun.sh all
```

### Option 2: Step-by-Step
```bash
# 1. Setup project and APIs
./deploy_cloudrun.sh setup

# 2. Deploy application
./deploy_cloudrun.sh deploy

# 3. Setup monitoring
./deploy_cloudrun.sh monitoring
```

### Option 3: Manual Deployment
```bash
# For more control over the process
./deploy_cloudrun.sh deploy-manual
```

## 🌐 Your Deployed Application

After deployment, you'll get URLs like:
```
Base URL: https://adversarial-prompt-detector-xxx-uc.a.run.app

🤖 Chat Interface: /chat
🔍 Health Check: /health  
📊 Metrics: /metrics
📖 API Documentation: /
```

## 📊 Cost Optimization Features

### Automatic Scaling
- **Scale to Zero**: No idle costs when not used
- **Max Instances**: 10 (prevents runaway costs)
- **Concurrency**: 100 requests per instance
- **CPU/Memory**: Right-sized for ML workloads

### Resource Configuration
```yaml
CPU: 2 cores          # Optimal for ML inference
Memory: 4GB           # Handles model loading
Timeout: 300s         # For slower model operations
Min Instances: 0      # Scale to zero for cost savings
```

## 🔧 Configuration Options

### Environment Variables
```bash
TOKENIZERS_PARALLELISM=false      # Prevent tokenizer warnings
PYTORCH_ENABLE_MPS_FALLBACK=1     # GPU fallback
HF_HUB_DISABLE_SYMLINKS_WARNING=1 # Reduce warnings
PORT=8080                         # Cloud Run port
```

### Custom Deployment
Modify `cloudbuild.yaml` for custom settings:
```yaml
# Example: Change region
substitutions:
  _REGION: 'us-west1'  # or europe-west1, asia-east1

# Example: Add environment variables
--set-env-vars: 'CUSTOM_VAR=value,ANOTHER_VAR=value'
```

## 📈 Monitoring & Logging

### Built-in Google Cloud Monitoring
```bash
# View logs
./deploy_cloudrun.sh logs

# Check service status
./deploy_cloudrun.sh status

# Google Cloud Console
# https://console.cloud.google.com/run
```

### Application Metrics
Your app automatically exports:
- **Request metrics**: Latency, error rates
- **Custom metrics**: Model inference times, detection rates
- **Resource metrics**: CPU, memory usage
- **Health metrics**: Uptime, availability

### Alerting Setup
```bash
# Create uptime checks
gcloud alpha monitoring uptime create \
    --display-name="Adversarial Detector Uptime" \
    --http-check-path="/health"
```

## 🚀 CI/CD Pipeline

### Automatic Deployments
1. **Push to GitHub** → Triggers Cloud Build
2. **Cloud Build** → Builds + Tests + Deploys
3. **Cloud Run** → Zero-downtime deployment

### Setup GitHub Integration
```bash
# Connect GitHub repository
gcloud alpha builds triggers create github \
    --repo-name="LLM-Adversarial-Prompt-Detector" \
    --repo-owner="Andre-Williams22" \
    --branch-pattern="^main$" \
    --build-config="cloudbuild.yaml"
```

## 💵 Cost Estimation

### Free Tier (Perfect for testing)
- **2 million requests/month FREE**
- **360,000 GB-seconds/month FREE**
- **180,000 vCPU-seconds/month FREE**

### Production Usage
**Light usage** (1000 requests/day):
- Cost: ~$3-5/month
- Mostly free tier

**Medium usage** (10,000 requests/day):
- Cost: ~$15-25/month
- Still very affordable

**Heavy usage** (100,000 requests/day):
- Cost: ~$50-80/month
- Still cheaper than BentoML!

## 🔒 Security Features

### Built-in Security
- **HTTPS**: Automatic SSL certificates
- **IAM**: Fine-grained access control
- **VPC**: Network isolation
- **Secrets**: Secure environment variables

### Authentication (Optional)
```bash
# Require authentication
gcloud run services update adversarial-prompt-detector \
    --region=us-central1 \
    --clear-env-vars \
    --no-allow-unauthenticated
```

## 🛠️ Maintenance & Updates

### Zero-Downtime Updates
```bash
# Deploy new version
./deploy_cloudrun.sh deploy

# Automatic traffic shifting
# Old version → New version gradually
```

### Rollback if Needed
```bash
# List revisions
gcloud run revisions list \
    --service=adversarial-prompt-detector \
    --region=us-central1

# Rollback to previous version
gcloud run services update-traffic adversarial-prompt-detector \
    --region=us-central1 \
    --to-revisions=REVISION-NAME=100
```

## 🆘 Troubleshooting

### Common Issues

1. **Cold Start Latency**
   - Solution: Set `--min-instances=1` for critical apps
   - Cost: ~$30/month for always-warm

2. **Memory Issues**
   - Increase memory: `--memory=6Gi`
   - Monitor usage in Cloud Console

3. **Build Timeouts**
   - Increase timeout in `cloudbuild.yaml`
   - Use Cloud Build's high-CPU machines

### Debug Commands
```bash
# Check logs
./deploy_cloudrun.sh logs

# Test locally
docker build -f Dockerfile.cloudrun -t test .
docker run -p 8080:8080 test

# Check service details
gcloud run services describe adversarial-prompt-detector \
    --region=us-central1
```

## 🎯 Production Checklist

### Before Going Live
- [ ] Test all endpoints work
- [ ] Verify model loading in cloud environment
- [ ] Setup monitoring alerts
- [ ] Configure custom domain (optional)
- [ ] Setup backup/disaster recovery
- [ ] Review security settings

### Post-Deployment
- [ ] Monitor costs in billing dashboard
- [ ] Check error rates and latencies
- [ ] Setup log-based metrics
- [ ] Configure auto-scaling thresholds

## 🔄 Migration from Current Setup

### Advantages over CapRover
1. **No server management** - Google handles everything
2. **Better scaling** - Automatic based on traffic
3. **Cost efficiency** - Pay per use vs fixed costs
4. **Reliability** - Google's global infrastructure
5. **Monitoring** - Built-in observability

### Migration Steps
1. Deploy to Cloud Run (parallel to existing)
2. Test thoroughly with real traffic
3. Update DNS to point to Cloud Run
4. Decommission CapRover setup

Ready to deploy? Start with: `./deploy_cloudrun.sh all`

Your adversarial prompt detector will be running on Google's global infrastructure at a fraction of the cost! 🚀
